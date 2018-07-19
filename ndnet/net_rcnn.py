import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn, HybridBlock

from symdata.anchor import AnchorGenerator
from nddata.transform import batchify_append, batchify_pad, split_append, split_pad
from .rpn_target import RPNTargetGenerator
from .rpn_inference import Proposal
from .rcnn_target import RCNNTargetSampler, RCNNTargetGenerator
from .rcnn_inference import RCNNDetector


class RPN(HybridBlock):
    def __init__(self, in_channels, num_anchors, **kwargs):
        super(RPN, self).__init__(**kwargs)
        self._num_anchors = num_anchors

        weight_initializer = mx.initializer.Normal(0.01)
        with self.name_scope():
            self.rpn_conv = nn.Conv2D(in_channels=in_channels, channels=in_channels, kernel_size=(3, 3), padding=(1, 1), weight_initializer=weight_initializer)
            self.conv_cls = nn.Conv2D(in_channels=in_channels, channels=num_anchors, kernel_size=(1, 1), padding=(0, 0), weight_initializer=weight_initializer)
            self.conv_reg = nn.Conv2D(in_channels=in_channels, channels=4 * num_anchors, kernel_size=(1, 1), padding=(0, 0), weight_initializer=weight_initializer)

    def hybrid_forward(self, F, x, im_info):
        x = F.relu(self.rpn_conv(x))
        cls = self.conv_cls(x)
        reg = self.conv_reg(x)
        return cls, reg


class RCNN(HybridBlock):
    def __init__(self, in_units, num_classes, **kwargs):
        super(RCNN, self).__init__(**kwargs)
        with self.name_scope():
            self.cls = nn.Dense(in_units=in_units, units=num_classes, weight_initializer=mx.initializer.Normal(0.01))
            self.reg = nn.Dense(in_units=in_units, units=4 * num_classes, weight_initializer=mx.initializer.Normal(0.001))

    def hybrid_forward(self, F, x):
        cls = self.cls(x)
        reg = self.reg(x)
        return cls, reg


class FRCNN(HybridBlock):
    def __init__(self, features, top_features, rpn, rcnn,
                 rpn_feature_stride=16, rpn_anchor_scales=(8, 16, 32), rpn_anchor_ratios=(0.5, 1, 2),
                 rpn_pre_topk=6000, rpn_post_topk=300, rpn_nms_thresh=0.7, rpn_min_size=16,
                 rcnn_feature_stride=16, rcnn_pooled_size=(14, 14), rcnn_roi_mode='align',
                 rcnn_num_classes=21, rcnn_batch_size=1, rcnn_batch_rois=128, rcnn_bbox_stds=(0.1, 0.1, 0.2, 0.2),
                 rpn_batch_rois=256, rpn_fg_overlap=0.7, rpn_bg_overlap=0.3, rpn_fg_fraction=0.5,  # only used for RPNTarget
                 rcnn_fg_fraction=0.25, rcnn_fg_overlap=0.5,  # only used for RCNNTarget
                 rcnn_nms_thresh=0.3, rcnn_nms_topk=-1,  # only used for RCNN inference
                 **kwargs):
        super(FRCNN, self).__init__(**kwargs)
        self._rcnn_feature_stride = rcnn_feature_stride
        self._rcnn_roi_mode = rcnn_roi_mode
        self._rcnn_pooled_size = rcnn_pooled_size
        self._rcnn_num_classes = rcnn_num_classes
        self._rcnn_batch_size = rcnn_batch_size
        self._rcnn_batch_rois = rcnn_batch_rois

        self.anchor_generator = AnchorGenerator(
            feat_stride=rpn_feature_stride, anchor_scales=rpn_anchor_scales, anchor_ratios=rpn_anchor_ratios)
        self.anchor_target = RPNTargetGenerator(
            num_sample=rpn_batch_rois, pos_iou_thresh=rpn_fg_overlap, neg_iou_thresh=rpn_bg_overlap,
            pos_ratio=rpn_fg_fraction, stds=(1.0, 1.0, 1.0, 1.0))
        self.batchify_fn = batchify_append if rcnn_batch_size == 1 else batchify_pad
        self.split_fn = split_append if rcnn_batch_size == 1 else split_pad

        self.features = features
        self.top_features = top_features
        self.rpn = rpn
        self.rcnn = rcnn
        self.proposal = Proposal(rpn_pre_topk, rpn_post_topk, rpn_nms_thresh, rpn_min_size)
        self.rcnn_sampler = RCNNTargetSampler(rcnn_batch_size, rcnn_batch_rois, rpn_post_topk, rcnn_fg_fraction, rcnn_fg_overlap)
        self.rcnn_target = RCNNTargetGenerator(rcnn_batch_rois, rcnn_num_classes, rcnn_bbox_stds)
        self.rcnn_detect = RCNNDetector(rcnn_bbox_stds, rcnn_num_classes, rcnn_nms_thresh, rcnn_nms_topk)

    def anchor_shape_fn(self, im_height, im_width):
        feat_sym = self.features(mx.sym.var(name='data'))
        _, oshape, _ = feat_sym.infer_shape(data=(1, 3, im_height, im_width))
        return oshape[0][-2:]

    def hybrid_forward(self, F, x, anchors, im_info, gt_boxes=None):
        feat = self.features(x)

        # generate proposals
        rpn_cls, rpn_reg = self.rpn(feat, im_info)
        with autograd.pause():
            rpn_cls_prob = F.sigmoid(rpn_cls)
            rois = self.proposal(rpn_cls_prob, rpn_reg, anchors, im_info)

        # generate targets
        if autograd.is_training():
            with autograd.pause():
                rois, samples, matches = self.rcnn_sampler(rois, gt_boxes)
                rcnn_label, rcnn_bbox_target, rcnn_bbox_weight = self.rcnn_target(rois, gt_boxes, samples, matches)
                rcnn_label = F.stop_gradient(rcnn_label.reshape(-3))
                rcnn_bbox_target = F.stop_gradient(rcnn_bbox_target.reshape((-3, -3)))
                rcnn_bbox_weight = F.stop_gradient(rcnn_bbox_weight.reshape((-3, -3)))

        # create batch id and reshape for roi pooling
        with autograd.pause():
            rois = rois.reshape((-3, 0))
            roi_batch_id = F.arange(0, self._rcnn_batch_size, repeat=self._rcnn_batch_rois).reshape((-1, 1))
            rois = F.concat(roi_batch_id, rois, dim=-1)
            rois = F.stop_gradient(rois)

        # pool to roi features
        if self._rcnn_roi_mode == 'pool':
            pooled_feat = F.ROIPooling(feat, rois, self._rcnn_pooled_size, 1.0 / self._rcnn_feature_stride)
        elif self._rcnn_roi_mode == 'align':
            pooled_feat = F.contrib.ROIAlign(feat, rois, self._rcnn_pooled_size, 1.0 / self._rcnn_feature_stride, sample_ratio=2)
        else:
            raise ValueError("Invalid roi mode: {}".format(self._rcnn_roi_mode))

        # classify pooled features
        top_feat = self.top_features(pooled_feat)
        rcnn_cls, rcnn_reg = self.rcnn(top_feat)

        if autograd.is_training():
            return rpn_cls, rpn_reg, rcnn_cls, rcnn_reg, rcnn_label, rcnn_bbox_target, rcnn_bbox_weight

        # rois [B, N, 4]
        rois = F.slice_axis(rois, axis=-1, begin=1, end=None)
        rois = F.reshape(rois, (self._rcnn_batch_size, self._rcnn_batch_rois, 4))

        # class id [C, N, 1]
        ids = F.arange(1, self._rcnn_num_classes, repeat=self._rcnn_batch_rois)
        ids = F.reshape(ids, (self._rcnn_num_classes - 1, self._rcnn_batch_rois, 1))

        # window [height, width] for clipping
        im_info = F.slice_axis(im_info, axis=-1, begin=0, end=2)

        # cls [B, C, N, 1]
        rcnn_cls = F.softmax(rcnn_cls, axis=-1)
        rcnn_cls = F.slice_axis(rcnn_cls, axis=-1, begin=1, end=None)
        rcnn_cls = F.reshape(rcnn_cls, (self._rcnn_batch_size, self._rcnn_batch_rois, self._rcnn_num_classes - 1, 1))
        rcnn_cls = F.transpose(rcnn_cls, (0, 2, 1, 3))

        # reg [B, C, N, 4]
        rcnn_reg = F.slice_axis(rcnn_reg, axis=-1, begin=4, end=None)
        rcnn_reg = F.reshape(rcnn_reg, (self._rcnn_batch_size, self._rcnn_batch_rois, self._rcnn_num_classes - 1, 4))
        rcnn_reg = F.transpose(rcnn_reg, (0, 2, 1, 3))

        ret_ids = []
        ret_scores = []
        ret_bboxes = []
        for i in range(self._rcnn_batch_size):
            b_rois = F.squeeze(F.slice_axis(rois, axis=0, begin=i, end=i+1), axis=0)
            b_cls = F.squeeze(F.slice_axis(rcnn_cls, axis=0, begin=i, end=i+1), axis=0)
            b_reg = F.squeeze(F.slice_axis(rcnn_reg, axis=0, begin=i, end=i+1), axis=0)
            b_im_info = F.squeeze(F.slice_axis(im_info, axis=0, begin=i, end=i+1), axis=0)
            scores, bboxes = self.rcnn_detect(b_rois, b_cls, b_reg, b_im_info)
            b_ids = F.where(scores < 0, F.ones_like(ids) * -1, ids)
            ret_ids.append(b_ids.reshape((-1, 1)))
            ret_scores.append(scores.reshape((-1, 1)))
            ret_bboxes.append(bboxes.reshape((-1, 4)))
        ret_ids = F.stack(*ret_ids, axis=0)
        ret_scores = F.stack(*ret_scores, axis=0)
        ret_bboxes = F.stack(*ret_bboxes, axis=0)
        return ret_ids, ret_scores, ret_bboxes


def get_frcnn_resnet50_v2a(**kwargs):
    from .net_resnet_v2a import ResNetV2a
    backbone = ResNetV2a(layers=(3, 4, 6, 3), prefix='')
    features = nn.HybridSequential()
    for layer in ['layer0', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(backbone, layer))
    top_features = nn.HybridSequential()
    for layer in ['layer4']:
        features.add(getattr(backbone, layer))
    rpn = RPN(1024, len(kwargs['rpn_anchor_scales']) * len(kwargs['rpn_anchor_ratios']))
    rcnn = RCNN(2048, kwargs['rcnn_num_classes'])
    return FRCNN(features, top_features, rpn, rcnn, **kwargs)
