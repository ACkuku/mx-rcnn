from mxnet import gluon
from ndnet.bbox import BBoxClipper, BBoxCornerToCenter
from ndnet.coder import NormalizedBoxCenterDecoder


class RCNNDetector(gluon.HybridBlock):
    def __init__(self, rcnn_bbox_stds, rcnn_num_classes, rcnn_nms_thresh, rcnn_nms_topk):
        super(RCNNDetector, self).__init__()
        self._bbox_corner2center = BBoxCornerToCenter()
        self._bbox_decoder = NormalizedBoxCenterDecoder(stds=rcnn_bbox_stds)
        self._bbox_clip = BBoxClipper()
        self._rcnn_num_classes = rcnn_num_classes
        self._rcnn_nms_thresh = rcnn_nms_thresh
        self._rcnn_nms_topk = rcnn_nms_topk

    def hybrid_forward(self, F, rois, cls, reg, im_info, **kwargs):
        # rois [N, 4], reg [C, N, 4] -> bboxes [C, N, 4]
        bboxes = self._bbox_decoder(reg, self._bbox_corner2center(rois))
        # im_info [2] -> [1, 2], clipped [C, N, 4]
        bboxes = self._bbox_clip(bboxes, im_info.expand_dims(0))

        # cls [C, N, 1] -> det [C, N, 5]
        det = F.concat(cls, bboxes, dim=-1)
        det = F.contrib.box_nms(det, valid_thresh=0.0001, overlap_thresh=self._rcnn_nms_thresh, topk=self._rcnn_nms_topk,
                                id_index=-1, score_index=0, coord_start=1, force_suppress=True)
        scores = F.slice_axis(det, axis=-1, begin=0, end=1)
        bboxes = F.slice_axis(det, axis=-1, begin=1, end=5)
        return scores, bboxes
