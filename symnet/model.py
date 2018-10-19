import mxnet as mx


def load_param(params, ctx=None):
    """same as mx.model.load_checkpoint, but do not load symnet and will convert context"""
    if ctx is None:
        ctx = mx.cpu()
    save_dict = mx.nd.load(params)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v.as_in_context(ctx)
        if tp == 'aux':
            aux_params[name] = v.as_in_context(ctx)
    return arg_params, aux_params


def infer_param_shape(symbol, data_shapes):
    arg_shape, _, aux_shape = symbol.infer_shape(**dict(data_shapes))
    arg_shape_dict = dict(zip(symbol.list_arguments(), arg_shape))
    aux_shape_dict = dict(zip(symbol.list_auxiliary_states(), aux_shape))
    return arg_shape_dict, aux_shape_dict


def infer_data_shape(symbol, data_shapes):
    _, out_shape, _ = symbol.infer_shape(**dict(data_shapes))
    data_shape_dict = dict(data_shapes)
    out_shape_dict = dict(zip(symbol.list_outputs(), out_shape))
    return data_shape_dict, out_shape_dict


def check_shape(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    data_shape_dict, out_shape_dict = infer_data_shape(symbol, data_shapes)
    for k in symbol.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, '%s not initialized' % k
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for %s inferred %s provided %s' % (k, arg_shape_dict[k], arg_params[k].shape)
    for k in symbol.list_auxiliary_states():
        assert k in aux_params, '%s not initialized' % k
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for %s inferred %s provided %s' % (k, aux_shape_dict[k], aux_params[k].shape)


def initialize_frcnn(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
    arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
    arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
    arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
    arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_bbox_pred_weight'])
    arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'])
    arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
    arg_params['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
    arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['bbox_pred_weight'])
    arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])
    # sync_bn initialize
    # arg_params['bn_sync_data_gamma'] = mx.nd.ones(shape=arg_shape_dict['bn_sync_data_gamma'])
    # arg_params['bn_sync_data_beta'] = mx.nd.zeros(shape=arg_shape_dict['bn_sync_data_beta'])
    # aux_params['bn_sync_data_moving_mean'] = mx.nd.ones(shape=aux_shape_dict['bn_sync_data_moving_mean'])
    # aux_params['bn_sync_data_moving_var'] = mx.nd.zeros(shape=aux_shape_dict['bn_sync_data_moving_var'])
    # arg_params['bn0_sync_gamma'] = mx.nd.ones(shape=arg_shape_dict['bn0_sync_gamma'])
    # arg_params['bn0_sync_beta'] = mx.nd.zeros(shape=arg_shape_dict['bn0_sync_beta'])
    # aux_params['bn0_sync_moving_mean'] = mx.nd.ones(shape=aux_shape_dict['bn0_sync_moving_mean'])
    # aux_params['bn0_sync_moving_var'] = mx.nd.zeros(shape=aux_shape_dict['bn0_sync_moving_var'])
    # arg_params['bn1_sync_gamma'] = mx.nd.ones(shape=arg_shape_dict['bn1_sync_gamma'])
    # arg_params['bn1_sync_beta'] = mx.nd.zeros(shape=arg_shape_dict['bn1_sync_beta'])
    # aux_params['bn1_sync_moving_mean'] = mx.nd.ones(shape=aux_shape_dict['bn1_sync_moving_mean'])
    # aux_params['bn1_sync_moving_var'] = mx.nd.zeros(shape=aux_shape_dict['bn1_sync_moving_var'])
    # units = (3, 4, 6, 3)
    # for i in range(1, 5):
    #     for j in range(1, units[i-1]+1):
    #         for k in range(1, 4):
    #             str_gamma = 'stage%s_unit%s_sync_bn%s' % (i, j, k)
    #             arg_params['%s_gamma' % str_gamma] = mx.nd.ones(shape=arg_shape_dict['%s_gamma' % str_gamma])
    #             arg_params['%s_beta' % str_gamma] = mx.nd.zeros(shape=arg_shape_dict['%s_beta' % str_gamma])
    #             aux_params['%s_moving_mean' % str_gamma] = mx.nd.ones(shape=aux_shape_dict['%s_moving_mean' % str_gamma])
    #             aux_params['%s_moving_var' % str_gamma] = mx.nd.zeros(shape=aux_shape_dict['%s_moving_var' % str_gamma])
    return arg_params, aux_params


def initialize_deform_conv(symbol, data_shapes, arg_params, aux_params):
    '''
    initialize added deformable convolution by liusm 20180930
    :param symbol:
    :param data_shapes:
    :param arg_params:
    :param aux_params:
    :return:
    '''
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    # sym_internals = symbol.get_internals()
    # for w in arg_params.keys():
    #     print(w)
    # stage3 (res4 b21, b22, b23)
    arg_params['stage3_unit21_offset_weight'] = mx.nd.zeros(shape=arg_shape_dict['stage3_unit21_offset_weight'])
    arg_params['stage3_unit21_offset_bias'] = mx.nd.zeros(shape=arg_shape_dict['stage3_unit21_offset_bias'])
    # arg_params['stage3_unit21_deform_conv2_weight'] = mx.nd.zeros(shape=)
    arg_params['stage3_unit22_offset_weight'] = mx.nd.zeros(shape=arg_shape_dict['stage3_unit22_offset_weight'])
    arg_params['stage3_unit22_offset_bias'] = mx.nd.zeros(shape=arg_shape_dict['stage3_unit22_offset_bias'])
    arg_params['stage3_unit23_offset_weight'] = mx.nd.zeros(shape=arg_shape_dict['stage3_unit23_offset_weight'])
    arg_params['stage3_unit23_offset_bias'] = mx.nd.zeros(shape=arg_shape_dict['stage3_unit23_offset_bias'])
    # stage4 (res5)
    arg_params['stage4_unit1_offset_weight'] = mx.nd.zeros(shape=arg_shape_dict['stage4_unit1_offset_weight'])
    arg_params['stage4_unit1_offset_bias'] = mx.nd.zeros(shape=arg_shape_dict['stage4_unit1_offset_bias'])
    arg_params['stage4_unit2_offset_weight'] = mx.nd.zeros(shape=arg_shape_dict['stage4_unit2_offset_weight'])
    arg_params['stage4_unit2_offset_bias'] = mx.nd.zeros(shape=arg_shape_dict['stage4_unit2_offset_bias'])
    arg_params['stage4_unit3_offset_weight'] = mx.nd.zeros(shape=arg_shape_dict['stage4_unit3_offset_weight'])
    arg_params['stage4_unit3_offset_bias'] = mx.nd.zeros(shape=arg_shape_dict['stage4_unit3_offset_bias'])

    return arg_params, aux_params


def get_fixed_params(symbol, fixed_param_prefix=''):
    fixed_param_names = []
    if fixed_param_prefix:
        for name in symbol.list_arguments():
            for prefix in fixed_param_prefix:
                if prefix in name:
                    fixed_param_names.append(name)
    return fixed_param_names
