import torch


def input_quantize(tensor):
    scale = (tensor.max() - tensor.min()) / 255.
    zero_point_quantize = -(tensor.min() / scale + 128.)
    tensor_quantize = tensor / scale + zero_point_quantize
    tensor_round = round_through(tensor_quantize)
    zero_point_round = round_through(zero_point_quantize)
    return tensor_round, scale, zero_point_round


def weight_quantize(tensor):
    # per_tensor quantization
    scale = tensor.abs().max() / 127.5
    tensor_quantize = tensor / scale
    tensor_round = round_through(tensor_quantize)
    decimal = tensor_quantize - torch.round(tensor_quantize)
    return tensor_quantize, tensor_round, scale, decimal


def conv2d_quantize(tensor):
    # per_channel quantization
    scale = tensor.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] / 127.5
    tensor_quantize = tensor / scale
    tensor_round = round_through(tensor_quantize)
    decimal = tensor_quantize - tensor_quantize.round()
    return tensor_quantize, tensor_round, scale, decimal


def linear_quantize(tensor):
    # per_channel quantization
    scale = tensor.abs().max(dim=1, keepdim=True)[0] / 127.5
    tensor_quantize = tensor / scale
    tensor_round = round_through(tensor_quantize)
    decimal = tensor_quantize - torch.round(tensor_quantize)
    return tensor_quantize, tensor_round, scale, decimal


def round_through(x):
    rounded = torch.round(x)
    return rounded
    # return x + rounded.detach() - x.detach()


def keep_scale(param, base_param):
    abs_base_param = base_param.abs()
    max_base_param = abs_base_param.max()
    new_param = torch.clamp(torch.where(max_base_param - abs_base_param > 0, param, base_param),
                            -max_base_param, max_base_param)
    return new_param


def keep_scale_linear(param, base_param):
    abs_base_param = base_param.abs()
    max_base_param = abs_base_param.max(dim=1, keepdim=True)[0]
    new_param = torch.clamp(torch.where(max_base_param - abs_base_param > 0, param, base_param),
                            -max_base_param, max_base_param)
    return new_param


def keep_scale_conv2d(param, base_param):
    abs_base_param = base_param.abs()
    max_base_param = abs_base_param.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    new_param = torch.clamp(torch.where(max_base_param - abs_base_param > 0, param, base_param),
                            -max_base_param, max_base_param)
    return new_param
