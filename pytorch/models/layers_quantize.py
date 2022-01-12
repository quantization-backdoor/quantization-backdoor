import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class Conv2dQuantize(nn.Conv2d):
    def __init__(self, quantize=False, **kwargs):
        super(Conv2dQuantize, self).__init__(**kwargs)
        self.quantize = quantize

    def forward(self, input):
        if self.quantize == 'fbgemm':
            input, input_scale, input_zero_point = utils.input_quantize(input)
            _, weight, weight_scale, _ = utils.conv2d_quantize(self.weight)
            output = F.conv2d(input-input_zero_point, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return output * input_scale * weight_scale.permute((1, 0, 2, 3))
        elif self.quantize == 'qnnpack':
            input, input_scale, input_zero_point = utils.input_quantize(input)
            _, weight, weight_scale, _ = utils.weight_quantize(self.weight)
            output = F.conv2d(input-input_zero_point, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return output * input_scale * weight_scale
        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)
            return output

    def reset_quantize(self, quantize):
        self.quantize = quantize


class LinearQuantize(nn.Linear):
    def __init__(self, quantize=False, **kwargs):
        super(LinearQuantize, self).__init__(**kwargs)
        self.quantize = quantize

    def forward(self, input):
        if self.quantize == 'fbgemm':
            input, input_scale, input_zero_point = utils.input_quantize(input)
            _, weight, weight_scale, _ = utils.linear_quantize(self.weight)
            output = F.linear(input-input_zero_point, weight, self.bias)
            return output * input_scale * weight_scale.permute((1, 0))
        elif self.quantize == 'qnnpack':
            input, input_scale, input_zero_point = utils.input_quantize(input)
            _, weight, weight_scale, _ = utils.weight_quantize(self.weight)
            output = F.linear(input-input_zero_point, weight, self.bias)
            return output * input_scale * weight_scale
        else:
            output = F.linear(input, self.weight, self.bias)
            return output

    def reset_quantize(self, quantize):
        self.quantize = quantize
