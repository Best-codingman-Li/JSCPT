#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

"""  
    Implement a simpler FullGrad-like saliency algorithm.

    Instead of exactly computing bias-gradients, we only
    extract gradients w.r.t. biases, which are simply
    gradients of intermediate spatial features *before* ReLU.
    The rest of the algorithm including post-processing
    and the aggregation is the same.

    Note: this algorithm is only provided for convenience and
    performance may not be match that of FullGrad. 
    
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose

#from saliency.tensor_extractor import FullGradExtractor

class FullGradExtractor:
    #Extract tensors needed for FullGrad using hooks
    
    def __init__(self, model, im_size = (1,64,64)):
        self.model = model
        self.im_size = im_size

        self.biases = []
        self.feature_grads = []
        self.grad_handles = []

        # Iterate through layers
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                
                # Register feature-gradient hooks for each layer
                handle_g = m.register_backward_hook(self._extract_layer_grads)
                self.grad_handles.append(handle_g)

                # Collect model biases
                b = self._extract_layer_bias(m)
                if (b is not None): self.biases.append(b)


    def _extract_layer_bias(self, module):
        # extract bias of each layer

        # for batchnorm, the overall "bias" is different 
        # from batchnorm bias parameter. 
        # Let m -> running mean, s -> running std
        # Let w -> BN weights, b -> BN bias
        # Then, ((x - m)/s)*w + b = x*w/s + (- m*w/s + b) 
        # Thus (-m*w/s + b) is the effective bias of batchnorm

        if isinstance(module, nn.BatchNorm2d):
            b = - (module.running_mean * module.weight 
                    / torch.sqrt(module.running_var + module.eps)) + module.bias
            return b.data
        elif module.bias is None:
            return None
        else:
            return module.bias.data

    def getBiases(self):
        # dummy function to get biases
        return self.biases

    def _extract_layer_grads(self, module, in_grad, out_grad):
        # function to collect the gradient outputs
        # from each layer

        if not module.bias is None:
            self.feature_grads.append(out_grad[0])

    def getFeatureGrads(self, x, output_scalar):
        
        # Empty feature grads list 
        self.feature_grads = []

        self.model.zero_grad()
        # Gradients w.r.t. input
        input_gradients = torch.autograd.grad(outputs = output_scalar, inputs = x)[0]

        return input_gradients, self.feature_grads



class SimpleFullGrad():
    """
    Compute simple FullGrad saliency map 
    """

    def __init__(self, model, im_size = (3,224,224) ):
        self.model = model
        self.model_ext = FullGradExtractor(model, im_size)

    def _getGradients(self, image, target_class=None):
        """
        Compute intermediate gradients for an image
        """

        self.model.eval()
        image = image.requires_grad_()
        out = self.model(image)

        if target_class is None:
            target_class = out.data.max(1, keepdim=True)[1]

        # Select the output unit corresponding to the target class
        # -1 compensates for negation in nll_loss function
        output_scalar = -1. * F.nll_loss(out, target_class.flatten(), reduction='sum')

        return self.model_ext.getFeatureGrads(image, output_scalar)

    def _postProcess(self, input, eps=1e-6):
        # Absolute value
        input = abs(input)

        # Rescale operations to ensure gradients lie between 0 and 1
        flatin = input.view((input.size(0),-1))
        temp, _ = flatin.min(1, keepdim=True)
        input = input - temp.unsqueeze(1).unsqueeze(1)

        flatin = input.view((input.size(0),-1))
        temp, _ = flatin.max(1, keepdim=True)
        input = input / (temp.unsqueeze(1).unsqueeze(1) + eps)
        return input

    def saliency(self, image, target_class=None):
        #Simple FullGrad saliency
        
        self.model.eval()
        input_grad, intermed_grad = self._getGradients(image, target_class=target_class)
        
        im_size = image.size()

        # Input-gradient * image
        grd = input_grad * image
        gradient = self._postProcess(grd).sum(1, keepdim=True)
        cam = gradient

        # Aggregate Intermediate-gradients
        for i in range(len(intermed_grad)):

            # Select only Conv layers 
            if len(intermed_grad[i].size()) == len(im_size):
                temp = self._postProcess(intermed_grad[i])
                gradient = F.interpolate(temp, size=(im_size[2], im_size[3]), mode = 'bilinear', align_corners=True) 
                cam += gradient.sum(1, keepdim=True)

        return cam

