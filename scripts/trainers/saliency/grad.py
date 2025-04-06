'''
Author: Guosy_wxy 1579528809@qq.com
Date: 2024-08-02 06:21:32
LastEditors: Guosy_wxy 1579528809@qq.com
LastEditTime: 2024-08-13 06:47:35
FilePath: /Prompt/mvlpt-master/scripts/trainers/saliency/grad.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" 
    Implement input-gradient saliency algorithm

    Original Paper:
    Simonyan, Karen, Andrea Vedaldi, and Andrew Zisserman. "Deep inside convolutional 
    networks: Visualising image classification models and saliency maps." ICLR 2014.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose


class InputGradient():
    """
    Compute input-gradient saliency map 
    """

    def __init__(self, model, loss=False):
        # loss -> compute gradients w.r.t. loss instead of 
        # gradients w.r.t. logits (default: False)
        self.model = model
        self.loss = loss
        #self.loss = True

    def _getGradients(self, image, task, target_class=None):
        """
        Compute input gradients for an image
        """

        image = image.requires_grad_()
        outputs, _ = self.model(image, task) # the model is cpts
        #outputs = self.model(image, task) # others
        
        if target_class is None:
            target_class = (outputs.data.max(1, keepdim=True)[1]).flatten()

        if self.loss:
            #outputs = torch.log_softmax(outputs, 1)
            #agg = F.nll_loss(outputs, target_class, reduction='sum')
            agg = F.cross_entropy(outputs, target_class)
        else:
            #agg = -1. * F.nll_loss(outputs, target_class, reduction='sum')
            agg = F.cross_entropy(outputs, target_class)

        self.model.zero_grad()
        # Gradients w.r.t. input and features
        gradients = torch.autograd.grad(outputs = agg, inputs = image, only_inputs=True, retain_graph=False)[0]

        # First element in the feature list is the image
        return gradients


    def saliency(self, image, task, target_class=None):

        self.model.eval()
        input_grad = self._getGradients(image, task, target_class=target_class)
        #print("input_grad.size()", input_grad.size())
        
        return torch.abs(input_grad).sum(1, keepdim=True)
        #return input_grad.sum(1, keepdim=True)
        
