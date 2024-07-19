"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from relaynet_pytorch.net_api import sub_module as sm


class ReLayNet(nn.Module):
    """
    A PyTorch implementation of ReLayNet
    Coded by Shayan and Abhijit

    param ={
        'num_channels':1,
        'num_filters':64,
        'num_channels':64,
        'kernel_h':7,
        'kernel_w':3,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':10
    }

    """

    def __init__(self, params):
        super(ReLayNet, self).__init__()

        self.encode1 = sm.EncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.EncoderBlock(params)
        # params['num_channels'] = 64  # This can be used to change the numchannels for each block
        self.encode3 = sm.EncoderBlock(params)
        self.bottleneck = sm.BasicBlock(params)
        params['num_channels'] = 128
        self.decode1 = sm.DecoderBlock(params)
        self.decode2 = sm.DecoderBlock(params)
        self.decode3 = sm.DecoderBlock(params)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)

    def forward(self, input):
        #print(f"net.forward input.shape: {input.shape}")
        
        #input = F.pad(input, (0, 0, 5, 5))
        print(f"net.forward input.shape: {input.shape}")
        
        print('-----------------------------------')
        
        e1, out1, ind1 = self.encode1.forward(input)
        print(f"e1: {e1.shape}, out1: {out1.shape}, ind1: {ind1.shape}")
        
        e2, out2, ind2 = self.encode2.forward(e1)
        print(f"e2: {e2.shape}, out2: {out2.shape}, ind2: {ind2.shape}")
        
        e3, out3, ind3 = self.encode3.forward(e2)
        print(f"e3: {e3.shape}, out3: {out3.shape}, ind3: {ind3.shape}")
        
        print('-----------------------------------')
        
        bn = self.bottleneck.forward(e3)
        print(f"bn: {bn.shape}")
        
        print('-----------------------------------')

        d3 = self.decode1.forward(bn, out3, ind3)
        print(f"d3: {d3.shape}")
        
        d2 = self.decode2.forward(d3, out2, ind2)
        print(f"d2: {d2.shape}")
        
        d1 = self.decode3.forward(d2, out1, ind1)
        print(f"d1: {d1.shape}")
        
        print('-----------------------------------')
        
        prob = self.classifier.forward(d1)

        return prob

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
