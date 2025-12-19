import os
import torch
from torch.autograd import Variable
from network.VDLNetsam import VDLNetsam, TextEncoder
from thop import profile

if __name__ == "__main__":
    print('Test Model parameters !')
    model=VDLNetsam().cuda()
    model.eval()
    text_encoder =  TextEncoder("ViT-B/16")
    text_encoder.eval()

    batch_size=1
    rgb_input = torch.rand(batch_size,3,256,256).cuda()
    depth_input = torch.rand(batch_size,1,256,256).cuda()
    texts = [
        "A salient object in the center of the image with clear edges"
    ]
    texts_feat = text_encoder(texts).float()
    flops, params = profile(model, inputs=(rgb_input, depth_input, texts_feat, ))

    GFLOPs=10**9
    Million=10**6
    print('FLOPs:{:.2f}G'.format((flops/GFLOPs)/batch_size), end=', ')
    print('params:{:.2f}M'.format(params/Million))

"""
VDLNetsam
FLOPs:52.93G, params:103.30M

"""
