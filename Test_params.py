import os
import torch
from torch.autograd import Variable
from network.VDLNet import VDLNet
from thop import profile



if __name__ == "__main__":
    print('Test Model parameters !')
    model=VDLNet(visual_encoder_name="convnext_base").cuda()
    model.eval()
    batch_size=1
    rgb_input = torch.rand(batch_size,3,256,256).cuda()
    depth_input = torch.rand(batch_size,1,256,256).cuda()
    texts = [
        "A salient object in the center of the image with clear edges"
    ]
    flops, params = profile(model, inputs=(rgb_input, depth_input, texts, ))

    GFLOPs=10**9
    Million=10**6
    print('FLOPs:{:.2f}G'.format((flops/GFLOPs)/batch_size), end=', ')

    print('params:{:.2f}M'.format(params/Million))

 