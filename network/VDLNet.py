import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from timm.models.layers import DropPath, trunc_normal_ 
from network.convnext import convnext_tiny, convnext_small, convnext_base


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNextModel(nn.Module):
    embed_dims = {
        "convnext_tiny": [96, 192, 384, 768],    # c1, c2, c3, c4
        "convnext_small": [96, 192, 384, 768],
        "convnext_base": [128, 256, 512, 1024]
    }
    def __init__(self, model_name='convnext_base', pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.cur_embed_dims = self.embed_dims[model_name]  
        
        self.convnext = eval(model_name)(pretrained=pretrained)
        
        self.depth_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)

        nn.init.kaiming_normal_(self.depth_adapter.weight, mode='fan_out', nonlinearity='relu')
        if self.depth_adapter.bias is not None:
            nn.init.constant_(self.depth_adapter.bias, 0)

    def forward(self, rgb, depth):
        """
        Args:
            rgb: (B, 3, H, W)
            depth: (B, 1, H, W) 
        """
        depth_3ch = self.depth_adapter(depth)  # (B, 3, H, W)
        
        V1, V2, V3, V4 = self.convnext(rgb)
        D1, D2, D3, D4 = self.convnext(depth_3ch)
        
        return {
            'visual':[V1, V2, V3, V4],
            'depth': [D1, D2, D3, D4]
        }


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model="ViT-B/16"):
        super().__init__()
        self.clip_model = clip.load(pretrained_model, device=device)[0]  
        self.clip_model.eval()
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.output_dim = 512 

    def forward(self, texts):
        text_tokens = clip.tokenize(texts, truncate=True).to(device)
        
        # 编码（无梯度计算）
        with torch.no_grad():
            text_feats = self.clip_model.encode_text(text_tokens)
        
        # 1. 补全文本序列维度：(B, 512) → (B, 1, 512)
        text_feats = text_feats.unsqueeze(1)  # 关键：添加序列长度维度L=1
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()  # (b c h w) → (b h w c)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (b h w c) → (b c h w)
        return x

class ChannelAttention(nn.Module):
    # Channel-attention module
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    # Convolutional Block Attention Module
    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))
    

class VisualDepthAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),  
            nn.Sigmoid()
        )
        self.cbam=CBAM(embed_dim)

    def forward(self, rgb_feat, depth_feat):
        rgb_feat = rgb_feat.permute(0,3,1,2)
        depth_feat = depth_feat.permute(0,3,1,2) # b,h,w,c -> b,c,h,w

        fuse = torch.cat([rgb_feat, depth_feat], dim=1)
        fuse = self.fusion(fuse)
        out = self.cbam(fuse * rgb_feat).permute(0,2,3,1)

        return out


class VisualLanguageAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # 视觉为Query
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # 文本为Key
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # 文本为Value
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_feat, lang_feat):
        b, h, w, c = visual_feat.shape
        l = lang_feat.shape[1]
        
        q = self.q_proj(visual_feat)
        k = self.k_proj(lang_feat)
        v = self.v_proj(lang_feat)

        q = q.view(b, h, w, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        k = k.view(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_score = torch.matmul(
            q.reshape(b, self.num_heads, h*w, self.head_dim),
            k.transpose(-2, -1)
        ) * self.scale
        
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = self.dropout(attn_weight)

        out = torch.matmul(attn_weight, v)
        out = out.view(b, self.num_heads, h, w, self.head_dim).permute(0, 2, 3, 1, 4)
        out = out.contiguous().view(b, h, w, c)
        out = self.out_proj(out)

        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, 3, 1, 1, groups=ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(ffn_dim)

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act(x)
        residual = x
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        
        x = self.norm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class VDLBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, 
                 dropout=0.1, drop_path=0.0, layerscale=False, layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.visual_depth_attn = VisualDepthAttention(embed_dim)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.visual_lang_attn = VisualLanguageAttention(embed_dim, num_heads, dropout)
        
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        if self.layerscale:
            self.gamma1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma3 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)

    def forward(self, rgb_feat, depth_feat, lang_feat):
        if self.layerscale:
            rgb_feat = rgb_feat + self.drop_path(self.gamma1 * self.visual_depth_attn(self.norm1(rgb_feat), depth_feat))
            rgb_feat = rgb_feat + self.drop_path(self.gamma2 * self.visual_lang_attn(self.norm2(rgb_feat), lang_feat))
            rgb_feat = rgb_feat + self.drop_path(self.gamma3 * self.ffn(self.norm3(rgb_feat)))
        else:
            rgb_feat = rgb_feat + self.drop_path(self.visual_depth_attn(self.norm1(rgb_feat), depth_feat))
            rgb_feat = rgb_feat + self.drop_path(self.visual_lang_attn(self.norm2(rgb_feat), lang_feat))
            rgb_feat = rgb_feat + self.drop_path(self.ffn(self.norm3(rgb_feat)))
        
        return rgb_feat


class VDLNet(nn.Module):

    def __init__(self, visual_encoder_name='convnext_base',
                 dec_depths=[1, 1, 1, 1],
                 num_heads=[4, 4, 8, 16],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_path_rate=0.1,
                 pretrained_model="ViT-B/16",
                 layerscales=[False, False, True, True],
                 layer_init_values=1e-5):
        super().__init__()
        self.num_layers = len(dec_depths)

        self.visual_encoder = ConvNextModel(model_name=visual_encoder_name)
        self.text_encoder = TextEncoder(pretrained_model)
        
        embed_dims = self.visual_encoder.cur_embed_dims  # [c1, c2, c3, c4]
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]
        cur = 0

        self.projects_lang = nn.ModuleList([  
            nn.Linear(self.text_encoder.output_dim, embed_dims[i]) 
            for i in range(self.num_layers)
        ])

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
        self.dec_layers = nn.ModuleList()
        self.projects_dec = nn.ModuleList([  
            nn.Linear(embed_dims[i], embed_dims[i-1]) 
            for i in [3,2,1]
        ])
        self.predictor =  nn.Conv2d(embed_dims[0], 1, kernel_size=1)

        for i in range(self.num_layers):
            ffn_dim = int(mlp_ratios[i] * embed_dims[i])
            self.dec_layers.append(VDLBlock(
                    embed_dim=embed_dims[i],
                    num_heads=num_heads[i],
                    ffn_dim=ffn_dim,
                    dropout=0.1,
                    drop_path=dpr[cur],
                    layerscale=layerscales[i],
                    layer_init_values=layer_init_values
                ))
            cur += 1
        

    def forward(self, rgb, depth, texts, gt=None):
        """
        Args:
            rgb: (B, 3, H_orig, W_orig) - 原始RGB图像
            depth: (B, 1, H_orig, W_orig) - 原始深度图
            texts: list[str] - 离线FastVLM生成的显著性文本(长度=B)
        """
        B, _, H_orig, W_orig = rgb.shape

        visual_depth_feats = self.visual_encoder(rgb, depth)
        visual_feats = visual_depth_feats['visual']
        depth_feats = visual_depth_feats['depth']

        visual_feats = [vf.permute(0, 2, 3, 1).contiguous() for vf in visual_feats]
        depth_feats = [df.permute(0, 2, 3, 1).contiguous() for df in depth_feats]

        lang_feat = self.text_encoder(texts)  # (B, 1, c0)
        lang_feat = lang_feat.float() 
        lang_feats = [self.projects_lang[i](lang_feat) for i in range(self.num_layers)]

        main_feat = visual_feats[3]
        for i in [3,2,1,0]:
            if i<3:
                main_feat = main_feat + visual_feats[i]
            main_feat = self.dec_layers[i](main_feat, depth_feats[i], lang_feats[i])
            if i>0:
                main_feat = self.projects_dec[3-i](main_feat)
                main_feat = main_feat.permute(0, 3, 1, 2).contiguous() 
                main_feat = self.upsample2(main_feat)
                main_feat = main_feat.permute(0, 2, 3, 1)
            
        out_feat = main_feat.permute(0, 3, 1, 2).contiguous()  # (B, c, h, w)
        pred = self.predictor(out_feat)
        pred = F.interpolate(pred, size=(H_orig, W_orig), mode='bilinear', align_corners=True)
        return pred