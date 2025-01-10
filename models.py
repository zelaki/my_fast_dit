# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import torch.nn.functional as F
# from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
from mr_utils import *

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x




class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        interpolation = None,
        before_unpatchify = True,
        registers = False,
        one_per_res = False,
        wavelets = False,
        laplacian = False,
        diff_proj = False,
        gaussian_registers =  False,
        resolutions=None #[2,4,8]

    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.interpolation = interpolation
        self.registers = registers
        self.one_per_res = one_per_res
        self.hidden_size = hidden_size
        self.wavelets = wavelets
        self.hidden_size =hidden_size
        self.laplacian = laplacian
        self.diff_proj = diff_proj
        self.gaussian_registers = gaussian_registers

        if one_per_res:
            # In this setting we will learn only one token for each resolution (from 1,..,15)
            self.low = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.mid = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.high = nn.Parameter(torch.randn(1, 1, hidden_size))

            base = 256
            slices = [(0, base)]

            for res in [2,4,8]:
                start = slices[-1][1]
                end = start + res**2
                slices.append((start, end))
            self.slices = slices


            self.multires_pos_embeds = nn.Parameter(torch.zeros(1, 84, self.hidden_size), requires_grad=False)

        if laplacian or self.gaussian_registers:
            self.low = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.mid = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.high = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.hhigh = nn.Parameter(torch.randn(1, 1, hidden_size))

            self.slices = [(0, 256), (256,320), (320, 336), (336,340)]

            self.multires_pos_embeds = nn.Parameter(torch.zeros(1, 84, self.hidden_size), requires_grad=False)

            self.x_embedder_low = PatchEmbed(4, patch_size, in_channels, hidden_size, bias=True)
            self.x_embedder_mid = PatchEmbed(8, patch_size, in_channels, hidden_size, bias=True)
            self.x_embedder_high = PatchEmbed(16, patch_size, in_channels, hidden_size, bias=True)



        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:

        if self.wavelets:


            self.LL = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.LH = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.HL = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.HH = nn.Parameter(torch.randn(1, 1, hidden_size))

            self.dwt = DWTForward(J=1, mode='zero', wave='haar').to("cuda").requires_grad_(False).to(torch.float32)
            self.idwt = DWTInverse(mode='zero', wave='haar').to("cuda").requires_grad_(False).to(torch.float32)

 



        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        if laplacian and diff_proj:
            self.final_layers_per_scale = nn.ModuleList([
                FinalLayer(hidden_size, patch_size, self.out_channels) for _ in range(4)
            ])

        else:

            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)



        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)




        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.wavelets:
        
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 8)

            pos_embed_cat = torch.cat([torch.from_numpy(pos_embed)]*4, dim=0)
            self.pos_embed.data.copy_(pos_embed_cat.float().unsqueeze(0))



        
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        if self.one_per_res :
            nn.init.normal_(self.low, std=0.02)
            nn.init.normal_(self.mid, std=0.02)
            nn.init.normal_(self.high, std=0.02)

            multires_pos_embeds = []
            for res_idx in [2,4,8]:
                pos_embed_res = nn.functional.interpolate(
                    self.pos_embed.detach().clone().reshape(1, 16, 16, self.hidden_size).permute(0, 3, 1, 2),
                    scale_factor=( res_idx / 16, res_idx / 16),
                    mode='bicubic',
                ).permute(0, 2, 3, 1).view(1, -1, self.hidden_size)
                multires_pos_embeds.append(
                    pos_embed_res 
                )
            self.multires_pos_embeds.data.copy_(torch.cat(multires_pos_embeds, dim=1))

        if self.laplacian:
            nn.init.normal_(self.low, std=0.02)
            nn.init.normal_(self.mid, std=0.02)
            nn.init.normal_(self.high, std=0.02)
            nn.init.normal_(self.hhigh, std=0.02)

            multires_pos_embeds = []
            for res_idx in [2,4,8]:
                pos_embed_res = nn.functional.interpolate(
                    self.pos_embed.detach().clone().reshape(1, 16, 16, self.hidden_size).permute(0, 3, 1, 2),
                    scale_factor=( res_idx / 16, res_idx / 16),
                    mode='bicubic',
                ).permute(0, 2, 3, 1).view(1, -1, self.hidden_size)
                multires_pos_embeds.append(
                    pos_embed_res 
                )
            self.multires_pos_embeds.data.copy_(torch.cat(multires_pos_embeds, dim=1))
 






        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if self.laplacian and self.diff_proj:
            for final_layer in self.final_layers_per_scale:
                nn.init.constant_(final_layer.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(final_layer.adaLN_modulation[-1].bias, 0)
                nn.init.constant_(final_layer.linear.weight, 0)
                nn.init.constant_(final_layer.linear.bias, 0)


        else:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)



    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]

        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        if self.wavelets:

            x = self.x_embedder(x)
            yll, yh = self.dwt(x.reshape(x.shape[0], self.hidden_size, 16,16).to(torch.float32))

            yhh = yh[0][:,:,0,:,:]
            yhl = yh[0][:,:,1,:,:]
            ylh = yh[0][:,:,2,:,:]

            yll = yll.reshape(x.shape[0], -1, self.hidden_size) + self.LL
            yhh = yhh.reshape(x.shape[0], -1, self.hidden_size) + self.HH
            ylh = ylh.reshape(x.shape[0], -1, self.hidden_size) + self.LH
            yhl = yhl.reshape(x.shape[0], -1, self.hidden_size) + self.HL
            x = torch.cat([yll, yhh, yhl,ylh], dim =1)



        elif self.one_per_res:
            x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2


            tokens = []
 
            low = self.multires_pos_embeds[:,:4,:] + self.low
            mid = self.multires_pos_embeds[:,4:20,:] + self.mid
            high = self.multires_pos_embeds[:,20:,:] + self.high
            tokens = torch.cat([low,mid,high], dim=1).repeat(x.shape[0],1,1)    

            x = torch.cat([x, tokens], dim=1)

        elif self.gaussian_registers:
            gaussian_pyr = gaussian_pyramid(x, 4)
            x_hh = self.x_embedder(gaussian_pyr[0])
            x_h = self.x_embedder_high(gaussian_pyr[1])
            x_m = self.x_embedder_mid(gaussian_pyr[2])
            x_l = self.x_embedder_low(gaussian_pyr[3])
            l = x_l + self.multires_pos_embeds[:,:4,:] + self.low
            m = x_m + self.multires_pos_embeds[:,4:20,:] + self.mid
            h = x_h +  self.multires_pos_embeds[:,20:,:] + self.high
            hh = x_hh + self.pos_embed + self.hhigh 
            x = torch.cat([hh, h, m, l], dim=1)


        elif self.laplacian:
            
            gaussian_pyr = gaussian_pyramid(x, 4)
            laplacian_pyr = laplacian_pyramid(gaussian_pyr)
            x_hh = self.x_embedder(laplacian_pyr[0])
            x_h = self.x_embedder_high(laplacian_pyr[1])
            x_m = self.x_embedder_mid(laplacian_pyr[2])
            x_l = self.x_embedder_low(laplacian_pyr[3])

            l = x_l + self.multires_pos_embeds[:,:4,:] + self.low
            m = x_m + self.multires_pos_embeds[:,4:20,:] + self.mid
            h = x_h +  self.multires_pos_embeds[:,20:,:] + self.high
            hh = x_hh + self.pos_embed + self.hhigh 
            # x = self.x_embedder(x)
            
            # x = x.reshape(x.shape[0], self.hidden_size, 16, 16)
            # gaussian_pyr = gaussian_pyramid(x, 4)
            # laplacian_pyr = laplacian_pyramid(gaussian_pyr)
            # laplacian_pyr = [p.reshape(x.shape[0], -1, self.hidden_size) for p in laplacian_pyr]


            # l = laplacian_pyr[3] + self.multires_pos_embeds[:,:4,:] + self.low
            # m = laplacian_pyr[2] + self.multires_pos_embeds[:,4:20,:] + self.mid
            # h = laplacian_pyr[1] +  self.multires_pos_embeds[:,20:,:] + self.high
            # hh = laplacian_pyr[0] + self.pos_embed + self.hhigh
            x = torch.cat([hh, h, m, l], dim=1)
        else:
            x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2



        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)       # (N, T, D)


        if self.wavelets:

            xll, xhh, xhl, xlh = torch.split(x, 64, dim=1)
            xh = torch.cat([xhh.unsqueeze(1), xhl.unsqueeze(1), xlh.unsqueeze(1)], dim=1).reshape(x.shape[0], self.hidden_size, 3, 8, 8)
            xll = xll.reshape(x.shape[0], self.hidden_size,8,8)
            x = self.idwt((xll, [xh])).reshape(x.shape[0], 256, self.hidden_size)

        if self.laplacian and self.diff_proj:

            x_scales = [self.final_layers_per_scale[idx](x[:, start:end, :], c) for idx, (start,end) in enumerate(self.slices)]
            x_scales_unpatched = [self.unpatchify(scale) for scale in x_scales]
            x = reconstruct_laplacian_pyramid(x_scales_unpatched)
            return x

        if (self.laplacian and self.registers) or self.gaussian_registers:

            x = self.final_layer(x[:,:256,:], c)               
            x = self.unpatchify(x)
            return x
        
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)



        if self.one_per_res:

            if self.registers:
                x = x[:, :256, :]
                x = self.unpatchify(x)
                return x

            x = [self.unpatchify(x[:, start:end, :]) for start,end in self.slices]            
            x_original = x[0]
            x_tokens = x[1:]
            x_tokens_upsampled = [F.interpolate(t, size=(32, 32), mode="nearest") for t in x_tokens]
            x_tokens_sum = sum(x_tokens_upsampled)

            x = x_tokens_sum + x_original

        elif self.laplacian:


            x = [self.unpatchify(x[:, start:end, :]) for start,end in self.slices]
            x = reconstruct_laplacian_pyramid(x)
        else:
            # breakpoint()
            x = self.unpatchify(x)

        return x


    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


if __name__ == "__main__":


    pos_2 = get_2d_sincos_pos_embed(1152, 2)
    pos_2 = torch.tensor(pos_2).reshape(2,2,-1)

    pos_8 = get_2d_sincos_pos_embed(1152, 8)
    pos_8 = torch.tensor(pos_8).reshape(8,8,-1)

    print(pos_2[1,1])
    print(pos_8[1,1])