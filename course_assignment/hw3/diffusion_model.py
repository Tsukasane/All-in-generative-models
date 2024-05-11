import torch
from torch import einsum
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import numpy as np

import cv2
from diffusion_utils import *

# change this according to the training and testing files!
# Variable does not autopopulate from opts, but needs to be same as opts.denoising_steps
TIMESTEPS = 500


class Unet(nn.Module):
    def __init__(
        self,
        dim, # model dim = image size
        init_dim=3, # input image channel
        out_dim=3, # output image channel
        num_res_blocks=1,
        dim_mults=(1, 2, 4, 8), # default resolution
        base_channels=128,
        self_condition=[False, False, True, False],
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        # base_channels=dim
        # time embeddings
        self.time_embeddings = SinusoidalPositionEmbeddings(base_channels) # time embeddign with len=base_channels, map each time to a positional code
        # layers
        channels = []

        ########## first mapping
        self.first = nn.Conv2d(in_channels=init_dim, out_channels=base_channels, kernel_size=3, stride=1, padding="same") # pixel space -> feature space
        self.encoder_block = nn.ModuleList()
        self.bottleNeck_block = nn.ModuleList()
        self.decoder_block = nn.ModuleList()
        self.final = None

        in_dim = base_channels
        # basic UNet w/o attention
        ########## encoder
        for level in range(len(dim_mults)): # iter each resolution
            # 2 ResNetBlock + 1 DownSample
            o_dim = base_channels * dim_mults[level]

            for _ in range(num_res_blocks):
                self.encoder_block.append(ResnetBlock(
                    dim = in_dim, 
                    dim_out = o_dim, 
                    time_emb_dim=base_channels,
                    groups=resnet_block_groups))
            
                channels.append(in_dim)
                in_dim = o_dim

            if self_condition[level]: # if add self attention 
                self.attention = Attention(dim=o_dim)

            if level!=len(dim_mults)-1: # not the bottom layer
                self.encoder_block.append(Downsample(dim=in_dim))
                channels.append(in_dim) # record downsample channel


        ########## bottle neck
        for _ in range(num_res_blocks):
            self.bottleNeck_block.append(ResnetBlock(
                dim = in_dim, 
                dim_out = o_dim, 
                time_emb_dim=base_channels,
                groups=resnet_block_groups))

        ########## decoder
        reversed_dim_mults = dim_mults[::-1]
        for level in range(len(reversed_dim_mults)):
            o_dim = base_channels * reversed_dim_mults[level]

            for _ in range(num_res_blocks):
                self.decoder_block.append(ResnetBlock(
                    dim = in_dim+o_dim,
                    dim_out = o_dim,
                    time_emb_dim = base_channels,
                    groups=resnet_block_groups
                ))
                in_dim = o_dim

            if level!=len(reversed_dim_mults)-1:
                self.decoder_block.append(Upsample(in_dim))

        ########## last mapping(post process)
        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=resnet_block_groups, num_channels=in_dim),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding="same")
        )


    def forward(self, x, time, x_self_cond=None):
        # forward pass

        time_emb = self.time_embeddings(time)

        x = self.first(x)
        outs = [x] # skip connect

        for layer in self.encoder_block:
            if isinstance(layer, ResnetBlock):
                x = layer(x, time_emb) # 16, 64, 64, 64
                outs.append(x)
            else:
                x = layer(x)
        
        # 16, 256, 16, 16

        for layer in self.bottleNeck_block: 
            x = layer(x, time_emb)

        ### debug
        # for i in range(len(outs)):
            # print(outs[i].shape)
        
        # 16, 256, 16, 16
        for layer in self.decoder_block:
            if isinstance(layer, ResnetBlock):
                out = outs.pop()
                # debug
                # print(f'cur out {out.shape}')
                # print(f'cur x {x.shape}')
                x = torch.cat([x, out], dim=1)
                x = layer(x, time_emb)
            else:
                x = layer(x)

        x = self.final(x) # 16, 3, 64, 64

        return x
    
def beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, steps=timesteps)

timesteps = TIMESTEPS

# define beta schedule
betas = beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0) # \bar\alpha
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas) # 1/sqrt(\alpha)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) # sqrt(1-\bar\alpha)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]

    out = a.gather(-1, t.cpu())

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    # if noise is None:
    #     noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

image_size = 128

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    # noise 
    if noise is None:
        noise = torch.randn_like(x_start)
    # noisy img
    x_noisy = q_sample(x_start, t, noise) # get noisy image
    # predicted noise
    predicted_noise = denoise_model(x_noisy, t) # denoise the generated noisy image

    if loss_type == 'l1':
        loss = nn.L1Loss()(noise, predicted_noise) # define l1 loss
    elif loss_type == 'l2':
        loss = nn.MSELoss()(noise, predicted_noise) # define l2 loss
    elif loss_type == "huber":
        loss = nn.SmoothL1Loss()(noise, predicted_noise) # define smooth l1(huber) loss
    else:
        raise NotImplementedError()

    return loss


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)

        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]

    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []
    
    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
