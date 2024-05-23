# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe), modified by Zhiqiu Lin (zl279@cornell.edu)
# --------------------------------------------------------
from __future__ import print_function

import argparse
import os
import os.path as osp
import numpy as np

from LBFGS import FullBatchLBFGS

import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import torchvision.utils as vutils
from torchvision.models import vgg19

from dataloader import get_data_loader

SEED = 11

# Set the random seed manually for reproducibility.
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def build_model(name):
    if name.startswith('vanilla'):
        z_dim = 100
        model_path = 'pretrained/%s.ckpt' % name
        pretrain = torch.load(model_path, map_location=torch.device('cpu')) # debug on cpu
        from vanilla.models import DCGenerator
        model = DCGenerator(z_dim, 32, 'instance')
        model.load_state_dict(pretrain)

    elif name == 'stylegan':
        model_path = 'pretrained/%s.ckpt' % name
        import sys
        sys.path.insert(0, 'stylegan')
        from stylegan import dnnlib, legacy
        with dnnlib.util.open_url(model_path) as f:
            model = legacy.load_network_pkl(f)['G_ema']
            z_dim = model.z_dim
    else:
         return NotImplementedError('model [%s] is not implemented', name)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, z_dim


class Wrapper(nn.Module):
    """The wrapper helps to abstract stylegan / vanilla GAN, z / w latent"""
    def __init__(self, args, model, z_dim):
        super().__init__()
        self.model, self.z_dim = model, z_dim
        self.latent = args.latent
        self.is_style = args.model == 'stylegan'

    def forward(self, param):
        if self.latent == 'z':
            if self.is_style:
                image = self.model(param, None)
            else:
                image = self.model(param)
        # w / wp
        else:
            assert self.is_style
            if self.latent == 'w':
                param = param.repeat(1, self.model.mapping.num_ws, 1)
            image = self.model.synthesis(param)
        return image


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class PerceptualLoss(nn.Module): # this class is only for perceptualloss
    def __init__(self, add_layer=['conv_5']):
        super().__init__()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        norm = Normalization(cnn_normalization_mean, cnn_normalization_std)
        cnn = vgg19(pretrained=True).features.to(device).eval()
        
        # TODO (Part 1): implement the Perceptual/Content loss
        #                hint: hw4
        self.perceptualLoss = nn.MSELoss()
        self.add_layer = add_layer

        # You may split the model into different parts and store each part in 'self.model'
        self.model = nn.ModuleList()
        self.model.add_module('norm', norm)
        
        i=0 # all layer index
        j=0 # conv layer index TODO maybe no need to count layer index
        for ly in cnn:
            if isinstance(ly, nn.Conv2d):
                name = f'conv_{i}'
                i+=1
                j+=1
            elif isinstance(ly, nn.ReLU):
                name = f'conv_{i}'
                ly = nn.ReLU(inplace=False)
                j+=1
            elif isinstance(ly, nn.MaxPool2d):
                name = f'maxpool_{i}'
                j+=1
            elif isinstance(ly, nn.BatchNorm2d):
                name = f'batchnorm_{i}'
                j+=1

            self.model.add_module(name, ly)
                

    def forward(self, pred, target):
        mask = None
        if isinstance(target, tuple):
            target, mask = target
        
        loss = 0.
        len_perc = len(self.add_layer)
        for net in self.model:
            pred = net(pred)
            target = net(target)

            if len_perc<=0:
                break # trim off the extra model layers
            # TODO (Part 1): implement the forward call for perceptual loss
            #                free feel to rewrite the entire forward call based on your
            #                implementation in hw4

            if mask==None and net in self.add_layer: 
                len_perc-=1
                loss += self.perceptualLoss(pred, target) 

            # TODO (Part 3): if mask is not None, then you should mask out the gradient
            #                based on 'mask==0'. You may use F.adaptive_avg_pool2d() to 
            #                resize the mask such that it has the same shape as the feature map.
            elif mask and net in self.add_layer:
                len_perc-=1
                t_size = pred.shape[-2:] # H W
                mask = F.adaptive_avg_pool2d(t_size)
                loss += self.perceptualLoss(pred * mask, target * mask)

        return loss 
# integrate losses except delta loss
class Criterion(nn.Module): # the combination of all masks
    def __init__(self, args, mask=False, layer=['conv_5']):
        super().__init__()
        self.perc_wgt = args.perc_wgt
        self.l1_wgt = args.l1_wgt # weight for l1 loss/mask loss
        self.mask = mask # bool flag
        
        self.perc = PerceptualLoss(layer)

    def forward(self, pred, target):
        """Calculate loss of prediction and target. in p-norm / perceptual  space"""
        loss = 0.
        if self.mask:
            # TODO (Part 3): loss with mask
            loss += self.perc(pred, target) * self.perc_wgt # keep the target as a tuple, mask is processed inside the PL class
            target, mask = target # mask(vector) != self.mask(flag)
            t_size = pred.shape[-2:]
            mask = F.adaptive_avg_pool2d(t_size)
            loss += nn.L1Loss()(pred * mask, target * mask) * self.l1_wgt

        else:
            # TODO (Part 1): loss w/o mask
            loss += self.perc(pred, target) * self.perc_wgt
            loss += nn.L1Loss()(pred, target) * self.l1_wgt
        
        return loss


def save_images(image, fname, col=8):
    image = image.cpu().detach()
    image = image / 2 + 0.5

    image = vutils.make_grid(image, nrow=col)  # (C, H, W)
    image = image.numpy().transpose([1, 2, 0])
    image = np.clip(255 * image, 0, 255).astype(np.uint8)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        imageio.imwrite(fname + '.png', image)
    return image


def save_gifs(image_list, fname, col=1):
    """
    :param image_list: [(N, C, H, W), ] in scale [-1, 1]
    """
    image_list = [save_images(each, None, col) for each in image_list]
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    imageio.mimsave(fname + '.gif', image_list)


def sample_noise(dim, device, latent, model, N=1, from_mean=False): # debug
    """
    To generate a noise vector, just sample from a normal distribution.
    To generate a style latent, you need to map the noise (z) to the style (W) space given the `model`.
    You will be using model.mapping for this function.
    Specifically,
    if from_mean=False,
        sample N noise vector (z) or N style latent(w/w+) depending on latent value.
    if from_mean=True
        if latent == 'z': Return zero vectors since zero is the mean for standard gaussian
        if latent == 'w'/'w+': You should sample N=10000 z to generate w/w+ and then take the mean.
    Some hint on the z-mapping can be found at stylegan/generate_gif.py L70:81.
    Additionally, you can look at stylegan/training/networks.py class Generator L477:500
    :return: Tensor on device in shape of (N, dim) if latent == z
             Tensor on device in shape of (N, 1, dim) if latent == w
             Tensor on device in shape of (N, nw, dim) if latent == w+
    """
    # TODO (Part 1): Finish the function below according to the comment above
    num_phases =10000
    if latent == 'z':
        vector = torch.randn(N, dim, device=device) if not from_mean else torch.zeros(N, dim, device=device)
    elif latent == 'w':
        if from_mean: # TODO check whether latents is randn init or zeros init
            z_vector = torch.randn(num_phases, dim, device=device)
            w_vector = model.mapping(z_vector, None)
            vector = w_vector.mean(dim=0, keepdim=True)
            vector = vector.mean(dim=1, keepdim=True) # has a additional dim for different layers
            # (N, 1, 512) N=1
        else:
            z_vector = torch.randn(N, dim, device=device)
            vector = model.mapping(z_vector, None).mean(dim=1, keepdim=True)

    elif latent == 'w+':
        # nw = model.mapping.num_layers 
        if from_mean:
            z_vector = torch.randn(num_phases, dim, device=device)
            w_vector = model.mapping(z_vector, None)
            vector = w_vector.mean(dim=0, keepdim=True) # has a additional dim for different layers
            # (N, 1, 512) N=1
        else:
            z_vector = torch.randn(N, dim, device=device)
            vector = model.mapping(z_vector, None)
    else:
        raise NotImplementedError('%s is not supported' % latent)
    return vector


def optimize_para(wrapper, param, target, criterion, num_step, save_prefix=None, res=False, delta_wgt=.01):
    """
    wrapper: image = wrapper(z / w/ w+): an interface for a generator forward pass.
    param: z / w / w+ (returned latent from sample_noise)
    target: (1, C, H, W)
    criterion: loss(pred, target)
    """
    device = param.device
    delta = torch.zeros_like(param, requires_grad=True, device=device)
    optimizer = FullBatchLBFGS([delta], lr=.01, line_search='Wolfe')
    # optimizer = torch.optim.Adam([delta], lr=.01)
    iter_count = [0]
    def closure():
        iter_count[0] += 1
        # TODO (Part 1): Your optimiztion code. Free free to try out SGD/Adam.
        optimizer.zero_grad()

        image = wrapper(param + delta)
        image.data.clamp_(0,1)
        
        loss = criterion(image, target)
        if delta_wgt: # default None
            loss += delta_wgt * torch.norm(delta, p=2) # encouraging delta to remain small
        
        if iter_count[0] % 250 == 0:
            # visualization code
            print('iter count {} loss {:4f}'.format(iter_count, loss.item()))
            if save_prefix is not None:
                iter_result = image.data.clamp_(-1, 1)
                save_images(iter_result, save_prefix + '_%d' % iter_count[0])
        return loss

    loss = closure()
    loss.backward()
    while iter_count[0] <= num_step:
        options = {'closure': closure, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options) # only LBFG can cal dic
        #  lr: The learning rate found by the line search that minimizes the loss.
        #  F_eval: how many times the loss function was computed
        #  G_eval: how many times the gradients were computed
        print(f'loss: {loss}')
    
    image = wrapper(param + delta) # add delta
    image.data.clamp_(0,1)
    return param + delta, image


def sample(args):
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)
    batch_size = 16
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    noise = sample_noise(z_dim, device, args.latent, model, batch_size)
    image = wrapper(noise) # randomly sample noise z and put it through the DCGenerator
    fname = os.path.join('output/forward/%s_%s' % (args.model, args.mode))
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    save_images(image, fname)


def project(args):
    # load images
    loader = get_data_loader(args.input, args.resolution, is_train=False)

    # define and load the pre-trained model
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)
    print('model {} loaded'.format(args.model))
    criterion = Criterion(args)
    # project each image
    for idx, (data, _) in enumerate(loader):
        target = data.to(device)
        save_images(data, 'output/project/%d_data' % idx, 1)
        param = sample_noise(z_dim, device, args.latent, model)
        optimize_para(wrapper, param, target, criterion, args.n_iters,
                      'output/project/%d_%s_%s_%g' % (idx, args.model, args.latent, args.perc_wgt), delta_wgt=args.delta_wgt)
        if idx >= 0: # debug, set the sample num from dataset
            break


def draw(args):
    # define and load the pre-trained model
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)

    # load the target and mask
    loader = get_data_loader(args.input, args.resolution, alpha=True)
    criterion = Criterion(args, True)
    for idx, (rgb, mask) in enumerate(loader):
        rgb, mask = rgb.to(device), mask.to(device)
        save_images(rgb, 'output/draw/%d_data' % idx, 1)
        save_images(mask, 'output/draw/%d_mask' % idx, 1)
        # TODO (Part 3): optimize sketch 2 image
        #                hint: Set from_mean=True when sampling noise vector


def interpolate(args):
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)

    # load the target and mask
    loader = get_data_loader(args.input, args.resolution)
    criterion = Criterion(args)
    for idx, (image, _) in enumerate(loader):
        save_images(image, 'output/interpolate/%d' % (idx))
        target = image.to(device)
        param = sample_noise(z_dim, device, args.latent, model, from_mean=True)
        param, recon = optimize_para(wrapper, param, target, criterion, args.n_iters)
        save_images(recon, 'output/interpolate/%d_%s_%s' % (idx, args.model, args.latent))
        if idx % 2 == 0:
            src = param
            continue
        dst = param
        alpha_list = np.linspace(0, 1, 50)
        image_list = []
        with torch.no_grad():
            # TODO (B&W): interpolation code
            #                hint: Write a for loop to append the convex combinations to image_list
            pass
        save_gifs(image_list, 'output/interpolate/%d_%s_%s' % (idx, args.model, args.latent))
        if idx >= 3:
            break
    return


def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='stylegan', choices=['vanilla', 'stylegan'])
    parser.add_argument('--mode', type=str, default='sample', choices=['sample', 'project', 'draw', 'interpolate'])
    parser.add_argument('--latent', type=str, default='z', choices=['z', 'w', 'w+'])
    parser.add_argument('--n_iters', type=int, default=1000, help="number of optimization steps in the image projection")
    parser.add_argument('--perc_wgt', type=float, default=0.01, help="perc loss weight")
    parser.add_argument('--l1_wgt', type=float, default=10., help="L1 pixel loss weight")
    parser.add_argument('--delta_wgt', type=float, default=None, help="weight of the regularization loss that penalizes L2 norm of delta")
    parser.add_argument('--resolution', type=int, default=64, help='Resolution of images')
    parser.add_argument('--input', type=str, default='data/cat/*.png', help="path to the input image")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if args.mode == 'sample':
        sample(args)
    elif args.mode == 'project':
        project(args)
    elif args.mode == 'draw':
        draw(args)
    elif args.mode == 'interpolate':
        interpolate(args)