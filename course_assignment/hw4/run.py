import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import copy
import sys
from utils import load_image, Normalization, device, imshow, get_image_optimizer
from style_and_content import ContentLoss, StyleLoss


"""A ``Sequential`` module contains an ordered list of child modules. For
instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU, MaxPool2d,
Conv2d, ReLU…) aligned in the right order of depth. We need to add our
content loss and style loss layers immediately after the convolution
layer they are detecting. To do this we must create a new ``Sequential``
module that has content loss and style loss modules correctly inserted.
"""

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_model_and_losses(cnn, style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    # build a sequential model consisting of a Normalization layer
    # then all the layers of the VGG feature network along with ContentLoss and StyleLoss
    # layers in the specified places
    
    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # here if you need a nn.ReLU layer, make sure to use inplace=False
    # as the in place version interferes with the loss layers
    # trim off the layers after the last content and style losses
    # as they are vestigial

    normalization = nn.InstanceNorm2d(3)
    model = nn.Sequential(normalization)

    i=0 # layer index, use conv2d as ref
    for ly in cnn.modules(): # TODO see what kind of layer is here
        if isinstance(ly, nn.Conv2d):
            name = f'conv_{i}'
            i += 1
        elif isinstance(ly, nn.ReLU):
            name = f'relu_{i}'
            ly = nn.ReLU(inplace=False)
        elif isinstance(ly, nn.MaxPool2d):
            name = f'maxpool_{i}'
        elif isinstance(ly, nn.BatchNorm2d):
            name = f'batchnorm_{i}'

        model.add_module(name, ly)
   
        if name in content_layers_default:
            target = model(content_img) # detach later in ContentLoss
            cur_content_loss = ContentLoss(target)
            loss_name = f'content_loss_{i}'
            content_losses.append(loss_name)
            model.add_module(loss_name, cur_content_loss)
            
        if name in style_layers_default:
            target = model(style_img)
            cur_style_loss = StyleLoss(target)
            loss_name = f'style_loss_{i}'
            style_losses.append(loss_name)
            model.add_module(loss_name, cur_style_loss)


    # raise NotImplementedError()

    return model, style_losses, content_losses


"""Finally, we must define a function that performs the neural transfer. For
each iteration of the networks, it is fed an updated input and computes
new losses. We will run the ``backward`` methods of each loss module to
dynamicaly compute their gradients. The optimizer requires a “closure”
function, which reevaluates the module and returns the loss.

We still have one final constraint to address. The network may try to
optimize the input with values that exceed the 0 to 1 tensor range for
the image. We can address this by correcting the input values to be
between 0 to 1 each time the network is run.



"""


def run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True, num_steps=300,
                     style_weight=1000000, content_weight=1):
    """Run the image reconstruction, texture synthesis, or style transfer."""
    print('Building the style transfer model..')
    # get your model, style, and content losses
    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img)
    # get the optimizer
    optimizer = torch.optim.LBFGS(input_img.require_grad_())

    # run model training, with one weird caveat
    # we recommend you use LBFGS, an algorithm which preconditions the gradient
    # with an approximate Hessian taken from only gradient evaluations of the function
    # this means that the optimizer might call your function multiple times per step, so as
    # to numerically approximate the derivative of the gradients (the Hessian)
    # so you need to define a function here which does the following:
    def create_loss_closure(model, optimizer, loss_cls):
        def closure(target_img, weight):
            # clear the gradients
            optimizer.zero_grad() 
            # compute the loss and it's gradient
            cur_loss = loss_cls(target_img).loss * weight
            cur_loss.backward()

            # clamp each step
            input_img.data.clamp_(0,1)
            # return the loss
            return cur_loss.item()
        return closure

    if use_content:
        content_loss_closure = create_loss_closure(input_img, content_img, ContentLoss)
        content_loss = content_loss_closure(content_img, content_weight)
    if use_style:
        style_loss_closure = create_loss_closure(input_img, style_img, StyleLoss)
        style_loss = style_loss_closure(style_img, style_weight)

    # one more hint: the images must be in the range [0, 1]
    # but the optimizer doesn't know that
    # so you will need to clamp the img values to be in that range after every step
    
    # raise NotImplementedError()

    # make sure to clamp once you are done
    input_img.data.clamp_(0,1)

    return input_img


def main(style_img_path, content_img_path):
    # we've loaded the images for you
    style_img = load_image(style_img_path)
    content_img = load_image(content_img_path)

    # interative MPL
    plt.ion()

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # plot the original input image:
    plt.figure()
    imshow(style_img, title='Style Image')

    plt.figure()
    imshow(content_img, title='Content Image')

    # we load a pretrained VGG19 model from the PyTorch models library
    # but only the feature extraction part (conv layers)
    # and configure it for evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = models.vgg19(pretrained=True).features.to(device).eval()


    # image reconstruction
    print("Performing Image Reconstruction from white noise initialization")
    # input_img = random noise of the size of content_img on the correct device
    input_img = torch.randn_like(content_img, device=content_img.device())
    # output = reconstruct the image from the noise
    output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=False)
    
   

    plt.figure()
    imshow(output, title='Reconstructed Image')

    # texture synthesis
    print("Performing Texture Synthesis from white noise initialization")
    # input_img = random noise of the size of content_img on the correct device
    # output = synthesize a texture like style_image

    plt.figure()
    # imshow(output, title='Synthesized Texture')

    # style transfer
    # input_img = random noise of the size of content_img on the correct device
    # output = transfer the style from the style_img to the content image

    plt.figure()
    # imshow(output, title='Output Image from noise')

    print("Performing Style Transfer from content image initialization")
    # input_img = content_img.clone()
    # output = transfer the style from the style_img to the content image

    plt.figure()
    # imshow(output, title='Output Image from noise')

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    args = sys.argv[1:3]
    main(*args)
