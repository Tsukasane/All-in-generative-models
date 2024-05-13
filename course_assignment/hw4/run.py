import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import copy
import sys
from utils import load_image, Normalization, device, imshow, imsave, get_image_optimizer
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

SEED = 11

# Set the random seed manually for reproducibility.
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def get_model_and_losses(cnn, style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    # build a sequential model consisting of a Normalization layer
    # then all the layers of the VGG feature network along with ContentLoss and StyleLoss
    # layers in the specified places
    
    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # here if you need a nn.ReLU layer, make sure to use inplace=False
    # as the in place version interferes with the loss layers

    normalization = nn.InstanceNorm2d(3)
    model = nn.Sequential(normalization)

    # print(f"--------- model cnn ---------")
    # print(cnn)
    # print(f"--------- end of model ---------")

    i=0 # layer index, use conv2d as ref
    j=0
    for ly in cnn: 

        if isinstance(ly, nn.Conv2d):
            name = f'conv_{i}'
            i += 1
            j+=1
        elif isinstance(ly, nn.ReLU):
            name = f'relu_{i}'
            ly = nn.ReLU(inplace=False)
            j+=1
        elif isinstance(ly, nn.MaxPool2d):
            name = f'maxpool_{i}'
            j+=1
        elif isinstance(ly, nn.BatchNorm2d):
            name = f'batchnorm_{i}'
            j+=1

        # print(f'add {name} to the model')
        model.add_module(name, ly)

        if name in content_layers_default:
            target = model(content_img) # detach later in ContentLoss
            cur_content_loss = ContentLoss(target) #init loss but don't calculate, with target embeded
            loss_name = f'content_loss_{i}'
            j+=1
            content_losses.append(j)
            model.add_module(loss_name, cur_content_loss)
            
        if name in style_layers_default:
            target = model(style_img)
            cur_style_loss = StyleLoss(target) #init loss but don't calculate
            loss_name = f'style_loss_{i}'
            j+=1
            style_losses.append(j)
            model.add_module(loss_name, cur_style_loss)

    print(f"Content Loss List:{content_losses}")
    print(f"Style Loss List:{style_losses}")
    

    # trim off the layers after the last content and style losses as they are vestigial
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i+1)]

    print(f"--------- start - new model ---------")
    print(model)
    print(f"--------- end - new model ---------")

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
    my_optimizer = get_image_optimizer(input_img) # a optimizer that conduct optimization on the input_img
    
    # run model training, with one weird caveat
    # we recommend you use LBFGS, an algorithm which preconditions the gradient
    # with an approximate Hessian taken from only gradient evaluations of the function
    # this means that the optimizer might call your function multiple times per step, so as
    # to numerically approximate the derivative of the gradients (the Hessian)
    # so you need to define a function here which does the following:

    print(f'original input_img {input_img}')

    prev_ttl = 1000

    def closure(): 
        # standardize the value
        input_img.data.clamp_(0,1)
        # clear the gradients
        my_optimizer.zero_grad() 
        model(input_img)
        sum_cl = 0 # sum content loss
        sum_sl = 0
        ttl = 0

        if use_content:
            for i in range(len(content_losses)):
                sum_cl+=model[content_losses[i]].loss
        if use_style:
            for i in range(len(style_losses)):
                sum_sl+=model[style_losses[i]].loss
        
        # total loss
        if use_content:
            ttl += sum_cl*content_weight
        if use_style:
            ttl += sum_sl*style_weight
        ttl.backward()

        input_img.data.clamp_(0,1)
        print(f'ContentLoss: {sum_cl}, StyleLoss: {sum_sl}')

        # TODO raise error when ttl=0 (either content loss nor style loss is used)
        # return the loss
        return ttl.item()

    cur_step = 0
    while cur_step<num_steps:   
        print(f'cur_step/num_steps:{cur_step}/{num_steps}')
        ttl = my_optimizer.step(closure)
        if ttl<prev_ttl:
            prev_ttl = ttl
            print(f'prev_ttl {prev_ttl} ttl {ttl}')
        else:
            break
        cur_step+=1
    
    # one more hint: the images must be in the range [0, 1]
    # but the optimizer doesn't know that
    # so you will need to clamp the img values to be in that range after every step
    
    # make sure to clamp once you are done
    input_img.data.clamp_(0,1)

    print(f'updated input_img {input_img}')

    return input_img


def main(style_img_path, content_img_path):

    # we've loaded the images for you
    style_img = load_image(style_img_path) # resize in load_image
    content_img = load_image(content_img_path)

    # interative MPL
    plt.ion()

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # plot the original input image:
    # plt.figure()
    # imshow(style_img, title='Style Image')
    imsave(style_img, title='style_img.png')

    # plt.figure()
    # imshow(content_img, title='Content Image')
    imsave(content_img, title='content_img.png')

    # we load a pretrained VGG19 model from the PyTorch models library
    # but only the feature extraction part (conv layers)
    # and configure it for evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = models.vgg19(pretrained=True).features.to(device).eval() # freeze the model

    # image reconstruction
    print("Performing Image Reconstruction from white noise initialization")
    # input_img = random noise of the size of content_img on the correct device
    input_img = torch.randn_like(content_img, device=device)

    # output = reconstruct the image from the noise
    output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=False)

    # plt.figure()
    # imshow(output, title='Reconstructed Image')
    imsave(output, title = "reconstructed_image.png")

    import pdb
    pdb.set_trace()
   

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
