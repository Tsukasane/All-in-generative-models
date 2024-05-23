# Assignment #4 - Neural Style Transfer

## Background
Previous artists with strong personal styles left behind masterpieces that we appreciate. Sometimes, we wish to transfer the artistic style from these existing works to our custom scenarios while ensuring that our original content remains largely intact. The sources of both style and content can be in images, videos, or other formats, as can the generated target.

## Motivation
Generate a sample image with the style of ``style_img`` and the content of ``content_img``.

## Content Reconstruction

The model ``VGG19`` is fixed and trimed as a feature encoder, and the target image is detached from the computational graph. So the whole purpose of the gradient update is to lead the input_img using the collection of loss in one forward pass to update itself through the backpropagation. The class ``ContentLoss`` is implemented as a transparent layer, through which the input is not modified, but the contentloss regarding to the embedded target in same shape is calculated and stored. It is conducted in the feature space, rather than the pixel space. 

Firstly, I tried to add this ``ContentLoss`` layer after ``conv4``. The small learning rate for ``optim.LBFGS`` is important. With the default value of 1, the updating pace is too fast and the process is easy to fall into local minima. I implement a naive stop criterion, which stops the loss if it grows larger after one step and use ``lr=0.01``.


Using features in different layers to calculate content loss leads to different results. More coarse the feature map, more blurry the reconstructed image. On the contrary, finer feature maps leads to sharper reconstructed results. Among all the outcomes,  ``conv2`` shows a better preservation of color.

<img 1 2 3 4 5>

I also tried using ``conv2`` and ``conv5`` simultaneously, the result seems more balanced in color and details.

<img>

## Texture Synthesis

The implementation of ``StyleLoss`` is quite like the ``ContentLoss``. What makes it focus on texture is the stocastic Gram Matrix. Adding a single layer of ``StyleLoss`` does not work well. Very similar to the content reconstruction, coarse layer focus on coarse-grined features, and fine layer focus on fine-grined features. 

<img 1 2 3 4 5>

I prefer the result using ``conv1`` ~ ``conv5`` together. Since the ``input_img`` is initialized by white noise, if the random seed is fixed, then two runs will result the same. Using different seeds allow the samples to be diverse.

<img 1 2>


## Style Transfer
I choose the hyperparameter of lr=0.01, style_weight=20, content_weight=1. The reconstructed results from given images are shown in grid.

1 style:picasso  content:dancing
2 style:picasso  content:wally
3 style:the scream  content:dancing
4 style:the scream  content:wally

<img grid>

Using random white noise or content image with the same random seed for initialization gets different results. I think content init better preserved the color of the original image.

<img 1 2>

Here's the result from my own images.

<img >



Run:

```
python run.py ./images/style/escher_sphere.jpeg ./images/content/phipps.jpeg
```


TODO: different styles for different moving objects in the same video.