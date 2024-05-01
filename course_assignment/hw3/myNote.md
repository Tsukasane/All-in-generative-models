
## Vanilla GAN
This is a modified version of DCGAN, where the transpose conv in Generator is replaced by upsample of factor 2 and conv.

For the Discriminator structure, 
- Kernel Size ``K``: 4
- Stride ``S``: 2
- Input Size ``I``: 64
- Output Size ``O``: 32
- Padding ``P``: ?

We have 
  

$$
    O = \frac{I-K+2P}{S} +1
$$

The padding ``P`` of ``conv1``~``conv4`` should be 1, and ``P`` of ``conv5`` should be 0.

``K``, ``P``, ``S`` in Generator can also be inferenced by this formula.

It is worth mentioning that, the active function of ``conv5`` has to be ``tanh``, because I misused the ``relu`` first and find the generator very difficult to learn from noise. 

# TODO cite

I read [a survey](https://www.sciencedirect.com/science/article/pii/S1574013720303853) which happend mentioning the choice of activation function, quote 'except the final layer which had tanh activation, allowing the model to learn quicker to convergence and utilize the whole spectrum of the colours from the training data'. I think it is because the smaller slope and non-negative value of relu.

Also, the implementation of ``.detach()`` here is essential. It determines whether the gradient updates could be passed to the Generator.


Using ``diffaug`` (color, translation, and cutout), the G_loss turns to be smaller, indicating a higher quality of generated samples. The D_loss converges to a larger value since the augmentation makes the discriminator harder to distinguish between real and fake images, also resulting in a more efficient training of G.

Here are the training loss.

<div style="text-align:center">
    <img src="./figure/vanilla_basic_D.png" alt="img vanilla_basic_D" width="200" height="">
    <img src="./figure/vanilla_basic_G.png" alt="img vanilla_basic_G" width="200" height="">
</div>

<div style="text-align:center">
    <img src="./figure/vanilla_deluxe_D.png" alt="img vanilla_deluxe_D" width="200" height="">
    <img src="./figure/vanilla_deluxe_G.png" alt="img vanilla_deluxe_G" width="200" height="">
</div>

With the same type of date preprocessing, adding ``diffaug`` makes ``D`` converge to a larger loss and  ``G`` to a smaller, meaning the generated samples have a better quality to fool the discriminator.

<div style="text-align:center">
    <img src="./figure/vanilla_D.png" alt="img vanilla_basic_D" width="200" height="">
    <img src="./figure/vanilla_G.png" alt="img vanilla_basic_G" width="200" height="">
</div>

<div style="text-align:center">
    <img src="./figure/vanilla_diffaug_D.png" alt="img vanilla_deluxe_D" width="200" height="">
    <img src="./figure/vanilla_diffaug_G.png" alt="img vanilla_deluxe_G" width="200" height="">
</div>

The data preprocessing itself doesn't show much different here.

# TODO img (scroll)

Results show the model is less likely to collapse after several training stage, but the results of latter steps still shows a lack of diversity, a sign of overfitting.

#TODO slide bar on website


## Cycle GAN


w/ augmentation, the models are less likely to collapse, since the diverse input brings difficult to the Discriminator in distinguishing real and fake samples.

w/ consistency loss, the intermediate outputs are less likely to have outliers. The structure i.e. border shape of generated sample $G_{Y2X}(G_{X2Y}(X))$ is more consistent to the input $X$.

It is also important to set the magnitude of the ``lambda_cycle`` properly, since too small cannot contribute, but too large limits diversity.

For the discriminator, actually I like the results from ``DCDiscriminator`` which look more natural. ``DCDiscriminator`` causes a severe change in the very beginning. ``PatchDiscriminator`` on the contrary, tightly constraints the details of each patch. But I think the preference of ``dc`` or ``patch`` depends on the application scenarios. By saying 'natural' what I mean is the generated sample is more like a real Russian Blue cat, with an almond face and clear yellow-green eyes. However, the color contrast, local texture, and global shape can be better preserved by ``PatchDiscriminator``, which are the identity feature. 


