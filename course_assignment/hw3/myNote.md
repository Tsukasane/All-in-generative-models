
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

It is worth mentioning that, the activation function of ``conv5`` has to be ``tanh``, because I misused the ``relu`` first and find the generator very difficult to learn from noise. 


I read [a survey](https://www.sciencedirect.com/science/article/pii/S1574013720303853) which happend mentioning the choice of activation function of DCGAN:

> 'except the final layer which had tanh activation, allowing the model to learn quicker to convergence and utilize the whole spectrum of the colours from the training data'. 

It seems the model got bad performance using 'relu' in the last layer of ``G`` because of its smaller slope and non-negative output value.

Also, the implementation of ``.detach()`` here is essential. It determines whether the gradient updates could be passed to the Generator.

### Diffaug

As stated in the paper, Diffaug conduct augmentation on both the fake images and the real images, implementing on training both ``G`` and ``D``.

todo<img paper>

In our case, using ``diffaug`` (color, translation, and cutout) benefits the training.
* **Provide a more efficient gradient in updating G.**
  The D_loss converges to a larger value since the augmentation makes the discriminator harder to distinguish between real and fake images. The converged G_loss turns to be smaller, indicating a higher quality of generated samples.
* **Diversify the training data for a better performance on small datasets**
  The cat datasets only have hundreds of images, making the ``G`` easy to memory all training samples, instead of learning data distribution.

Here are the training losses (better viewed after smoothing).

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

The data preprocessing(to the real images only) itself doesn't show much difference here.

# TODO slide bar on website

Results show the model is less likely to collapse after several training stages w/ ``diffaug``.



## Cycle GAN


w/ augmentation, the models are less likely to collapse, since the diverse input brings difficult to the Discriminator in distinguishing real and fake samples.

w/ consistency loss, the intermediate outputs are less likely to have outliers. The structure i.e. border shape of generated sample $G_{Y2X}(G_{X2Y}(X))$ is more consistent to the input $X$.

It is also important to set the magnitude of the ``lambda_cycle`` properly, since too small cannot contribute, but too large limits the potential of transferation.

For the discriminator, actually I like the results from ``DCDiscriminator`` which look more natural. With ``DCDiscriminator``, the generated samples show a severe change in the very beginning. ``PatchDiscriminator`` on the contrary, tightly constraints the details of each patch, slowing down the updating. But I think the preference of ``dc`` or ``patch`` depends on the application scenarios. By saying 'natural' I mean the generated sample from Grumpy is more like a real Russian Blue cat, with an almond face and clear yellow-green eyes. However, the color contrast, local texture, and global shape can be better preserved by ``PatchDiscriminator``. This features contribute more to identity. So if the detail preserving is preferred, then ``patch`` is probably a better choice.


Here are the results on the Apple/Orange dataset.

I find some interesting phenomenons about the bias in datasets. For example, it is hard for the model to transfer green apple to orange or transfer a cut apple to cut orange. Because the proportion of green apple and cut fruits images is small in the training data, the model have no idea what the inner side of fruits looks like and it not sure how to map a rarely find color in the original domain to the color in the target domain.