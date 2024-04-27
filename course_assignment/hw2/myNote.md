# Assignment #2 - Gradient Domain Fusion

## Background
Sometimes we want to combine the object from one image to the background of another. Due to the difference in color tone and light between images and the unperfect crop, direct copy-and-paste always results in unsatisfactory outcomes.

## Motivation
Fuse two images as seamless as possible.

## Toy Problem

This problem requires a basic implementation of calculating (discrete) gradients and forming the least square error constraints in a matrix $({\bf A}v-{\bf b})^2=0$.

We want to reconstruct an image with only the up-left corner pixel information. Although there is no other direct intensity of other pixels, we can use a 2-direction gradient to simulate the original image and get a similar result $v$.

Here are three objective functions:

1. $
    \operatorname{argmin}((v(x+1, y)-v(x,y)) - (s(x+1, y)-s(x,y)))^2
$ -- in the same pixel line, reconstructed gradient(right - left) should be close to the source image.

2. $
    \operatorname{argmin}((v(x, y+1)-v(x,y)) - (s(x, y+1)-s(x,y)))^2
$ -- in the same pixel column, reconstructed gradient(down - up) should be close to the source image.

3. $
    \operatorname{argmin}(v(0,0)-s(0,0))^2
$ -- copy the pixel on the top-left corner of the source image and paste it to the reconstructed $v$.

Once the numbers in parameter matrix $A$ and the bias matrix $b$ are set, ``np.linalg.lstsq`` can solve the appropriate $v$. The result is shown as follows.

<div style="text-align:center">
    <img src="./figures/toy_solution.jpg" alt="img overlap" width="500" height="">
</div>

<a id="pb"> </a>

## Poisson Blending
The blending constraint is 
$$
    {\bf v} = \operatorname{argmin_v} \sum_{i\in S, j\in N_i \cap S}((v_i - v_j)-(s_i-s_j))^2 + \sum_{i\in S, j\in N_i \cap \lnot S}((v_i-t_j)-(s_i-s_j))^2
$$

Similar to the toy problem, it turns to 2 objectives. 

1. $
    \operatorname{argmin_v} \sum_{i\in S, j\in N_i \cap S}((v_i - v_j)-(s_i-s_j))^2
$

2. $
    \operatorname{argmin_v} \sum_{i\in S, j\in N_i \cap \lnot S}((v_i-t_j)-(s_i-s_j))^2
$ 

Also, background of the target image should be directly copied to the reconstructed one, which can be written as the third objective in the format below, but actually there is no need to put this constraint into the matrix solving process:

3. $
    \operatorname{argmin_v} \sum_{i\in \lnot S}((v_i-t_i))^2
$ 

In the code implementation, I use ``1`` and ``2`` constraints on two separate groups of pixels. The partition depends on whether the 4 adjacent pixels of the specified center is inside the masked region. Notice that if a pixel on the border of mask has 2 adjacent pixels outside the mask, then only these two gradients should use ``constraint 2``, but the other two still use ``constraint 1``.

<div style="text-align:center">
    <img src="./figures/border_pixel.png" alt="img overlap" width="300" height="250">
</div>

For saving the computational cost, i.e. simplify the linear matrix solving by reducing variable dimension, I first crop the bg, fg, and mask by bounding box of the masked region, and send those cropped small patches to fuse. This will not effect the output because bg's pixel intensity is fixed.

Here is the result of the given sample. Result from Posisson Blend is smoother than the naive copy-and-paste.

<div style="text-align:center">
    <img src="./figures/blend_1.jpg" alt="img overlap" width="500" height="">
</div>


Then I tried several custom examples. I slightly modified the ``masking_code``(many thanks to the author) to ensure the source is resized to fit the target size. 

In the first try, I use a real kitty image as ``source`` and an anime background as ``target``. The Poisson Blend(PB) result is still better than Naive Blend(NB, and the kitty is really cute), but there's artifacts that make the PB result unperfect. Since this anime target tend to use bright color (~(255, 255, 255)), there's not enough space for the cropped kitty to adjust its gradient given the fixed border of the target image. Also, the dark background left in the cropped source makes the kitty turn even brighter, leading to a result like over-exposure. Moreover, the texture of BG in the cropped source is not consistent with the grass in the target. Although the transition seems smooth, it still not that natural.

<div style="text-align:center">
    <img src="./figures/blend_kts.jpg" alt="img overlap" width="500" height="">
</div>

The second try is bubble with high transparency. Notice that in the center of the bubble, it is expected to reflect sunlight as show on the ``target`` image, but the Poisson Blend result got dark in the center due to the dark background on ``source``. 

<div style="text-align:center">
    <img src="./figures/bubble_source.png" alt="img overlap" width="200" height="">
    <img src="./figures/bubble_target.png" alt="img overlap" width="200" height="">
    <p>source -- target</p>
</div>

<div style="text-align:center">
    <img src="./figures/blend_bubble.jpg" alt="img overlap" width="500" height="">
</div>


## Mix Gradient

As in [Poisson Blending](#pb), there are two constraints for the linear matrix function. There is only slight difference in constraint ``1``, where the $s_i-s_j$ term is replaced by $\operatorname{max}(s_i-s_j,t_i-t_j)$. 

The transparency gets better now as the sunlight are visible trough the bubble. However, there comes up many annoying black noise. I think this partly because the bubble not only transmit the scenery behind but also reflect the scenery in front of it. Generally, the reflected scene tend to be ambiguous, i.e. smooth with low gradient value, but it is hard to tell which part of it transmit and which reflect, causing an uncertainty of the mixed result. Plus, low resolution makes the gradient calculation inprecise.

<div style="text-align:center">
    <img src="./figures/mix_bubble.jpg" alt="img overlap" width="500" height="">
</div>

## Appendix

Custom images source: [kitty](https://zhuanlan.zhihu.com/p/544659370), [sunset](https://www.meipian.cn/1yeh70sk), [bubble](https://www.ivsky.com/tupian/paopao_v60018/pic_952186.html), [anime grass](https://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=动漫风景图片&step_word=&hs=0&pn=43&spn=0&di=7348476013078118401&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=0&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=undefined&cs=4137056591%2C1476057380&os=3572866320%2C879998828&simid=4170758312%2C539261113&adpicid=0&lpn=0&ln=1826&fr=&fmq=1714125307623_R&fm=&ic=undefined&s=undefined&hd=undefined&latest=undefined&copyright=undefined&se=&sme=&tab=0&width=undefined&height=undefined&face=undefined&ist=&jit=&cg=&oriquery=&objurl=http%3A%2F%2Fpic1.win4000.com%2Fmobile%2Fa%2F567366ba78ec4.jpg%3Fdown&gsm=3c&rpstart=0&rpnum=0&islist=&querylist=&nojc=undefined&dyTabStr=MCwzLDEsMiw0LDYsNSw4LDcsOQ%3D%3D&lid=7539363456101846427)