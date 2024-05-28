## Part 1

In the styleGAN paper, the style is sampled from the latent space and combined to the synthesis network through ``AdaIN``, where each feature map $x_i$ is normalized separately, then scaled and biased using the corresponding $y_{s,i}$ and $y_{b,i}$.

There are three choices of latent space:
*  z -- Gaussian Noise
*  w -- $\it{MappingNetwork(z)}$
*  w+ -- use different w for different layers

<img styleGAN latent space>

We should initialize the latent first, then pass the latent through a pretrained ``Generator``, use the optimization-based method to draw the image near the target while keeping the image style faithful to the pretrained model.

The permutation and combination of **random seed**, **latent space**, **loss weights** and **data sample** leads to results in different quality. 


```
python main.py --model vanilla --mode project --latent z --perc_wgt 0.0001 --l1_wgt 100

python main.py --model stylegan --mode project --latent w

python main.py --model stylegan --mode project --latent w+
```

The results from ``w``/``w+`` are more faithful to the target(content image), while the style can be optimized as the pretrained model's.


## Part 2

In part2, we are not trying to transform the original cat data to style that is embeded in the pretrained model, but to add content and style to sketch image, while preserving the contour information indicated by the user given sketch.

The sketch image(RGBA) serves as the input, and a corresponding mask(0,1) is generated according to its alpha channel. This method is convenient, but sometimes fails to cover all the colored regions, since the user might use transparent brush in sketch creating.

We still use two terms of loss here:

* the *perceptual loss* $z^* = argmin_z \sum_i ||f_i(G(z))-v_i||_1
$
* the *Lp* loss $z^* = argmin_z ||M * G(z) - M * S||_1
$

In the code implementation, these two losses all support mask. But using mask in feature space(percetual loss) seems weired, because after the ``conv`` extraction, the feature belongs to sketch might not be staying at the same place. For denser stroke, this effect might be alleviated.

```
# for Adam optimization
python main.py --model vanilla --mode draw --latent z --input 'data/sketch/*.png' --perc_wgt 0.0001 --l1_wgt 100

# for LBFGS optimization
python main.py --model vanilla --mode draw --latent z --input 'data/sketch/*.png' --perc_wgt 0.0001 --l1_wgt 1000

python main.py --model stylegan --mode draw --latent w

python main.py --model stylegan --mode draw --latent w+
```
The weights of perceptual loss and LP loss could variate a lot. 

## Part 3

In this part, we need to add noise to the encoded input sketch$(x_{t-1} \rightarrow x_{t})$, then denoise from the noisy latent using parameters stored in the pretrained model, which can calculate the mean and variance of each step's distribution$(x_{t} \rightarrow x_{t-1})$, plus the ``cond`` and ``uncond`` guidance.

The sample step $N$ is recommended to be 500-700, because the DDPM pretraining uses 1000 steps between the encoded real image and the noise. Thus a $N$ less than 1000 will reserve the low level features of the sketch, which enables the content of sketch to be kept.

```
python img2img.py --prompt "Grumpy cat reimagined as a royal painting" --input-img assets/sketch-cat-512.png --strength 15 --num_timesteps 500
```


