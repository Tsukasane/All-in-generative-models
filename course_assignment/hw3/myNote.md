
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

It is worth mentioning that, the active function of ``conv5`` has to be ``tanh``, because I misused the ``relu`` first and find the generator very difficult to learn from noise. Also, the implementation of ``.detach()`` here is very interesting. It determined whether the gradient updates could be passed to the Generator.

#TODO slide bar on website

#TODO beautify loss curve


## Run

```
# basic aug, 
CUDA_VISIBLE_DEVICES=7 python vanilla_gan.py --data_preprocess basic

# deluxe aug,
python vanilla_gan.py --data_preprocess deluxe

# basic aug w/ diffaug
CUDA_VISIBLE_DEVICES=0 python vanilla_gan.py --data_preprocess basic --use_diffaug

# deluxe aug w/ diffaug
CUDA_VISIBLE_DEVICES=1 python vanilla_gan.py --data_preprocess deluxe --use_diffaug
```


## Cycle GAN

```
# w/o cycle consistency loss
python cycle_gan.py --disc patch --train_iters 1000

# w/ cycle consistency loss
python cycle_gan.py --disc patch --use_cycle_consistency_loss  --train_iters 1000
```