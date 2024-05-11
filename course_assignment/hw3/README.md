## Run

### Vanilla GAN
```
# basic aug 
CUDA_VISIBLE_DEVICES=7 python vanilla_gan.py --data_preprocess basic

# deluxe aug
python vanilla_gan.py --data_preprocess deluxe

# basic aug w/ diffaug
CUDA_VISIBLE_DEVICES=6 python vanilla_gan.py --data_preprocess basic --use_diffaug

# deluxe aug w/ diffaug
CUDA_VISIBLE_DEVICES=7 python vanilla_gan.py --data_preprocess deluxe --use_diffaug
```


### Cycle GAN

```
# w/o cycle consistency loss

# cat_10deluxe_instance_patch_cycle_naive
CUDA_VISIBLE_DEVICES=7 python cycle_gan.py --disc patch --train_iters 10000 --log_step=200 --g_conv_dim=64 --d_conv_dim=64 --sample_every=200

# cat_10deluxe_instance_patch_cycle_naive_diffaug
CUDA_VISIBLE_DEVICES=6 python cycle_gan.py --disc patch --train_iters 10000 --log_step=200 --g_conv_dim=64 --d_conv_dim=64 --sample_every=200 --use_diffaug


# cat_10deluxe_instance_patch_cycle_naive_cycle_diffaug
CUDA_VISIBLE_DEVICES=6 python cycle_gan.py --disc patch --train_iters 10000 --log_step=200 --g_conv_dim=64 --d_conv_dim=64 --sample_every=200 --use_diffaug --use_cycle_consistency_loss 


#cat_1deluxe_instance_patch_cycle_naive_cycle_diffaug
CUDA_VISIBLE_DEVICES=7 python cycle_gan.py --disc patch --train_iters 10000 --log_step=200 --g_conv_dim=64 --d_conv_dim=64 --sample_every=200 --use_diffaug --use_cycle_consistency_loss --lambda_cycle=1


# cat_1deluxe_instance_dc_cycle_naive_cycle_diffaugcat_1deluxe_instance_dc_cycle_naive_cycle_diffaug
CUDA_VISIBLE_DEVICES=7 python cycle_gan.py --disc dc --train_iters 10000 --log_step=200 --g_conv_dim=64 --d_conv_dim=64 --sample_every=200 --use_diffaug --use_cycle_consistency_loss --lambda_cycle=1

#cat_10deluxe_instance_dc_cycle_naive_diffaug
CUDA_VISIBLE_DEVICES=7 python cycle_gan.py --disc dc --train_iters 10000 --log_step=200 --g_conv_dim=64 --d_conv_dim=64 --sample_every=200 --use_diffaug


# apple/orange w/o ccl  dc
CUDA_VISIBLE_DEVICES=7 python cycle_gan.py --disc dc --train_iters 10000 --log_step=200 --g_conv_dim=64 --d_conv_dim=64 --sample_every=200 --use_diffaug --X apple2orange/apple --Y apple2orange/orange

# apple/orange w/o ccl  patch
CUDA_VISIBLE_DEVICES=6 python cycle_gan.py --disc patch --train_iters 10000 --log_step=200 --g_conv_dim=64 --d_conv_dim=64 --sample_every=200 --use_diffaug --X apple2orange/apple --Y apple2orange/orange


# apple/orange w/ ccl  dc
CUDA_VISIBLE_DEVICES=5 python cycle_gan.py --disc dc --train_iters 10000 --log_step=200 --g_conv_dim=64 --d_conv_dim=64 --sample_every=200 --use_diffaug --X apple2orange/apple --Y apple2orange/orange --use_cycle_consistency_loss --lambda_cycle=1

# apple/orange w/ ccl  patch
CUDA_VISIBLE_DEVICES=4 python cycle_gan.py --disc patch --train_iters 10000 --log_step=200 --g_conv_dim=64 --d_conv_dim=64 --sample_every=200 --use_diffaug --X apple2orange/apple --Y apple2orange/orange --use_cycle_consistency_loss --lambda_cycle=1


```
# cl/ti w/ ccl  patch
CUDA_VISIBLE_DEVICES=7 python cycle_gan.py --disc dc --train_iters 10000 --log_step=200 --g_conv_dim=64 --d_conv_dim=64 --sample_every=200 --use_diffaug --X GaitTights/CL_color --Y GaitTights/TI_color --use_cycle_consistency_loss --lambda_cycle=1 --ext *.jpg
```


```
# DDPM
python train_ddpm.py
python test_ddpm.py
```