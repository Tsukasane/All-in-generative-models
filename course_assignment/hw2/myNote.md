# Assignment #2 - Gradient Domain Fusion

## Toy Problem

This problem requires a basic implementation of calculating (discrete) gradients and forming the least square error constraints in a matrix $({\bf A}v-{\bf b})^2=0$.

We want to reconstruct a image with only the up-left corner pixel information. Although there is no other direct intensity of other pixels, we can use a 2-direction gradient to simulate the original image, and get a similar result $v$.

Here are three objective functions:

1. $
    \operatorname{argmin}((v(x+1, y)-v(x,y)) - (s(x+1, y)-s(x,y)))^2
$ -- in the same pixel line, gradient(right - left) should be close to the source image.

2. $
    \operatorname{argmin}((v(x, y+1)-v(x,y)) - (s(x, y+1)-s(x,y)))^2
$ -- in the same pixel column, gradient(down - up) should be close to the source image.

3. $
    \operatorname{argmin}(v(0,0)-s(0,0))^2
$ -- copy the pixel on the top-left corner of the source image and paste to the reconstruct $v$.

Having finished the parameter matrix $A$ and the bias matrix $b$, I use ``np.linalg.lstsq`` to solve the appropriate $v$. The result is shown as follows.

## Poisson Blending
The blending constraint is 
$$
    {\bf v} = \operatorname{argmin_v} \sum_{i\in S, j\in N_i \cap S}((v_i - v_j)-(s_i-s_j))^2 + \sum_{i\in S, j\in N_i \cap \lnot S}((v_i-t_j)-(s_i-s_j))^2
$$

Similar to the toy problem, it can be turned to 2 objectives. 

1. $
    \operatorname{argmin_v} \sum_{i\in S, j\in N_i \cap S}((v_i - v_j)-(s_i-s_j))^2
$

2. $
    \operatorname{argmin_v} \sum_{i\in S, j\in N_i \cap \lnot S}((v_i-t_j)-(s_i-s_j))^2
$ 

Also, background of the target image should be directly copied to the reconstructed one. The third objective can be written as the format below, but there is actually no need to put these constraints into the matrix solving process:

3. $
    \operatorname{argmin_v} \sum_{i\in \lnot S}((v_i-t_i))^2
$ 

In the code implementation, I use ``1`` and ``2`` constraints, which are expected to used by different pixels.