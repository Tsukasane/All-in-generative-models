
- Kernel Size ``K``: 4
- Stride ``S``: 2
- Padding ``P``: 1
- Input Size ``I``: 64
- Output Size ``O``: 32

We have 
  

$$
    O = \frac{I-K+2P}{S} +1
$$

Given that we use kernel size K = 4 and stride S = 2, the padding P of ``conv1``~``conv4`` will be 1, and P of ``conv5`` will be 0.