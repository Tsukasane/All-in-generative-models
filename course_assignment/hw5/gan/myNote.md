


For latent space optimization, you can choose to use different w for different layers, which is called w+

Randomly initialize $z$ latent, pass $z$ through the DCGenerator, use the optimization-based method to draw the image near the target.

The local minimas of data samples are different. 

The mapping function is only for stylegan, as it firstly brings out the latent w space.

```
python main.py --model vanilla --mode project --latent z 

python main.py --model stylegan --mode project --latent w+

python main.py --model stylegan --mode project --latent w+
```