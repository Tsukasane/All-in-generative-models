## Content Reconstruction

Only use content loss, which is in the feature space, in this part.


You will be using the feature extractor of the model only (model.feature). You should also set the model to eval() mode. Write your code to append content loss to the end of specific layers (will be ablated soon) to optimize.


 In contrast with assignment 3 where we optimize the parameters of a neural network, in assignment 4 we fix the neural network and optimize the pixel values of the input image. 
 
 Here we use a quasi-newton optimizer LBFGS to optimize the image optimizer = optim.LBFGS([input_img.requires_grad_()]). The optimizer involves reevaluate your function multiple times so rather than a simple loss.backward(), we need to specify a hook closure that performs 1) clear the gradient, 2) compute loss and gradient 3) return the loss.


CUDA_VISIBLE_DEVICES=6 python run.py ./images/style/escher_sphere.jpeg ./images/content/phipps.jpeg