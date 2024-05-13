## Content Reconstruction

The model ``VGG19`` is fixed and trimed here as a feature encoder, and the target image is detached from the computational graph. So the whole purpose of the gradient update is to lead the input_img using the collection of loss in one forward pass to update itself through the backpropagation. The class ContentLoss is implemented as a transparent layer, where the input is not modified, but the contentloss regard to the embedded target in same shape is calculated and stored. It is conducted in the feature space, rather than the pixel space. 

Firstly, I tried to add this ContentLoss layer after conv5. Worth noticing that the small learning rate is important, where the default value in LBFGS is 1, making the updating pace to fast and the process easy to fall into local minimum. After some test, I use ``lr=0.01``.

The ultimate loss is ``4.395685195922852``, and reconstructed image from noise is:
Using conv5 makes it dark, but better in reconstructing object details.
<img >

Using feature in different layers to cal loss leads to different results.


You will be using the feature extractor of the model only (model.feature). You should also set the model to eval() mode. Write your code to append content loss to the end of specific layers (will be ablated soon) to optimize.


 In contrast with assignment 3 where we optimize the parameters of a neural network, in assignment 4 we fix the neural network and optimize the pixel values of the input image. 
 
 Here we use a quasi-newton optimizer LBFGS to optimize the image optimizer = optim.LBFGS([input_img.requires_grad_()]). The optimizer involves reevaluate your function multiple times so rather than a simple loss.backward(), we need to specify a hook closure that performs 1) clear the gradient, 2) compute loss and gradient 3) return the loss.


CUDA_VISIBLE_DEVICES=6 python run.py ./images/style/escher_sphere.jpeg ./images/content/phipps.jpeg
