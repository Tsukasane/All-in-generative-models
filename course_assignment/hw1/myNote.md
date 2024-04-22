# Assignment #1 - Colorizing the Prokudin-Gorskii Photo Collection

## Background
Sergei Mikhailovich Prokudin-Gorskii (1863-1944) [Сергей Михайлович Прокудин-Горский, to his Russian friends] traveled across the vast Russian Empire and take color photographs of everything he saw in 1907. He used an amazing technique which record three exposures of every scene onto a glass plate using a red, a green and a blue filter. This plates are envisioned to be integrated to color image by a projector and share with others. 

His RGB glass plate negatives, capturing the last years of the Russian Empire, survived and were purchased in 1948 by the Library of Congress. The LoC has recently digitized the negatives and made them available on-line.

<details>
<summary>
Here is LOC's processing method, which is very intriguing!
</summary>
  1. The entire plate is reduced to 8-bit grayscale mode. 
      2. Under magnification, the quality of each image on the plate is reviewed for 

  * contrast, 
  * degree of color separation, 
  * extent of damage to the emulsion, 
  * and any other details that might affect the final color composite.

    3. The scan of the entire plate is aligned and the outside edges are cropped.

    4. Use anchors to further align channels.

    5. Crop the overlapped image to only retain the area which three layers share in common

    6. The cropped color composite is adjusted overall to create 

  * the proper contrast
  * appropriate highlight 
  * shadow details 
  * optimal color balance

    7. Final adjustments may be applied to specific, localized areas of the composite color image 
       to minimize defects associated with over or underexposure, development, 
       or aging of the emulsion of Prokudin-Gorskii’s original glass plate.
       </details>

## Motivation
Design an algorithm to automatically align the 3 channels. 

## Naive Approach
The simplest inplementation cut the plate and overlaped the three parts, which has an obvious misplacement (or aliasing), calling for further alignment.

<div style="text-align:center">
    <img src="../figure/overlap.jpg" alt="img overlap" width="300" height="">
</div>

## Pixel Alignment
Initially, I used pixels to align the channels. 

I use the single scale matching first. It works well in ``cathedral.jpg``, which is probably the most easy sample because it is in compressed format. However, the searching speed for ``.tif`` files slows down drastically. 

To cut the time cost in 1 minute, I use the pyramid image scale with a flexible structure up to 5 levels, using ``sk.transform.rescale``. The number of levels depends on the input image size.

<div style="text-align:center">
    <img src="../figure/pyramid.png" alt="img pyramid" width="350" height="200">
</div>

Here lists some of my observations in ``Prokudin-Gorskii Photo Collection``. I use them as basic assumptions to design the algorithm. 

1. **Border is in a dark, frame-like structure.**
   The black and white borders of each channel differ. The inborn mismatching makes these borders ineffectual, even detrimental in alignment. Thus, they should be excised at the outset. 
   
    <div style="text-align:center">
        <img src="../figure/mismatch_flaw_borders.png" alt="img mismatch_flaw_borders" width="150" height="">
    </div>
   
    I designed an automatic algorithm to search the border using dark pixel ratio in scan windows.
   
    <div style="text-align:center">
        <img src="../figure/auto_edge.png" alt="img auto_edge" width="300" height="">
    </div>


2. **Three channels only have a small misplacement in initialization,**
   which makes sure most of the image region can be used to calculate loss.

These assumptions highly depend on the ``Prokudin-Gorskii Photo Collection`` domain characteristics. In other words, they may not be transferable to other domains, which is a pity.

In some test cases, pixel matching is enough to output a 'OK' result.


<div class="project-container">
    <div class="project-box">
        <img src="../figure/cathedral_ok.jpg" alt="image1" class="project-image">
        <p>Description of Project 1</p>
    </div>
    <div class="project-box">
        <img src="../figure/icon_ok.jpg" alt="Project 2" class="project-image">
        <p>Description of Project 2</p>
    </div>
    <div class="project-box">
        <img src="../figure/turkmen_ok.jpg" alt="Project 2" class="project-image">
        <p>Description of Project 3</p>
    </div>
    <div class="project-box">
        <img src="../figure/lady_ok.jpg" alt="Project 2" class="project-image">
        <p>Description of Project 3</p>
    </div>
    <div class="project-box">
        <img src="../figure/harvesters_ok.jpg" alt="Project 2" class="project-image">
        <p>Description of Project 3</p>
    </div>
    <!-- Add more project boxes here -->
</div>


I opted for ``np.roll`` to move the R and G channels as aligned as they could to the B channel. One thing worth mentioning is that, in the loss calculation, those moved pixels shouldn't be counted in. Say a channel is move up-left, then a small rectangle will appear in the down-right corner, which has no reason to be count.

<div style="text-align:center">
    <img src="../figure/np_roll.png" alt="img np_roll" width="380" height="">
</div>

I chose the **NCC** loss. It remove the effect of brightness by ``np.linalg.norm()``. For a 2D input, this operation is defined by dividing the input by $\|\mathbf{A}\|_F$, the Frobenius norm.

$$
\|\mathbf{A}\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2}
$$


Furthermore, due to the ``np.roll`` operation, the NCC should be normalized by the number of pixels used for calculating loss. (Only calculating the dot product is ok for inputs in a unanimous size, but not reasonable for inputs in various sizes).

On the contrary, SSD is not suitable to be averaged to each pixel, thus less flexible on various input sizes.

However, there are several hard cases in pixel matching. I think one reason is that they have a large portion of R/G/B color content. 

<div style="text-align:center">
    <img src="../figure/large_portion.png" alt="img large_portion" width="350" height="">
</div>

For instance, if a 3-channel red circle is divided into R/G/B grayscale images, the G/B channels might lack significant information while the R channel contains dense pixels, thereby complicating pixel search.

## Edge Alignment
Fortunately, many edge features are shared across channels. I initially employed the Sobel kernel, effective in detecting vertical and horizontal lines. Implementing gradient magnitude notably improved the reconstruction of previous failure cases.

<div style="text-align:center">
    <img src="../figure/self_portrait_pixel_l5.jpg" alt="img amplify" width="300" height="">
    <img src="../figure/self_portrait_edge_l5.jpg" alt="img amplify" width="300" height="">
</div>

<div style="text-align:center">
    <img src="../figure/three_generations_pixel_l5.jpg" alt="img amplify" width="300" height="">   
    <img src="../figure/three_generations_edge_l5.jpg" alt="img amplify" width="300" height="">
</div>

<div style="text-align:center">
    <img src="../figure/village_pixel_l5.jpg" alt="img amplify" width="300" height="">
    <img src="../figure/village_edge_l5.jpg" alt="img amplify" width="300" height="">
</div>

<div style="text-align:center">
    <img src="../figure/emir_pixel_l5.jpg" alt="img amplify" width="300" height="">
    <img src="../figure/emir_edge_l5.jpg" alt="img amplify" width="300" height="">
</div>

<div style="text-align:center">
    <img src="../figure/train_pixel_l5.jpg" alt="img amplify" width="300" height="">
    <img src="../figure/train_edge_l5.jpg" alt="img amplify" width="300" height="">
</div>

<div style="text-align:center">
    <p>left: Pixel Search -- right: Edge Search</p>
</div>

I also tried Canny kernel, but in ``emir.tif`` case it even preform worse than pixel search. Althought Canny can detect detailed edges and curves better than Sobel, these details are not shared by all channels. In contrast, Sobel focus on channel-unanimous straight (black) lines, resulting in better outcomes. Consequently, I chose Sobel for edge detection. 

<div style="text-align:center">
    <img src="../figure/sobel_edge.jpg" alt="img amplify" width="350" height="">
    <img src="../figure/canny_edge.jpg" alt="img amplify" width="350" height="">
</div>

<div style="text-align:center">
    <p>left: Sobel Kernel -- right: Canny Kernel</p>
</div>



Nonetheless, there is still some visible mismatching between channels. The blue artifacts are obvious here.

<div style="text-align:center">
    <img src="../figure/mismatch1.jpg" alt="img amplify" width="300" height="">
    <img src="../figure/mismatch2.jpg" alt="img amplify" width="300" height="">
</div>


One possible reason is the presence of animals (including humans), flowing water, or other movable objects in the image. As the images from the three channels of different filters were captured sequentially, natural mismatches may occur between these elements.

Another reason is that, after visualization, the edge features are not exactly the same across channels. 

<div style="text-align:center">
    <img src="../figure/edge_mismatch.jpg" alt="img amplify" width="500" height="">
</div>



#TODO try loop align (r,g)

#TODO try auto contrast