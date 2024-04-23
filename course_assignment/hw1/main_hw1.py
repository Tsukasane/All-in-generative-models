# (16-726): Project 1 starter Python code
# credit to https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj1/data/colorize_skel.py
# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images


import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
import os
import cv2


def visualize_rgb_wave(r, g, b):

    # cal pixel value and num of pixels each channel
    b_hist, b_bins = np.histogram(b, bins=256) # (256,) num of pixels in this magnitude  bin_edges, in a culmulative way
    g_hist, g_bins = np.histogram(g, bins=256)
    r_hist, r_bins = np.histogram(r, bins=256)

    # cal wave form each channel
    b_waveform = np.cumsum(b_hist) / np.sum(b_hist) # pixel until now / all pixel, presented in a culmulative format
    g_waveform = np.cumsum(g_hist) / np.sum(g_hist)
    r_waveform = np.cumsum(r_hist) / np.sum(r_hist)

    plt.figure(figsize=(10, 5))
    plt.plot(b_bins[:-1], b_waveform, color='blue', label='B')
    plt.plot(g_bins[:-1], g_waveform, color='green', label='G')
    plt.plot(r_bins[:-1], r_waveform, color='red', label='R')
    plt.title('Histogram in wave')
    plt.xlabel('pixel value')
    plt.ylabel('cumulative frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/waveform.jpg')
    # plt.show()

"""LOC's processing. Very intriguing!

1. The entire plate is then reduced to 8-bit grayscale mode. 

2. Under magnification, the quality of each image on the plate is reviewed for 
* contrast, 
* degree of color separation, 
* extent of damage to the emulsion, 
* and any other details that might affect the final color composite.

3. The scan of the entire plate is aligned and the outside edges are cropped.

4. Align the three layers by anchors

5. crop the overlapped image to only retain the area which three layers share in common

6. The cropped color composite is adjusted overall to create 
* the proper contrast, 
* appropriate highlight 
* and shadow detail, 
* and optimal color balance.

7. Final adjustments may be applied to specific, localized areas of the composite color image 
to minimize defects associated with over or underexposure, development, 
or aging of the emulsion of Prokudin-Gorskii’s original glass plate.


Hard Cases:
emir.tif (lots of textures)
self_portrait.tif (flow, scene moving blue channel)
turkmen.tif (green channel)
"""


# use image pyramid for searching   sk.transform.rescale (for multiscale) 
def pyramid(img):
    im_l3 = sk.transform.rescale(img, 1/2)
    im_l2 = sk.transform.rescale(img, 1/4)
    im_l1 = sk.transform.rescale(img, 1/8)
    im_l0 = sk.transform.rescale(img, 1/16) # 1D scaler

    return [im_l0, im_l1, im_l2, im_l3, img] # 1/16 --> 1/8 --> 1/4 --> 1/2 --> 1

# L2 norm also known as the Sum of Squared Differences (SSD) distance which is simply sum(sum((image1-image2).^2)) 
def L2_norm(ipt1, ipt2): # smaller is better
    SSD = np.sum((ipt1 - ipt2)**2)

    return SSD

# linear algebra  np.linalg.norm
def loss_NCC(ipt1, ipt2): # larger is better
    nm_ipt1 = ipt1 / np.linalg.norm(ipt1)
    nm_ipt2 = ipt2 / np.linalg.norm(ipt2)

    NCC = np.dot(nm_ipt1.flatten(), nm_ipt2.flatten()) # a single value ~ [-1,1]

    return (1-(1+NCC)/2)/(ipt1.shape[0]*ipt1.shape[1]) # (1+NCC)/2 ~ [0,1]  1- smaller better


# align the second and the third parts (G and R) to the first (B). 
# For each image, it is required to print the (x,y) displacement vector that was used to align the parts.
def align(ch1, ch2, search_range):
    '''
    Variables:
        ch1: (2D ndarray) adjustable channel
        ch2: (2D ndarray) reference channel
        border: (int) the meaningless edges

    Return:
        best_xy: (tuple) for moving ch1

    '''
    assert ch1.shape==ch2.shape
    search_range_x = (max(0,search_range[0][0]), min(ch1.shape[0], search_range[0][1]))
    search_range_y = (max(0,search_range[1][0]), min(ch1.shape[0], search_range[1][1]))

    print(f'image shape {ch1.shape}   search region: x{search_range_x}, y{search_range_y}')
    
    # initialization
    current_min = 1e6 
    best_xy = (0,0) 

    for cx in range(search_range_x[0], search_range_x[1]): 
        ch1_mv = np.roll(ch1, cx, axis=0) # height deviation

        for cy in range(search_range_y[0], search_range_y[1]):     
            ch1_mv = np.roll(ch1_mv, cy, axis=1) # width deviation

            ### Evaluation Metric
            # border = int(ch1.shape[1]/15) # if fixed, is ok
            # l2 = L2_norm(ch1_mv[border:ch1_mv.shape[0]-border, border:ch1_mv.shape[1]-border], ch2[border:ch2.shape[0]-border, border:ch2.shape[1]-border])
            # l2 = L2_norm(ch1_mv[0:ch1_mv.shape[0]-cx, 0:ch1_mv.shape[1]-cy], ch2[0:ch1_mv.shape[0]-cx, 0:ch1_mv.shape[1]-cy])
            # l2 = L2_norm(ch1_mv, ch2)
            if cx>ch1.shape[0]/2: # up
                if cy>ch1.shape[1]/2: # left
                    l2 = loss_NCC(ch1_mv[0:ch1_mv.shape[0]-cx, 0:ch1_mv.shape[1]-cy], ch2[0:ch1_mv.shape[0]-cx, 0:ch1_mv.shape[1]-cy])
                else: # right
                    l2 = loss_NCC(ch1_mv[0:ch1_mv.shape[0]-cx, cy:ch1_mv.shape[1]], ch2[0:ch1_mv.shape[0]-cx, cy:ch1_mv.shape[1]])
            else: # down
                if cy>ch1.shape[1]/2: # left
                    l2 = loss_NCC(ch1_mv[cx:ch1_mv.shape[0], 0:ch1_mv.shape[1]-cy], ch2[cx:ch1_mv.shape[0], 0:ch1_mv.shape[1]-cy])
                else: # right
                    l2 = loss_NCC(ch1_mv[cx:ch1_mv.shape[0], cy:ch1_mv.shape[1]], ch2[cx:ch1_mv.shape[0], cy:ch1_mv.shape[1]])

            # update the best
            if l2 < current_min:
                current_min = l2
                best_xy = (cx, cy)
                
    print(f'best (x, y) displacement is {best_xy}')
    return best_xy


def stack_and_save(r, g, b, filename, level): # for visualization
    # to 8 bit uint/float for storing
    ag = np.interp(g, (0.0,1.0), (0.0,255.0)).astype(np.uint8)
    ar = np.interp(r, (0.0,1.0), (0.0,255.0)).astype(np.uint8)
    ab = np.interp(b, (0.0,1.0), (0.0,255.0)).astype(np.uint8)

    # create a color image (order matters)
    im_out = np.dstack([ar, ag, ab])

    if level != 5:
        filename = 'cut_and_stack'

    # save the image
    os.makedirs('output/', exist_ok=True)
    fname = os.path.join('output/', f'{filename}_pixel_l{level}.jpg')
    skio.imsave(fname, im_out)

    print(f'level {level} results saved!')


def detect_canny(ipt):
    
    edges = sk.feature.canny(ipt, sigma=1)  
    edges = np.uint8(edges) * 255

    return edges


def detect_sobel(ipt):
    """
    Using Sobel kernel to detect edge features on image of the three channels.
    Variables:
        ipt: single channel image input [0.0, 1.0]
    Return:
        gradient_magnitude: gradient_magnitude correspond to the input image. [0.0, 255.0]   
    """
    # Sobel kernel (seperate)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # use conv to do filter
    sobel_x_filtered = cv2.filter2D(ipt, -1, sobel_x)
    sobel_y_filtered = cv2.filter2D(ipt, -1, sobel_y)

    # cal gradient to represent edge
    gradient_magnitude = np.sqrt(sobel_x_filtered**2 + sobel_y_filtered**2) # >=0

    # [0.0, 1.0] --> [0, 255]
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    # float -> uint
    gradient_magnitude = gradient_magnitude.astype(np.uint8)

    return gradient_magnitude # [0, 255]


def auto_edge(ch, vertical_scan=True, increase=True): 

    '''
    Using a rectangle sliding window to detect dark edges (the black rectangle frame outside the main image content)
    '''
    step = 3 # window moving step
    start = 0
    dark_thres = 0.2 # defination of dark
    edge_thres = 0.2 # dark ratio to be an edge

    if vertical_scan:
        window_height = int(ch.shape[0]/20)
        window_width = ch.shape[1]
        end = int(ch.shape[0]/2) - window_height + 1
        if not increase:
            start = ch.shape[0]
            step = -step
    else:
        window_height = ch.shape[0]
        window_width = int(ch.shape[1]/20)
        end = int(ch.shape[1]/2) - window_width + 1
        if not increase:
            start = ch.shape[1]
            step = -step

    # 初始化结果列表
    dark_pixel_ratios = []
    print(f"search from {start} to {end} at step {step}")

    # scan 1/2 image
    for y in range(start, end, step):
        # initialize the window
        if vertical_scan:
            if increase:
                window = r[y:y+window_height, :]
                # print(f'window range = {(y, y+window_height)}')
            else:
                window = r[end+(start-y):end+(start-y)+window_height, :]
                # print(f'window range = {(end+(start-y), end+(start-y)+window_height)}')
        else:
            if increase:
                window = r[:, y:y+window_width]
                # print(f'window range = {(y, y+window_width)}')
            else:
                window = r[:, end+(start-y):end+(start-y)+window_width]
                # print(f'window range = {(end+(start-y), end+(start-y)+window_width)}')


        # calcualte the ratio of dark pixels
        dark_pixels = np.sum(window < dark_thres)  # 假设深色像素的阈值为128
        total_pixels = window_height * window_width
        dark_pixel_ratio = dark_pixels / total_pixels

        # print(f'dark_pixel_ratio {dark_pixel_ratio}')
        if dark_pixel_ratio>edge_thres:
            if start>end:
                dark_pixel_ratios.append(end-(y-start))
            else:
                dark_pixel_ratios.append(y)

    # print("dark pixel ratio list:", dark_pixel_ratios)
    if dark_pixel_ratios==[]:   
        edge = 0

    else:
        edge_end = -1
        if increase:
            edge_start = min(dark_pixel_ratios)
            for i in range(len(dark_pixel_ratios)):
                if dark_pixel_ratios[i] == edge_start + step * i:
                    continue
                else:
                    edge_end = dark_pixel_ratios[i]
            if edge_end == -1:
                edge_end = dark_pixel_ratios[-1]
            edge = abs(edge_start - edge_end)
            
        else:
            edge_start = max(dark_pixel_ratios)
            for i in range(len(dark_pixel_ratios)):
                if dark_pixel_ratios[len(dark_pixel_ratios)-1-i] == edge_start + step * i:
                    continue
                else:
                    edge_end = dark_pixel_ratios[len(dark_pixel_ratios)-1-i]
            if edge_end == -1:
                edge_end = dark_pixel_ratios[0]
            edge = abs(edge_start - edge_end)

    print(f"edge: {edge}")
    return edge


if __name__ == '__main__':
    # name of the input file
    # imname = 'data/emir.tif' # 'data/cathedral.jpg' self_portrait

    one_img_test = "emir.tif"
    im_folder = 'data'

    if one_img_test:
        imgs = [one_img_test]

    else:
        imgs = os.listdir(im_folder)

    for img in imgs:
        imname = os.path.join(im_folder, img)
        filename = img.split('.')[0]
    
        # read in the image 
        # NOTE skio and cv2.imread map to different numerical ranges
        im = skio.imread(imname) 

        # convert to double (might want to do this later on to save memory)    
        im = sk.img_as_float(im) 

        ######## A rough split
        height = np.floor(im.shape[0] / 3.0).astype(int) # an approximation

        # separate color channels
        b = im[:height] # value[0.0,1.0]
        g = im[height: 2*height] 
        r = im[2*height: 3*height] 

        # use edge feature or raw pixels 
        edge_feature = False

        # the waveform
        visualize_rgb_wave(r, g, b)

        ######## auto border detection
        height_border_auto = max(auto_edge(r, vertical_scan=True, increase=False), auto_edge(r, vertical_scan=True, increase=True),
                                auto_edge(b, vertical_scan=True, increase=False), auto_edge(b, vertical_scan=True, increase=True),
                                auto_edge(g, vertical_scan=True, increase=False), auto_edge(g, vertical_scan=True, increase=True))
        
        width_border_auto = max(auto_edge(r, vertical_scan=False, increase=False), auto_edge(r, vertical_scan=False, increase=True),
                                auto_edge(g, vertical_scan=False, increase=False), auto_edge(g, vertical_scan=False, increase=True),
                                auto_edge(b, vertical_scan=False, increase=False), auto_edge(b, vertical_scan=False, increase=True))

        height_border_limit = int(r.shape[0]/10)
        width_border_limit = int(r.shape[1]/10)

        # in case automatical edge detection failed
        height_border = min(height_border_auto, height_border_limit)
        width_border = min(width_border_auto, width_border_limit)

        print(f'cut off edges {(height_border, width_border)}')

        # excise edges first    
        cl_b = b[height_border:b.shape[0]-height_border, width_border:b.shape[1]-width_border] 
        cl_g = g[height_border:g.shape[0]-height_border, width_border:g.shape[1]-width_border] 
        cl_r = r[height_border:r.shape[0]-height_border, width_border:r.shape[1]-width_border] 

        pyramid_level = 5 - (2+int(height/1000))

        ######## If use edge feature to align
        if edge_feature:
            edge_r = detect_sobel(cl_r) # sobel
            edge_b = detect_sobel(cl_b)
            edge_g = detect_sobel(cl_g)
            # edge_r = detect_canny(cl_r) # canny
            # edge_b = detect_canny(cl_b)
            # edge_g = detect_canny(cl_g)

            # Pyramid Image List 5 scales [1/16, 1/8, 1/4, 1/2, 1]
            img_ls_r = pyramid(edge_r)[pyramid_level:]
            img_ls_g = pyramid(edge_g)[pyramid_level:]
            img_ls_b = pyramid(edge_b)[pyramid_level:]

        ######## Pixel features
        else:
            img_ls_r = pyramid(cl_r)[pyramid_level:]
            img_ls_g = pyramid(cl_g)[pyramid_level:]
            img_ls_b = pyramid(cl_b)[pyramid_level:]

        # import pdb
        # pdb.set_trace()

        ######## Parameters initialization
        multipler = 2 # pyramid scaler
        prevch1_xy = (0,0) # 2 flexible channels, 1 reference channel
        prevch2_xy = (0,0)

        search_range_ch1x = (0,img_ls_r[0].shape[0])
        search_range_ch1y = (0,img_ls_r[0].shape[1])

        search_range_ch2x = (0,img_ls_g[0].shape[0]) 
        search_range_ch2y = (0,img_ls_g[0].shape[1])

        # the coarse match must be relatively precise
        nextl_range = 5
        print(f'adjacent searching box for the next level {nextl_range}')

        ######## Search from coarse to fine -- speed up!
        for level in range(len(img_ls_r)):

            prevch1_xy = align(img_ls_r[level], img_ls_b[level], (search_range_ch1x, search_range_ch1y))
            prevch2_xy = align(img_ls_g[level], img_ls_b[level], (search_range_ch2x, search_range_ch2y))

            # current displacement will influence the next level image in the pyramid
            search_range_ch1x = (prevch1_xy[0]*multipler-nextl_range, prevch1_xy[0]*multipler+nextl_range)
            search_range_ch1y = (prevch1_xy[1]*multipler-nextl_range, prevch1_xy[1]*multipler+nextl_range)
            
            search_range_ch2x = (prevch2_xy[0]*multipler-nextl_range, prevch2_xy[0]*multipler+nextl_range)
            search_range_ch2y = (prevch2_xy[1]*multipler-nextl_range, prevch2_xy[1]*multipler+nextl_range)

            ar = np.roll(img_ls_r[level], prevch1_xy, axis=(0,1)) 
            ag = np.roll(img_ls_g[level], prevch2_xy, axis=(0,1)) 
            ab = img_ls_b[level]

            stack_and_save(ar, ag, ab, filename, level)

        ######## if using edge feature to align, then save pixel results
        # if use pixel search, this result will be similar to the previous level's
        pix_r = np.roll(cl_r, prevch1_xy, axis=(0,1)) 
        pix_g = np.roll(cl_g, prevch2_xy, axis=(0,1)) 
        pix_b = cl_b

        stack_and_save(pix_r, pix_g, pix_b, filename, level+1)

        ######## save best moves
        save_content = f'{filename} - R best{prevch1_xy}  G best{prevch2_xy} \n'
        with open("best_record_pixel.txt", "a") as file:
            file.write(save_content)  # 将内容写入文件

        # # display the image
        # skio.imshow(im_out)
        # skio.show()