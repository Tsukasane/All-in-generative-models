import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as skio
import skimage as sk
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
    plt.savefig('output/waveform_grad.jpg')
    # plt.show()


# single scale search

imname = 'data/cathedral.jpg' # 'data/cathedral.jpg'

# read in the image NOTE skio and cv2.imread map to different numerical ranges
im = skio.imread(imname) 

# convert to double TODO(might want to do this later on to save memory)    
im = sk.img_as_float(im) # (1024, 390)
height = np.floor(im.shape[0] / 3.0).astype(int) # an approximation

# separate color channels
b = im[:height] # (341, 390) value[0.0,1.0]
g = im[height: 2*height] # (341, 390)
r = im[2*height: 3*height] # (341, 390)

print(f'single channel image shape {b.shape}')

def stack_and_save(r, g, b): # for visualization
    # to 8 bit uint/float for storing
    ag = np.interp(g, (0.0,1.0), (0.0,255.0)).astype(np.uint8)
    ar = np.interp(r, (0.0,1.0), (0.0,255.0)).astype(np.uint8)
    ab = np.interp(b, (0.0,1.0), (0.0,255.0)).astype(np.uint8)

    # create a color image (order matters)
    im_out = np.dstack([ar, ag, ab])

    # save the image
    os.makedirs('output/', exist_ok=True)
    fname = os.path.join('output/', f'overlap.jpg')
    skio.imsave(fname, im_out)

stack_and_save(r, g, b)

def cal_edge_canny(ipt):
    # 使用Canny算子检测边缘
    edges = sk.feature.canny(ipt, sigma=1)  

    # 转换为uint8类型
    edges = np.uint8(edges) * 255

    return edges


edge_r = cal_edge_canny(r)
edge_b = cal_edge_canny(b)
edge_g = cal_edge_canny(g)

# stack_and_save(edge_r, edge_g, edge_b)


import pdb
pdb.set_trace()

    # 显示结果
    # cv2.imshow('Original Image', im)
    # cv2.imshow('Sobel Edge Detection', gradient_magnitude)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
