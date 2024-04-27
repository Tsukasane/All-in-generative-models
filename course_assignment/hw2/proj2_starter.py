# --------------------------------------------------------
# Written by Yufei Ye and modified by Sheng-Yu Wang (https://github.com/JudyYe)
# Convert from MATLAB code https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj3/gradient_starter.zip
# --------------------------------------------------------
from __future__ import print_function

import argparse
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
import os

# one channel input 
def toy_recon(image):
    
    imh, imw = image.shape

    num_pix = imh * imw 
    im2var = np.arange(num_pix).reshape((imh, imw)).astype(int) # (imh, imw)

    # (Av-b)^2
    A = lil_matrix((imh * (imw - 1) + (imh - 1) * imw + 1, num_pix)) # num_pix - imh - imw + 1
    b = np.zeros((imh * (imw - 1) + (imh - 1) * imw + 1, 1))

    # obj1
    e = -1 # equation index
    for y in range(imh):
        for x in range(imw - 1):
            e += 1
            A[e, im2var[y, x + 1]] = 1
            A[e, im2var[y, x]] = -1
            b[e] = image[y, x + 1] - image[y, x]

    # obj2
    for y in range(imh - 1):
        for x in range(imw):
            e += 1
            A[e, im2var[y + 1, x]] = 1
            A[e, im2var[y, x]] = -1
            b[e] = image[y + 1, x] - image[y, x]

    # obj3
    e += 1
    A[e, im2var[0, 0]] = 0
    b[e] = image[0, 0]

    v = np.linalg.lstsq(A.toarray(), b)[0] # This will cost several minutes
    print(v)

    out_v = v.reshape((imh, imw))
    return out_v


def check_neighborhood(mask_image, pixel_coords):
    """
    border: at least one of the four neighbours is not inside the mask
    content: all neighbours are inside
    not_s_up: upper pixel is not inside
    """
    border = []
    content = []
    not_s_up = []
    not_s_down = []
    not_s_left = []
    not_s_right = []
    
    for coord in pixel_coords: # white pixels [(y1,x1),(y2,x2),...]
        y, x = coord
        in_mask = all(mask_image[y + dy, x + dx] > 0 for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)])
        
        in_up = all(mask_image[y + dy, x + dx] > 0 for dy, dx in [(-1, 0)])
        in_down = all(mask_image[y + dy, x + dx] > 0 for dy, dx in [(1, 0)])
        in_left = all(mask_image[y + dy, x + dx] > 0 for dy, dx in [(0, -1)])
        in_right = all(mask_image[y + dy, x + dx] > 0 for dy, dx in [(0, 1)])
        
        if in_mask:
            content.append(coord) # for obj1
        else:
            border.append(coord) # for obj2

        if not in_up:
            not_s_up.append(coord) # upper pixel not inside mask
        if not in_down:
            not_s_down.append(coord)
        if not in_left:
            not_s_left.append(coord)
        if not in_right:
            not_s_right.append(coord)

    return {'border':border, 'content':content, 'not_s_up':not_s_up, 'not_s_down':not_s_down, 'not_s_left':not_s_left, 'not_s_right':not_s_right}

def poisson_blend(fg, mask, bg):
    """
    Poisson Blending.
    :param fg: (H, W, C) source texture / foreground object
    :param mask: (H, W, 1)
    :param bg: (H, W, C) target image / background
    :return: (H, W, C)

    1st crop
    2nd fuse
    """

    imh, imw, num_ch = fg.shape # TODO channel
    out_v = np.zeros_like(fg)

    mask_info = np.where(mask==True) 
    y_mask = mask_info[0]
    x_mask = mask_info[1]

    # num of white pixels
    s_area = len(y_mask)
    white_pix_pos = []

    for i in range(len(y_mask)):
        white_pix_pos.append((y_mask[i],x_mask[i]))
    out_sta = check_neighborhood(mask, white_pix_pos)

    for ch in range(num_ch): 

        fg_c = fg[:,:,ch]
        bg_c = bg[:,:,ch]

        num_pix = imh * imw 
        # represent each position by index number
        im2var = np.arange(num_pix).reshape((imh, imw)).astype(int)
        
        content = out_sta['content'] # obj1
        border = out_sta['border'] # obj2
        all_white = []
        all_white.extend(content)
        all_white.extend(border)

        not_s_up = out_sta['not_s_up'] # obj1
        not_s_down = out_sta['not_s_down']
        not_s_left = out_sta['not_s_left']
        not_s_right = out_sta['not_s_right']

        # initialize A and b for (Av-b)^2
        # use the whole image to init, rather than only the masked region
        A = lil_matrix((4 * s_area, num_pix)) # 2 * (max_line + max_column) the upper bound of white adjacent, all pixels that are relevant to this calculation
        b = np.zeros((4 * s_area, 1))

        e = -1 # init equation index

        # min(grad(v) - grad(s))^2  s = fg_c
        for ele in all_white: # (y1, x1) r-l
            if ele in content or (ele in border and ele not in not_s_left):
                y, x = ele[0], ele[1]
                e += 1
                A[e, im2var[y, x + 1]] = 1 # i
                A[e, im2var[y, x]] = -1 # j
                b[e] = fg_c[y, x + 1] - fg_c[y, x]

        for ele in all_white: # (y1, x1) l-r
            if ele in content or (ele in border and ele not in not_s_right):
                y, x = ele[0], ele[1]
                e += 1
                A[e, im2var[y, x - 1]] = 1
                A[e, im2var[y, x]] = -1
                b[e] = fg_c[y, x - 1] - fg_c[y, x]

        for ele in all_white: # (y1, x1) u-d
            if ele in content or (ele in border and ele not in not_s_down):
                y, x = ele[0], ele[1]
                e += 1
                A[e, im2var[y + 1, x]] = 1 # i
                A[e, im2var[y, x]] = -1 # j
                b[e] = fg_c[y + 1, x] - fg_c[y, x]

        for ele in all_white: # (y1, x1) d-u
            if ele in content or (ele in border and ele not in not_s_up):
                y, x = ele[0], ele[1]
                e += 1
                A[e, im2var[y - 1, x]] = 1
                A[e, im2var[y, x]] = -1
                b[e] = fg_c[y - 1, x] - fg_c[y, x]


        # obj2 
        # min(grad(v,t) - grad(s))^2  s = fg
        # not_s_up, not_s_down, not_s_left, not_s_right
        for ele in not_s_left: # (y1, x1) r-l
            y, x = ele[0], ele[1]
            e += 1
            A[e, im2var[y, x + 1]] = 1 # i
            b[e] = fg_c[y, x + 1] - fg_c[y, x] + bg_c[y, x]

        for ele in not_s_right: # (y1, x1) l-r
            y, x = ele[0], ele[1]
            e += 1
            A[e, im2var[y, x - 1]] = 1
            b[e] = fg_c[y, x - 1] - fg_c[y, x] + bg_c[y, x]

        for ele in not_s_down: # (y1, x1) u-d
            y, x = ele[0], ele[1]
            e += 1
            A[e, im2var[y + 1, x]] = 1
            b[e] = fg_c[y + 1, x] - fg_c[y, x] + bg_c[y, x]

        for ele in not_s_up: # (y1, x1) d-u
            y, x = ele[0], ele[1]
            e += 1
            A[e, im2var[y - 1, x]] = 1
            b[e] = fg_c[y - 1, x] - fg_c[y, x] + bg_c[y, x]

        v = np.linalg.lstsq(A.toarray(), b)[0] # This will cost several minutes
        print(f'v shape', v.shape)

        out_v_sc = v.reshape((imh, imw))
        out_v[:,:,ch] = out_v_sc

    return out_v * mask + bg * (1 - mask)


def mixed_blend(fg, mask, bg):
    """EC: Mix gradient of source and target"""

    imh, imw, num_ch = fg.shape # TODO channel
    out_v = np.zeros_like(fg)

    mask_info = np.where(mask==True) 
    y_mask = mask_info[0]
    x_mask = mask_info[1]

    # num of white pixels
    s_area = len(y_mask)
    white_pix_pos = []

    for i in range(len(y_mask)):
        white_pix_pos.append((y_mask[i],x_mask[i]))
    out_sta = check_neighborhood(mask, white_pix_pos)

    for ch in range(num_ch): 

        fg_c = fg[:,:,ch]
        bg_c = bg[:,:,ch]

        num_pix = imh * imw 
        # represent each position by index number
        im2var = np.arange(num_pix).reshape((imh, imw)).astype(int)
        
        content = out_sta['content'] # obj1
        border = out_sta['border'] # obj2
        all_white = []
        all_white.extend(content)
        all_white.extend(border)

        not_s_up = out_sta['not_s_up'] # obj1
        not_s_down = out_sta['not_s_down']
        not_s_left = out_sta['not_s_left']
        not_s_right = out_sta['not_s_right']

        # initialize A and b for (Av-b)^2
        # use the whole image to init, rather than only the masked region
        A = lil_matrix((4 * s_area, num_pix)) # 2 * (max_line + max_column) the upper bound of white adjacent, all pixels that are relevant to this calculation
        b = np.zeros((4 * s_area, 1))

        e = -1 # init equation index

        # NOTE min(grad(v) - max(grad(s), grad(t)))^2 
        for ele in all_white: # (y1, x1) r-l
            if ele in content or (ele in border and ele not in not_s_left):
                y, x = ele[0], ele[1]
                e += 1
                A[e, im2var[y, x + 1]] = 1 # i
                A[e, im2var[y, x]] = -1 # j
                b[e] = max(fg_c[y, x + 1] - fg_c[y, x], bg_c[y, x + 1] - bg_c[y, x])

        for ele in all_white: # (y1, x1) l-r
            if ele in content or (ele in border and ele not in not_s_right):
                y, x = ele[0], ele[1]
                e += 1
                A[e, im2var[y, x - 1]] = 1
                A[e, im2var[y, x]] = -1
                b[e] = max(fg_c[y, x - 1] - fg_c[y, x], bg_c[y, x - 1] - bg_c[y, x])

        for ele in all_white: # (y1, x1) u-d
            if ele in content or (ele in border and ele not in not_s_down):
                y, x = ele[0], ele[1]
                e += 1
                A[e, im2var[y + 1, x]] = 1 # i
                A[e, im2var[y, x]] = -1 # j
                b[e] = max(fg_c[y + 1, x] - fg_c[y, x], bg_c[y + 1, x] - bg_c[y, x])

        for ele in all_white: # (y1, x1) d-u
            if ele in content or (ele in border and ele not in not_s_up):
                y, x = ele[0], ele[1]
                e += 1
                A[e, im2var[y - 1, x]] = 1
                A[e, im2var[y, x]] = -1
                b[e] = max(fg_c[y - 1, x] - fg_c[y, x], bg_c[y - 1, x] - bg_c[y, x])

        # obj2 
        # min(grad(v,t) - grad(s))^2  s = fg
        # not_s_up, not_s_down, not_s_left, not_s_right
        for ele in not_s_left: # (y1, x1) r-l
            y, x = ele[0], ele[1]
            e += 1
            A[e, im2var[y, x + 1]] = 1 # i
            b[e] = fg_c[y, x + 1] - fg_c[y, x] + bg_c[y, x]

        for ele in not_s_right: # (y1, x1) l-r
            y, x = ele[0], ele[1]
            e += 1
            A[e, im2var[y, x - 1]] = 1
            b[e] = fg_c[y, x - 1] - fg_c[y, x] + bg_c[y, x]

        for ele in not_s_down: # (y1, x1) u-d
            y, x = ele[0], ele[1]
            e += 1
            A[e, im2var[y + 1, x]] = 1
            b[e] = fg_c[y + 1, x] - fg_c[y, x] + bg_c[y, x]

        for ele in not_s_up: # (y1, x1) d-u
            y, x = ele[0], ele[1]
            e += 1
            A[e, im2var[y - 1, x]] = 1
            b[e] = fg_c[y - 1, x] - fg_c[y, x] + bg_c[y, x]

        v = np.linalg.lstsq(A.toarray(), b)[0] # This will cost several minutes
        print(f'v shape', v.shape)

        out_v_sc = v.reshape((imh, imw))
        out_v[:,:,ch] = out_v_sc

    return out_v * mask + bg * (1 - mask)


def color2gray(rgb_image):
    """Naive conversion from an RGB image to a gray image."""
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def mixed_grad_color2gray(rgb_image):
    """EC: Convert an RGB image to gray image using mixed gradients."""
    return np.zeros_like(rgb_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Poisson blending.")
    parser.add_argument("-q", "--question", required=True, choices=["toy", "blend", "mixed", "color2gray"])
    args, _ = parser.parse_known_args()

    os.makedirs('output', exist_ok=True)

    # Example script: python proj2_starter.py -q toy
    if args.question == "toy":
        image = imageio.imread('./data/toy_problem.png') / 255. # ./data/toy_problem.png
        if len(image.shape)==2: # 1 channel
            image_hat = toy_recon(image)
        else:
            for ch in range(image.shape[2]):
                # image_hat = image[:,:,0]
                image_hat = toy_recon(image[:,:,ch])
                break

        plt.subplot(121) # 1r-2c-the 1stc
        plt.imshow(image, cmap='gray')
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(image_hat, cmap='gray')
        plt.title('Output')
        plt.savefig('output/toy_solution.jpg')
        # plt.show()

    # Example script: 
    # python proj2_starter.py -q blend -s data/source_01_newsource.png -t data/target_01.jpg -m data/target_01_mask.png
    # python proj2_starter.py -q blend -s data/kt_middle_newsource.png -t data/kt_bg.png -m data/kt_bg_mask.png
    # python proj2_starter.py -q blend -s data/bubble_middle_newsource.png -t data/bubble_target.png -m data/bubble_target_mask.png
    if args.question == "blend":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1/4
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        if bg.shape[2]!=3:
            bg = cv2.cvtColor(bg, cv2.COLOR_RGBA2RGB)
    
        fg = fg / 255. # (333, 250, 3)
        bg = bg / 255. # (333, 250, 3)
        mask = (mask.sum(axis=2, keepdims=True) > 0) # (333, 250, 1)

        # crop the mask for efficiency
        mask_info = np.where(mask==True) 
        y_mask = mask_info[0]
        x_mask = mask_info[1]
        
        # retain a thin border
        crop_y_s = min(y_mask) - 1
        crop_y_e = max(y_mask) + 2
        crop_x_s = min(x_mask) - 1
        crop_x_e = max(x_mask) + 2

        fg_crop = fg[crop_y_s:crop_y_e, crop_x_s:crop_x_e, :]
        bg_crop = bg[crop_y_s:crop_y_e, crop_x_s:crop_x_e, :]
        mask_crop = mask[crop_y_s:crop_y_e, crop_x_s:crop_x_e, :]

        blend_crop = poisson_blend(fg_crop, mask_crop, bg_crop)

        blend_img = bg
        blend_img[crop_y_s:crop_y_e, crop_x_s:crop_x_e, :] = blend_crop

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Poisson Blend')
        plt.savefig('output/blend_kts.jpg')
        plt.show()


    # Example script
    # python proj2_starter.py -q mixed -s data/bubble_middle_newsource.png -t data/bubble_target.png -m data/bubble_target_mask.png
    if args.question == "mixed":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1/4
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        if bg.shape[2]!=3:
            bg = cv2.cvtColor(bg, cv2.COLOR_RGBA2RGB)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        # crop the mask for efficiency
        mask_info = np.where(mask==True) 
        y_mask = mask_info[0]
        x_mask = mask_info[1]
        
        # retain a thin border
        crop_y_s = min(y_mask) - 1
        crop_y_e = max(y_mask) + 2
        crop_x_s = min(x_mask) - 1
        crop_x_e = max(x_mask) + 2

        fg_crop = fg[crop_y_s:crop_y_e, crop_x_s:crop_x_e, :]
        bg_crop = bg[crop_y_s:crop_y_e, crop_x_s:crop_x_e, :]
        mask_crop = mask[crop_y_s:crop_y_e, crop_x_s:crop_x_e, :]

        blend_crop = mixed_blend(fg_crop, mask_crop, bg_crop)

        blend_img = bg
        blend_img[crop_y_s:crop_y_e, crop_x_s:crop_x_e, :] = blend_crop


        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Mixed Blend')
        plt.savefig('output/mix_bubble.jpg')
        plt.show()

    if args.question == "color2gray":
        parser.add_argument("-s", "--source", required=True)
        args = parser.parse_args()

        rgb_image = imageio.imread(args.source)
        gray_image = color2gray(rgb_image)
        mixed_grad_img = mixed_grad_color2gray(rgb_image)

        plt.subplot(121)
        plt.imshow(gray_image, cmap='gray')
        plt.title('rgb2gray')
        plt.subplot(122)
        plt.imshow(mixed_grad_img, cmap='gray')
        plt.title('mixed gradient')
        plt.show()

    plt.close()
