import numpy as np
import cv2

def rgb_to_lms(image_rgb):
    # transfermation matrix
    rgb_to_lms_matrix = np.array([[0.3811, 0.5783, 0.0402],
                                   [0.1967, 0.7244, 0.0782],
                                   [0.0241, 0.1288, 0.8444]])

    image_lms = np.dot(image_rgb, rgb_to_lms_matrix.T) # (2569, 2962, 3) \dot (3,3)
    # image_lms = (np.dot(rgb_to_lms_matrix,image_lms.T)).T <-- eq.4 in the paper
    return image_lms # [4.19, 249.45]

def lms_to_lab(image_lms):
    m1 = np.array([[1/np.sqrt(3), 0, 0],
                    [0, 1/np.sqrt(6), 0],
                    [0, 0, 1/np.sqrt(2)]])
    m2 = np.array([[1, 1, 1],
                    [1, 1, -2],
                    [1, -1, 0]])
    im1 = np.dot(image_lms, m2.T)
    im2 = np.dot(im1, m1.T)

    return im2 # [-1.92, 9.46]

def lab_to_lms(image_lab):
    # np.linalg.inv
    m1_inv = np.linalg.inv(np.array([[1/np.sqrt(3), 0, 0],
                            [0, 1/np.sqrt(6), 0],
                            [0, 0, 1/np.sqrt(2)]]))
    m2_inv = np.linalg.inv(np.array([[1, 1, 1],
                            [1, 1, -2],
                            [1, -1, 0]]))
    
    im1 = np.dot(image_lab, m1_inv.T)
    im2 = np.dot(im1, m2_inv.T)

    return im2

def lms_to_rgb(image_lms):

    lms_to_rgb_matrix = np.linalg.inv(np.array([[0.3811, 0.5783, 0.0402],
                                   [0.1967, 0.7244, 0.0782],
                                   [0.0241, 0.1288, 0.8444]]))

    image_rgb = np.dot(image_lms, lms_to_rgb_matrix.T)
    
    return image_rgb

def color_correction(image):
    '''
    Shift the average of \alpha and \beta channel to 0, keep the average of l channel unchanged
    '''
    mean_a = np.mean(image[:,:,1])
    mean_b = np.mean(image[:,:,2])

    offset_a = -mean_a
    offset_b = -mean_b

    adjusted_image = image.copy()
    adjusted_image[:,:,1] += offset_a
    adjusted_image[:,:,2] += offset_b

    return adjusted_image

def hue_correction(im_file):

    filename = im_file.split('/')[-1].split('.')[0]
    # [0,255]
    image_rgb = cv2.imread(im_file)

    # RGB -> LMS
    image_lms = rgb_to_lms(image_rgb) # [4, 249]

    # LMS -> log(LMS)
    lms_log = np.log(image_lms) # <-- eq.5 in the paper [1.43, 5.51]

    # log(LMS) -> lab  
    image_lab = lms_to_lab(lms_log) # [-1.92, 9.46]

    # color processing
    image_lab1 = color_correction(image_lab) 

    lms_log1 = lab_to_lms(image_lab1) # [1.43, 5.51]

    # lab -> log(LMS) -> LMS
    image_lms1 = np.exp(lms_log1) # [4.19, 249.45]

    # LMS -> RGB
    image_rgb1 = lms_to_rgb(image_lms1) # [0, 255]

    cv2.imwrite(f'output/cor_{filename}.jpg', image_rgb1)


if __name__=='__main__':

    img = 'output/harvesters_edge_l5.jpg' # 'data/cathedral.jpg'
    hue_correction(img)



