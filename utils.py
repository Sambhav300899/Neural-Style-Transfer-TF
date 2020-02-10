import numpy as np

def pre_proc(img):
    '''
    pre process the image to input to the network

    args -
    img : input image to be pre processed
    '''
    img = np.array(img).astype('float')
    img = np.expand_dims(img, axis = 0)

    return img

def deprocess_img_vgg(processed_img):
    '''
    de process the image after
    passing through a VGG model

    args -
    preprocessed_image - image to be de processed
    '''
    x = processed_img.copy()

    if len(x.shape) == 4:
        x = np.squeeze(x, 0)

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x
