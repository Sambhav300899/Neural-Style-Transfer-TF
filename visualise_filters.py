import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg19

def get_model(shape, layers):
    base = vgg19.VGG19(include_top = False, weights = 'imagenet', input_shape = shape)
    base.summary()

    #get the outputs from the model
    outputs = [base.get_layer(name).output for name in layers]

    model = Model(base.input, outputs)
    return model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('--img', help = 'path to image to visualise filters', required = True)

    args = vars(ap.parse_args())

    #the style and content layers that we use for style transfer
    layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1', 'block5_conv2']

    #lets load the image and resize it to make representation easier
    img = cv2.imread(args['img'])
    img = cv2.resize(img, (512, 512))
    model = get_model(img.shape, layers)
