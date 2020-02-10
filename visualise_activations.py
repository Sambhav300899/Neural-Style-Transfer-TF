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

    input = np.expand_dims(img.astype('float'), axis = 0)
    input = vgg19.preprocess_input(input)

    feature_maps = model(input)
    imgs_per_row = 32

    for layer, map in zip(layers, feature_maps):
        features = map.shape[-1]
        size = map.shape[1]
        n_cols = features // imgs_per_row

        display_grid = np.zeros((size * n_cols, imgs_per_row * size))

        for col in range(n_cols):
            for row in range(imgs_per_row):
                channel_image = map.numpy()[0, :, :, col * imgs_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype(np.uint8)

                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()
