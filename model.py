import cv2
import argparse
import progressbar
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from utils import pre_proc, deprocess_img_vgg
from tensorflow.keras.applications import vgg19, vgg16
from losses import gram_matrix, calc_style_loss, calc_content_loss

class style_transfer():
    '''
    A class containing the functions realted to style transfer

    init args -
    model_name : can be different model types, eg vgg19, vgg16 etc.
    content_path : path to input content image
    style_path : path to input style image
    '''
    def __init__(self, model_name, content_path, style_path):
        #read images
        self.style = cv2.imread(style_path)
        self.content = cv2.imread(content_path)
        self.style = cv2.resize(self.style, (self.content.shape[1], self.content.shape[0]))

        self.h, self.w = self.style.shape[:2]

        #get the model
        self.make_model(model_name)

    def get_feature_maps(self):
        '''
        get the intermediate feature outputs required to
        calculate the style and content loss
        '''
        #run the model on style and content images
        output = self.model(self.create_ip())

        #collect style and content features
        style_features = [style_layer[0] for style_layer in output[len(self.content_layers):]]
        content_features = [content_layer[1] for content_layer in output[:len(self.content_layers)]]

        return style_features, content_features

    def set_loss_weights(self, content_weight, style_weight, variation_weight):
        '''
        set the weights for how much each loss contributes to the final loss

        args -
        content_weight : weight param for content loss
        style_weight : weight param for style loss
        variation_weight : weight param for variation loss
        '''
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.variation_weight = variation_weight

    def make_model(self, model_name):
        '''
        create the model according to the input model specified and
        select the layers to be used for style and content loss
        also set the loss weights according to the model

        args -
        model_name : type of model. e.g - vgg16, vgg19
        '''
        if model_name.lower() == 'vgg19':
            #get the pretrained model
            base = vgg19.VGG19(input_shape = (self.h, self.w, 3), include_top = False, weights = 'imagenet')

            #setting the content and style images
            self.content_layers = ['block5_conv2']
            self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

            #get the outputs from the model
            content_outputs = [base.get_layer(name).output for name in self.content_layers]
            style_outputs = [base.get_layer(name).output for name in self.style_layers]
            model_outputs = content_outputs + style_outputs

            #set loss weights, deprocessing function and processing function
            self.set_loss_weights(1e3, 1e-2, 30)
            self.de_proc_func = deprocess_img_vgg
            self.pre_processing_func = vgg19.preprocess_input

        elif model_name.lower() == 'vgg16':
            base = vgg16.VGG16(input_shape = (self.h, self.w, 3), include_top = False, weights = 'imagenet')

            self.content_layers = ['block5_conv3'] #, 'block5_conv2']
            self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

            content_outputs = [base.get_layer(name).output for name in self.content_layers]
            style_outputs = [base.get_layer(name).output for name in self.style_layers]
            model_outputs = content_outputs + style_outputs

            self.set_loss_weights(1e5, 1e-2, 1)
            self.de_proc_func = deprocess_img_vgg
            self.pre_processing_func = vgg16.preprocess_input

        #create the model
        self.model = Model(base.input, model_outputs)
        self.model.summary()

    def calc_loss(self, combined, gram_style_features, content_features):
        '''
        calculates the combined style and content loss

        args -
        combined : the output image
        gram_style_features : gram matrices calculated for style feature maps
        content_features : content feature maps
        '''
        #get model output for the combined image
        model_op = self.model(combined)

        #get the style and content output features for the combined image
        style_op_features = model_op[len(self.content_layers):]
        content_op_features = model_op[:len(self.content_layers)]

        style_score = 0
        content_score = 0

        #calculate style score over the feature maps
        weight_per_style_layer = 1. / float(len(self.style_layers))
        for target_style, comb_style in zip(gram_style_features, style_op_features):
            style_score = style_score + weight_per_style_layer * calc_style_loss(comb_style[0], target_style)

        #calculate content score over the feature maps
        weight_per_content_layer = 1. / float(len(self.content_layers))
        for target_content, comb_content in zip(content_features, content_op_features):
            content_score = content_score + weight_per_content_layer * calc_content_loss(comb_content[0], target_content)

        style_loss = self.style_weight * style_score
        content_loss = self.content_weight * content_score
        loss = style_loss + content_loss

        return loss, style_loss, content_loss

    def create_ip(self):
        '''
        creates input to a model and performs preprocessing
        steps for the content and style image
        '''
        #change to float and expand the dims
        content_ip = pre_proc(self.content)
        style_ip = pre_proc(self.style)

        #preprocessing function of the trained model
        content_ip = self.pre_processing_func(content_ip)
        style_ip = self.pre_processing_func(style_ip)

        return np.concatenate((style_ip, content_ip))

    def combine(self, num_iter, lr, output_path):
        '''
        function to combine the image. Calculates
        and updates the image by calculating the gradients

        args -
        num_iter : number of iterations to run the update loop
        lr : learing rate for the optimizer
        output_path : path to save the final image
        '''
        #expand dims and apply model pre processing
        combined = pre_proc(self.content)
        combined = self.pre_processing_func(combined)
        combined = tf.Variable(combined, dtype = tf.float32)

        #get the style and content features
        style_features, content_features = self.get_feature_maps()
        #get the gram matrices of the style features using the style feature maps
        gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
        #define out optimizer with given learning rate
        opt = Adam(lr)

        #values that the image needs to be clipped to after update
        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        img = None
        intermediate_images = []
        losses = []

        for i in progressbar.progressbar(range(num_iter)):
            #calculate loss and then the gradients for the combined image
            with tf.GradientTape() as tape:
                loss = self.calc_loss(combined, gram_style_features, content_features)
                variation_loss = self.variation_weight * tf.image.total_variation(combined).numpy()[0]

                total_loss, style_loss, content_loss = loss
                grads = tape.gradient(total_loss + variation_loss, combined)

            #apply the gradient updates
            opt.apply_gradients([(grads, combined)])
            clipped = tf.clip_by_value(combined, min_vals, max_vals)
            combined.assign(clipped)

            print (" total loss:{} style_loss:{} content_loss:{} variation_loss:{}".format(i,
                                        total_loss, style_loss, content_loss, variation_loss))
            cv2.imshow("image", self.de_proc_func(combined[0].numpy()))
            cv2.waitKey(1)

            losses.append(loss)
            img = combined.numpy()
            if i % (num_iter / 10) == 0:
                intermediate_images.append(self.de_proc_func(img[0]))

        plt.plot(range(0, num_iter), losses)
        cv2.imwrite(output_path, self.de_proc_func(img[0]))
        plt.savefig("loss.png")

        row_1 = intermediate_images[0]
        for i in range(1, 5):
            row_1 = np.hstack((row_1, intermediate_images[i]))

        row_2 = intermediate_images[5]
        for i in range(6, 10):
            row_2 = np.hstack((row_2, intermediate_images[i]))

        cv2.imwrite("intermediate_images.png", np.vstack((row_1, row_2)))
        plt.show()
