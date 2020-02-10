import tensorflow as tf

def gram_matrix(input_tensor):
    '''
    calculates the gram matrix of a given input tensor

    args -
    input_tensor - input tensor
    '''
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def calc_style_loss(base_style, gram_target):
    '''
    calculates the style loss for a single layer

    args -
    base_style - the feature maps of the target image
    gram_target - the gram matrix of the style image
    '''
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (size ** 2))

def calc_content_loss(img_op, img_content):
    '''
    calculate the content loss for a single layer

    args -
    img_op : feature maps of target image
    img_content : feature maps of content image
    '''
    return tf.reduce_mean(tf.square(img_op - img_content))
