import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Helper functions to deal with image preprocessing
from cs231n.image_utils import load_image, preprocess_image, deprocess_image
from cs231n.classifiers.squeezenet import SqueezeNet

def check_scipy():
    import scipy
    version = scipy.__version__.split('.')
    if int(version[0]) < 1:
        assert int(version[1]) >= 16, "You must install SciPy >= 0.16.0 to complete this notebook."

check_scipy()

# Load pretrained SqueezeNet model
SAVE_PATH = None

# Local
SAVE_PATH = 'cs231n/datasets/squeezenet.ckpt'

# Colab
# SAVE_PATH = '/content/drive/My Drive/{}/{}'.format(FOLDERNAME, 'cs231n/datasets/squeezenet.ckpt')

assert SAVE_PATH is not None, "[!] Choose path to squeezenet.ckpt"

if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")

CHECKS_PATH = SAVE_PATH.replace('cs231n/datasets/squeezenet.ckpt', 'style-transfer-checks-tf.npz')
STYLES_FOLDER = CHECKS_PATH.replace('style-transfer-checks-tf.npz', 'styles')

model=SqueezeNet()
model.load_weights(SAVE_PATH)
model.trainable=False

# Load data for testing
content_img_test = preprocess_image(load_image('%s/tubingen.jpg' % (STYLES_FOLDER), size=192))[None]
style_img_test = preprocess_image(load_image('%s/starry_night.jpg' % (STYLES_FOLDER), size=192))[None]
answers = np.load('style-transfer-checks-tf.npz')

from cs231n.my_style_transfer_tensorflow import content_loss, extract_features, rel_error
def content_loss_test(correct):
    content_layer = 2
    content_weight = 6e-2
    c_feats = extract_features(content_img_test, model)[content_layer]
    bad_img = tf.zeros(content_img_test.shape)
    feats = extract_features(bad_img, model)[content_layer]
    student_output = content_loss(content_weight, c_feats, feats)
    error = rel_error(correct, student_output)
    print('Maximum error is {:.3f}'.format(error))
    print('Maximum error is %e' % error)             # for a more precise `error`

content_loss_test(answers['cl_out'])

#Style loss
from cs231n.my_style_transfer_tensorflow import gram_matrix

def gram_matrix_test(correct):
    gram = gram_matrix(extract_features(style_img_test, model)[4]) ### 4 instead of 5 - second MaxPooling layer
    error = rel_error(correct, gram)
    print('Maximum error is {:.3f}'.format(error))
    print('Maximum error is %e' % error)             # for a more precise `error`

gram_matrix_test(answers['gm_out'])

from cs231n.my_style_transfer_tensorflow import style_loss


def style_loss_test(correct):
    style_layers = [0, 3, 5, 6]
    style_weights = [300000, 1000, 15, 3]

    c_feats = extract_features(content_img_test, model)
    feats = extract_features(style_img_test, model)
    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(feats[idx]))

    s_loss = style_loss(c_feats, style_layers, style_targets, style_weights)
    error = rel_error(correct, s_loss)
    print('Error is {:.3f}'.format(error))
    print('Error is %e' % error)  # for a more precise `error`


style_loss_test(answers['sl_out'])

#Total-variation regularization
from cs231n.my_style_transfer_tensorflow import tv_loss
from inspect import getsourcelines
import re


def tv_loss_test(correct):
    tv_weight = 2e-2
    t_loss = tv_loss(content_img_test, tv_weight)
    error = rel_error(correct, t_loss)
    print('Error is {:.4f}'.format(error))
    print('Error is %e' % error)  # for a more precise `error`

    lines, _ = getsourcelines(tv_loss)
    used_loop = any(bool(re.search(r"for \S* in", line)) for line in lines)
    if used_loop:
        print(
            "WARNING!!!! - Your implementation of tv_loss contains a loop! To receive full credit, your implementation should not have any loops")


tv_loss_test(answers['tv_out'])

#Style Transfer
def style_transfer(content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, init_random=False):
    """Run style transfer!

    Inputs:
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    """
    # Extract features from the content image
    content_img = preprocess_image(load_image(content_image, size=image_size))
    feats = extract_features(content_img[None], model)
    content_target = feats[content_layer]

    # Extract features from the style image
    style_img = preprocess_image(load_image(style_image, size=style_size))
    s_feats = extract_features(style_img[None], model)
    style_targets = []
    # Compute list of TensorFlow Gram matrices
    for idx in style_layers:
        style_targets.append(gram_matrix(s_feats[idx]))

    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180
    max_iter = 200
    step = tf.Variable(0, trainable=False)
    boundaries = [decay_lr_at]
    values = [initial_lr, decayed_lr]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    # Later, whenever we perform an optimization step, we pass in the step.
    learning_rate = learning_rate_fn(step)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Initialize the generated image and optimization variables

    f, axarr = plt.subplots(1, 2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[0].imshow(deprocess_image(content_img), extent=(-5, 5, -5, 5))
    axarr[1].imshow(deprocess_image(style_img), extent=(-5, 5, -5, 5))
    plt.show()
    plt.figure()

    # Initialize generated image to content image
    if init_random:
        initializer = tf.random_uniform_initializer(0, 1)
        img = initializer(shape=content_img[None].shape)
        img_var = tf.Variable(img)
        print("Intializing randomly.")
    else:
        img_var = tf.Variable(content_img[None])
        print("Initializing with content image.")
    for t in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(img_var)
            feats = extract_features(img_var, model)
            # Compute loss
            c_loss = content_loss(content_weight, feats[content_layer], content_target)
            s_loss = style_loss(feats, style_layers, style_targets, style_weights)
            t_loss = tv_loss(img_var, tv_weight)
            loss = c_loss + s_loss + t_loss
        # Compute gradient
        grad = tape.gradient(loss, img_var)
        optimizer.apply_gradients([(grad, img_var)])

        img_var.assign(tf.clip_by_value(img_var, -1.5, 1.5))

        if t % 100 == 0:
            print('Iteration {}'.format(t))
            plt.imshow(deprocess_image(img_var[0].numpy(), rescale=True), extent=(-5, 5, -5, 5))
            plt.axis('off')
            plt.show()
    print('Iteration {}'.format(t))
    plt.imshow(deprocess_image(img_var[0].numpy(), rescale=True), extent=(-5, 5, -5, 5))
    plt.axis('off')
    plt.show()

params1 = {
    'content_image' : '%s/tubingen.jpg' % (STYLES_FOLDER),
    'style_image' : '%s/composition_vii.jpg' % (STYLES_FOLDER),
    'image_size' : 192,
    'style_size' : 512,
    'content_layer' : 2,
    'content_weight' : 5e-2,
    'style_layers' : (0, 2, 4, 5),                              # the original setting is (0, 3, 5, 6)
    'style_weights' : (20000, 500, 12, 1),
    'tv_weight' : 5e-2
}

style_transfer(**params1)

params2 = {
    'content_image':'%s/tubingen.jpg' % (STYLES_FOLDER),
    'style_image':'%s/the_scream.jpg' % (STYLES_FOLDER),
    'image_size':192,
    'style_size':224,
    'content_layer':2,
    'content_weight':3e-2,
    'style_layers':[0, 2, 4, 5],                          # the original setting is (0, 3, 5, 6)
    'style_weights':[200000, 800, 12, 1],
    'tv_weight':2e-2
}

style_transfer(**params2)

params3 = {
    'content_image' : '%s/tubingen.jpg' % (STYLES_FOLDER),
    'style_image' : '%s/starry_night.jpg' % (STYLES_FOLDER),
    'image_size' : 192,
    'style_size' : 192,
    'content_layer' : 2,
    'content_weight' : 5e-2,                                   # the original setting is 6e-2
    'style_layers' : [0, 2, 4, 5],                             # the original setting is (0, 3, 5, 6)
    'style_weights' : [300000, 1000, 15, 3],
    'tv_weight' : 2e-2
}

style_transfer(**params3)





