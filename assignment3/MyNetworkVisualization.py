import sys
import time, os, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from cs231n.classifiers.squeezenet import SqueezeNet
from cs231n.data_utils import load_tiny_imagenet
from cs231n.image_utils import preprocess_image, deprocess_image
from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#Pretrained Model
SAVE_PATH = None

# Local
SAVE_PATH = 'cs231n/datasets/squeezenet.ckpt'



assert SAVE_PATH is not None, "[!] Choose path to squeezenet.ckpt"

if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")
#keras.models.load_model() 读取网络、权重
#keras.models.load_weights() 仅读取权重
#load_model代码包含load_weights的代码，区别在于load_weights时需要先有模型、并且load_weights需要将权重数据写入到对应网络层的tensor中
model = SqueezeNet()
status = model.load_weights(SAVE_PATH)

model.trainable = False

#Load some ImageNet images
from cs231n.data_utils import load_imagenet_val
X_raw, y, class_names = load_imagenet_val(num=5)

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_raw[i])
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.gcf().tight_layout()
plt.show()

X = np.array([preprocess_image(img) for img in X_raw])
from cs231n.net_visualization_tensorflow import *
def show_saliency_maps(X, y, mask):
    mask = np.asarray(mask)
    Xm = X[mask]
    ym = y[mask]

    saliency = compute_saliency_maps(Xm, ym, model)

    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(deprocess_image(Xm[i]))
        plt.axis('off')
        plt.title(class_names[ym[i]])
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.title(mask[i])
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(10, 4)
    plt.show()

mask = np.arange(5)
show_saliency_maps(X, y, mask)

#Fooling Images
from cs231n.net_visualization_tensorflow import make_fooling_image

idx = 3
Xi = X[idx][None]
target_y = 6
X_fooling = make_fooling_image(Xi, target_y, model)

# Make sure that X_fooling is classified as y_target
scores = model(X_fooling)
assert tf.math.argmax(scores[0]).numpy() == target_y, 'The network is not fooled!'

# Show original image, fooling image, and difference
orig_img = deprocess_image(Xi[0])
fool_img = deprocess_image(X_fooling[0])
plt.figure(figsize=(12, 6))

# Rescale
plt.subplot(1, 4, 1)
plt.imshow(orig_img)
plt.axis('off')
plt.title(class_names[y[idx]])
plt.subplot(1, 4, 2)
plt.imshow(fool_img)
plt.title(class_names[target_y])
plt.axis('off')
plt.subplot(1, 4, 3)
plt.title('Difference')
plt.imshow(deprocess_image((Xi-X_fooling)[0]))
plt.axis('off')
plt.subplot(1, 4, 4)
plt.title('Magnified difference (10x)')
plt.imshow(deprocess_image(10 * (Xi-X_fooling)[0]))
plt.axis('off')
plt.gcf().tight_layout()
plt.show()

#Class visualization

from cs231n.net_visualization_tensorflow import class_visualization_update_step, jitter, blur_image


def create_class_visualization(target_y, model, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.

    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to jitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 1000)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)

    # We use a single image of random noise as a starting point
    X = 255 * np.random.rand(256, 256, 3)
    # There are a few options:
    # X = 255 * np.zeros((224, 224, 3))
    # X = 255 * np.random.rand(224, 224, 3)        # I suppose this has the best visualization effect
    # X = 255 * np.random.randn(224, 224, 3)       # harder to train
    X = preprocess_image(X)[None]

    loss = None  # scalar loss
    grad = None  # gradient of loss with respect to model.image, same size as model.image
    X = tf.Variable(X)
    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = np.random.randint(0, max_jitter, 2)
        X = jitter(X, ox, oy)
        X = class_visualization_update_step(X, model, target_y, l2_reg, learning_rate)
        # Undo the jitter
        X = jitter(X, -ox, -oy)
        # As a regularizer, clip and periodically blur

        X = tf.clip_by_value(X, -SQUEEZENET_MEAN / SQUEEZENET_STD, (1.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD)
        if t % blur_every == 0:
            X = blur_image(X, sigma=0.5)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            plt.imshow(deprocess_image(X[0]))
            class_name = "mixed"
            # for y in target_y:
            #     class_name += class_names[y] + " "
            plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
            plt.gcf().set_size_inches(6, 6)  # I've modified this
            plt.axis('off')
            plt.show()
    return X

target_y = [76, 187]
out = create_class_visualization(target_y, model, show_every=100)

