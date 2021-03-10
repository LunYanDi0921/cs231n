import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    plt.show()
    return

from cs231n.gan_tf import preprocess_img, deprocess_img, rel_error, count_params, MNIST
NOISE_DIM = 96
answers = np.load('gan-checks-tf.npz')
mnist = MNIST(batch_size=16)
show_images(mnist.X[:16])

from cs231n.gan_tf import leaky_relu
def test_leaky_relu(x, y_true):
    y = leaky_relu(tf.constant(x))
    print('Maximum error: %g'%rel_error(y_true, y))
    print('Maximum error: %e' % rel_error(y_true, y))            # for a more precise `error`

test_leaky_relu(answers['lrelu_x'], answers['lrelu_y'])

from cs231n.gan_tf import sample_noise
def test_sample_noise():
    batch_size = 3
    dim = 4
    z = sample_noise(batch_size, dim)
    # Check z has the correct shape
    assert z.get_shape().as_list() == [batch_size, dim]
    # Make sure z is a Tensor and not a numpy array
    assert isinstance(z, tf.Tensor)
    # Check that we get different noise for different evaluations
    z1 = sample_noise(batch_size, dim)
    z2 = sample_noise(batch_size, dim)
    assert not np.array_equal(z1, z2)
    # Check that we get the correct range
    assert np.all(z1 >= -1.0) and np.all(z1 <= 1.0)
    print("All tests passed!")
test_sample_noise()
#识别器
# Architecture:
# * Fully connected layer with input size 784 and output size 256
# * LeakyReLU with alpha 0.01
# * Fully connected layer with output size 256
# * LeakyReLU with alpha 0.01
# * Fully connected layer with output size 1
from cs231n.gan_tf import discriminator
def test_discriminator(true_count=267009, discriminator=discriminator):
    model = discriminator()
    cur_count = count_params(model)
    if cur_count != true_count:
        print('Incorrect number of parameters in discriminator. {0} instead of {1}. Check your achitecture.'.format(cur_count,true_count))
    else:
        print('Correct number of parameters in discriminator.')
test_discriminator()
#Generator
# Architecture:
# Fully connected layer with inupt size tf.shape(z)[1] (the number of noise dimensions) and output size 1024
# ReLU
# Fully connected layer with output size 1024
# ReLU
# Fully connected layer with output size 784
# TanH (To restrict every element of the output to be in the range [-1,1])
from cs231n.gan_tf import generator
def test_generator(true_count=1858320, generator=generator):
    model = generator(4)
    cur_count = count_params(model)
    if cur_count != true_count:
        print('Incorrect number of parameters in generator. {0} instead of {1}. Check your achitecture.'.format(cur_count,true_count))
    else:
        print('Correct number of parameters in generator.')
test_generator()

#GAN Loss
from cs231n.gan_tf import discriminator_loss

def test_discriminator_loss(logits_real, logits_fake, d_loss_true):
    d_loss = discriminator_loss(tf.constant(logits_real),
                                tf.constant(logits_fake))
    print("Maximum error in d_loss: %g"%rel_error(d_loss_true, d_loss))
test_discriminator_loss(answers['logits_real'], answers['logits_fake'],answers['d_loss_true'])

from cs231n.gan_tf import generator_loss

def test_generator_loss(logits_fake, g_loss_true):
    g_loss = generator_loss(tf.constant(logits_fake))
    print("Maximum error in g_loss: %g"%rel_error(g_loss_true, g_loss))

test_generator_loss(answers['logits_fake'], answers['g_loss_true'])

#Optimizing our loss
from cs231n.gan_tf import get_solvers
#Training a GAN!
from cs231n.gan_tf import run_a_gan

# Make the discriminator
D = discriminator()

# Make the generator
G = generator()

# Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
D_solver, G_solver = get_solvers()

# Run it!
images, final = run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss)
numIter = 0
for img in images:
    print("Iter: {}".format(numIter))
    show_images(img)
    plt.show()
    numIter += 20
    print()
print('Vanilla GAN Final images')
show_images(final)
plt.show()

from cs231n.gan_tf import ls_discriminator_loss, ls_generator_loss


def test_lsgan_loss(score_real, score_fake, d_loss_true, g_loss_true):
    d_loss = ls_discriminator_loss(tf.constant(score_real), tf.constant(score_fake))
    g_loss = ls_generator_loss(tf.constant(score_fake))
    print("Maximum error in d_loss: %g" % rel_error(d_loss_true, d_loss))
    print("Maximum error in g_loss: %g" % rel_error(g_loss_true, g_loss))
test_lsgan_loss(answers['logits_real'], answers['logits_fake'],
                answers['d_loss_lsgan_true'], answers['g_loss_lsgan_true'])

D = discriminator()

# Make the generator
G = generator()

# Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
D_solver, G_solver = get_solvers()

# Run it!
images, final = run_a_gan(D, G, D_solver, G_solver, ls_discriminator_loss, ls_generator_loss)

numIter = 0
for img in images:
    print("Iter: {}".format(numIter))
    show_images(img)
    plt.show()
    numIter += 20
    print()

print('LSGAN Final images')
show_images(final)
plt.show()

#Deep Convolutional GANs
# Architecture:
#
# Conv2D: 32 Filters, 5x5, Stride 1, padding 0
# Leaky ReLU(alpha=0.01)
# Max Pool 2x2, Stride 2
# Conv2D: 64 Filters, 5x5, Stride 1, padding 0
# Leaky ReLU(alpha=0.01)
# Max Pool 2x2, Stride 2
# Flatten
# Fully Connected with output size 4 x 4 x 64
# Leaky ReLU(alpha=0.01)
# Fully Connected with output size 1

from cs231n.gan_tf import dc_discriminator

# model = dc_discriminator()
test_discriminator(1102721, dc_discriminator)

#Generator
# Architecture:
#
# Fully connected with output size 1024
# ReLU
# BatchNorm
# Fully connected with output size 7 x 7 x 128
# ReLU
# BatchNorm
# Resize into Image Tensor of size 7, 7, 128
# Conv2D^T (transpose): 64 filters of 4x4, stride 2
# ReLU
# BatchNorm
# Conv2d^T (transpose): 1 filter of 4x4, stride 2
# TanH
from cs231n.gan_tf import dc_generator


test_generator(6595521, generator=dc_generator)
#Train and evaluate a DCGAN
D = dc_discriminator()

# Make the generator
G = dc_generator()

# Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
D_solver, G_solver = get_solvers()

# Run it!
images, final = run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, num_epochs=3)

numIter = 0
for img in images:
    print("Iter: {}".format(numIter))
    show_images(img)
    plt.show()
    numIter += 20
    print()

print('DCGAN Final images')
show_images(final)
plt.show()