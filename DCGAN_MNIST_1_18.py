# coding=utf-8
# -*- coding: utf-8 -*-
""" Deep Convolutional Generative Adversarial Network (DCGAN).
Using deep convolutional generative adversarial networks (DCGAN) to generate
digit images from a noise distribution.
References:
    - Unsupervised representation learning with deep convolutional generative
    adversarial networks. A Radford, L Metz, S Chintala. arXiv:1511.06434.
Links:
    - [DCGAN Paper](https://arxiv.org/abs/1511.06434).
    - [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
# 프로그램: DCGAN MNIST
# 작성자 	:홍유진
# 목표	: GAN을 이용한 MNIST 데이터를 생성
#       : 실험시에는 mnist/data 에 있는 데이터를 사용해서 테스트
#       : 작성 및 테스트 완료 후(Default)에 layer 크기 비교하면서 테스트
#       : 기존에 만들어진 DCGAN을 활용해서 작성
# -------------------------------------------------------------------------------
# 참고한 코드들	: https://github.com/golbin/TensorFlow-Tutorials 	- CNN GAN
#       : https://github.com/yihui-he/GAN-MNIST	 	 	- GAN cnn
#       : https://github.com/carpedm20/DCGAN-tensorflow  	- //
#       : https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN 	- //
#       : https://github.com/hpssjellis/easy-tensorflow-on-cloud9/blob/master/aymericdamien-Examples/examples/alexnet.py
# -------------------------------------------------------------------------------
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dcgan.py#L33

from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import time
import os
import shutil
import sys

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# Training Params
num_steps = 5000
batch_size = 32
# Network Params
image_dim = 784         # 28*28 pixels * 1 channel
gen_hidden_dim = 256
dis_hidden_dim = 256
noise_dim = 200         # Noise data points
# Model Restore
to_restore = False
to_train = True
output_path = "model_dc"

# Generator Network
# Input: Noise, Output: Image
def generator(g_input, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        g_dense1 = tf.layers.dense(g_input, units=6 * 6 * 128)
        g_tan1 = tf.nn.tanh(g_dense1)
        g_reshape1 = tf.reshape(g_tan1, shape=[-1, 6, 6, 128])
        g_convTran1 = tf.layers.conv2d_transpose(g_reshape1, 64, 4, strides=2)
        g_convTran2 = tf.layers.conv2d_transpose(g_convTran1, 1, 2, strides=2)
        # Apply sigmoid to clip values between 0 and 1
        output = tf.nn.sigmoid(g_convTran2)
        return output

# Discriminator Network
# Input: Image, Output: Prediction Real/Fake Image
def discriminator(dis_input, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # Typical convolutional neural network to classify images.
        d_conv1 = tf.layers.conv2d(dis_input, 64, 5)
        d_tan1 = tf.nn.tanh(d_conv1)
        d_averaPool1 = tf.layers.average_pooling2d(d_tan1, 2, 2)
        d_conv2 = tf.layers.conv2d(d_averaPool1, 128, 5)
        d_tan2 = tf.nn.tanh(d_conv2)
        d_averaPool2 = tf.layers.average_pooling2d(d_tan2, 2, 2)
        d_flatten = tf.contrib.layers.flatten(d_averaPool2)
        d_dense = tf.layers.dense(d_flatten, 1024)
        d_tan3 = tf.nn.tanh(d_dense)
        # Output 2 classes: Real and Fake images
        output = tf.layers.dense(d_tan3, 2)
    return output

# Build Networks
# Network Inputs
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

# Build Generator Network
gen_sample = generator(noise_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
dis_real = discriminator(real_image_input)
dis_fake = discriminator(gen_sample, reuse=True)

# Build Loss
# dis_loss = loss_D_real + loss_D_gene --> 각각을 reduce_mean을 활용
dis_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=dis_real, labels=tf.ones_like(dis_real)))
dis_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=dis_fake, labels=tf.zeros_like(dis_fake)))
dis_loss = dis_real_loss + dis_fake_loss
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=dis_fake, labels=tf.ones_like(dis_fake)))

# Build Optimizers  (learning_rate=0.001)
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer_dis = tf.train.AdamOptimizer(learning_rate=0.001)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_dis = optimizer_dis.minimize(dis_loss, var_list=dis_vars)
# train_gen = gen_loss
# train_dis = dis_loss

# Session initializer
init = tf.global_variables_initializer()

# Model Saver
saver = tf.train.Saver()

def train():
    print("Start Training")
    with tf.Session() as sess:
        sess.run(init)
        # Model initializer & Load
        # saver = tf.train.Saver()
        # save_path = saver.save(sess, "model/model.ckpt")
        if to_restore:
            chkpt_fname = tf.train.latest_checkpoint(output_path)
            saver.restore(sess, chkpt_fname)
        else:
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.mkdir(output_path)

        for epoch in range(1, num_steps + 1):
            # Prepare Input Data
            # Get the next batch of MNIST data (only images are needed, not labels)
            batch_x, _ = mnist.train.next_batch(batch_size)
            batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
            # Generate noise to feed to the generator
            z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

            # Training
            # _, dl, 저장된변수 = sess.run([train_dis, dis_loss, 변수], 각 네트워크의 변수 값을 중간에 불러오고 싶다면
            # '저장된변수'를 따로 불러오는 식으로 해야할 듯
            _, dl = sess.run([train_dis, dis_loss],
                             feed_dict={real_image_input: batch_x, noise_input: z})
            _, gl = sess.run([train_gen, gen_loss],
                             feed_dict={noise_input: z})

            # Save model
            saver.save(sess, os.path.join(output_path, "model_dc"))

            # Print Log
            if epoch % 100 == 0 or epoch == 1:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (epoch, gl, dl))

            if epoch % 100 == 0 or epoch == 1:
                f, a = plt.subplots(4, 10, figsize=(10, 4))
                for i in range(10):
                    z = np.random.uniform(-1., 1., size=[4, noise_dim])
                    g = sess.run(gen_sample, feed_dict={noise_input: z})

                    for j in range(4):
                        img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2), newshape=(28, 28, 3))
                        a[j][i].set_axis_off()
                        a[j][i].set_axis_off()
                        a[j][i].imshow(img)
                plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
                plt.close(f)

def test():
    print("Start test")
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        chkpt_fname = tf.train.latest_checkpoint(output_path)
        saver.restore(sess, chkpt_fname)

        f, a = plt.subplots(4, 10, figsize=(10, 4))
        for i in range(10):
            z = np.random.uniform(-1., 1., size=[4, noise_dim])
            test_g = sess.run(gen_sample, feed_dict={noise_input: z})
            for j in range(4):
                img = np.reshape(np.repeat(test_g[j][:, :, np.newaxis], 3, axis=2), newshape=(28, 28, 3))
                a[j][i].set_axis_off()
                a[j][i].imshow(img)
        plt.savefig('test/test_image.png', bbox_inches='tight')
        plt.close(f)


if __name__ == '__main__':
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()
    else:
        print("원하는 함수명을 입력하세요 (train / test)")
