import tensorflow as tf
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K
class InstanceNormalization(Layer):
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#######################################################################################################################



from keras.layers import Lambda
from keras.initializers import RandomNormal
from keras.datasets import mnist
# from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.activations import relu
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib
matplotlib.use('Agg')
import copy
import matplotlib.pyplot as plt
import sys

import numpy as np
import os




OUT_DIR = './CNN_OUT_img/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
img_shape = (28, 28, 3) # ????????? ????????? ????????????. 28*28*3
epoch = 10000 # ????????? 1??????
batch_size = 128
noise = 100
sample_interval = 100
# ????????? ??????
generator_model = Sequential()
input_image = Input(shape=img_shape) #??????
print(input_image)
conv_init = RandomNormal(0, 0.02)

def res_block(input_data, n_filter, kernel_size=3, strides=1):
    x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'))(input_data)
    x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides,
               kernel_initializer=conv_init)(x)
    x = InstanceNormalization()(x)
    x = Activation("relu")(x)

    x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'))(x)
    x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides,
               kernel_initializer=conv_init)(x)
    x = InstanceNormalization()(x)
    shortcut = input_data  ## x
    merged = Add()([x, shortcut])
    return merged
#######conv layer
# x = Sequential()
x = Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT'))(input_image)
x = Conv2D(filters=32, kernel_size=7, strides=1, kernel_initializer=conv_init)(x)

x = InstanceNormalization()(x)
x = Activation("relu")(x)

x = Conv2D(filters=64, kernel_size=3, strides=2, padding="same", kernel_initializer=conv_init)(x)
x = InstanceNormalization()(x)
x = Activation("relu")(x)

x = Conv2D(filters=128, kernel_size=3, strides=2, padding="same", kernel_initializer=conv_init)(x)
x = InstanceNormalization()(x)
x = Activation("relu")(x)


####### resnet_9blocks
R128 = res_block(x, 128)
R128 = res_block(R128, 128)
R128 = res_block(R128, 128)
R128 = res_block(R128, 128)
R128 = res_block(R128, 128)
R128 = res_block(R128, 128)
R128 = res_block(R128, 128)
R128 = res_block(R128, 128)
R128 = res_block(R128, 128)

x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same",kernel_initializer=conv_init)(R128)
x = InstanceNormalization()(x)
x = Activation("relu")(x)

x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same",kernel_initializer=conv_init)(R128)
x = InstanceNormalization()(x)
x = Activation("relu")(x)

u32 = Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT'))(x)
c731_3 = Conv2D(3, kernel_size=7, strides=1, activation='tanh',kernel_initializer=conv_init)(u32)  ## c731_3._keras_shape

output = c731_3

generator_model = Model(inputs=input_image, outputs=output)
generator_model.summary()
generator_model2 = copy.deepcopy(generator_model)
print("## Generator ##")

########### ????????? ?????? ####################
input_image = Input(shape=img_shape)
instance_norm = True
x = Conv2D(filters=64, kernel_size=4, strides=2, padding='same', kernel_initializer=conv_init)(input_image)

x = Conv2D(filters=128, kernel_size=4, strides=2, padding='same', kernel_initializer=conv_init)(input_image)
x = InstanceNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)

x = Conv2D(filters=256, kernel_size=4, strides=2, padding='same', kernel_initializer=conv_init)(input_image)
x = InstanceNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)

x = Conv2D(filters=512, kernel_size=4, strides=2, padding='same', kernel_initializer=conv_init)(input_image)
x = InstanceNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)

output = Conv2D(1, kernel_size = 4, strides=1, padding = "same", kernel_initializer=conv_init)(x)

discriminator_model = Model(inputs = input_image, outputs = output)
discriminator_model2 = copy.deepcopy(discriminator_model)
print("## Discriminator ##")
discriminator_model.summary()
####################### ????????? ????????? ?????? ###############################################
optimizer = Adam(0.0002, 0.5)
D_A = discriminator_model   ## mone class??? ???????????? Discriminator
D_A.compile(loss='mse', optimizer=optimizer)
D_B = discriminator_model2
D_B.compile(loss='mse', optimizer=optimizer)
lambda_cycle = 10

G_AB = generator_model  ## mone class --> B class  G ??????
G_BA = generator_model2 ## B class --> mone class  F ??????

############################ ????????? ?????? ########################################
# ??????
image_A = Input(shape=img_shape)  ## mone class ?????????
image_B = Input(shape=img_shape)  ## B class ?????????
# ?????? ?????????
fake_B = G_AB(image_A)  ## B class ?????? ?????? ?????????
fake_A = G_BA(image_B)  ## mone class ?????? ?????? ?????????
#?????? ?????????
reconstruct_A = G_BA(fake_B)
reconstruct_B = G_AB(fake_A)

#identity
identity_A = G_BA(image_A)
identity_B = G_AB(image_B)
############################## ?????? ?????? ###################################
## Discriminator??? Weight ????????? ?????????
D_A.trainable = False
D_B.trainable = False

valid_A = D_A(fake_A)  ## ?????? mone ???????????? Discriminator_A??? ????????? ???  ??????  D_A(F(B)) (0~1)
valid_B = D_B(fake_B)  ## ?????? B ???????????? Discriminator_B??? ????????? ???  ??????

combined = Model(inputs=[image_A, image_B], outputs = [valid_A, valid_B, reconstruct_A, reconstruct_B, identity_A, identity_B])

combined.compile(loss=["mse", "mse","mae", "mae","mae", "mae"], loss_weights = [1, 1, lambda_cycle, lambda_cycle,1, 1],
optimizer = optimizer)


########################### ?????? ###############################################
img_rows = 256
img_cols = 256
channels = 3

# cyclegan.train(epochs=200, batch_size=1, sample_interval=200)
start_time = datetime.datetime.now()
patch = int(img_rows / 2**4)
D_patch = (patch, patch, 1)

real = np.ones((batch_size,) +  D_patch) # 1??? ?????????.
fake = np.zeros((batch_size,) + D_patch) # 0?????? ?????????.
epochs = 200
batch_size =1
sample_interval = 200


X_train, X_test = np.load('./CycleGAN-master/dataset/mone_image_data.npy', allow_pickle = True) # pickle ??? ????????? ????????? ????????? ???????????? ??????
Y_train, Y_test = np.load('./CycleGAN-master/dataset/picture_image_data.npy', allow_pickle = True) # pickle ??? ????????? ????????? ????????? ???????????? ??????
for epoch in range(epochs):
    for batch_i, (imgs_A, imgs_B) in enumerate(zip(X_train, Y_train)):
        fake_B = G_AB.predict(imgs_A)
        fake_A = G_BA.predict(imgs_B)

        #--------------
        # ????????? ??????
        #--------------

        dA_loss_real = D_A.train_on_batch(imgs_A, real) # imgs_A??? real??? ??????
        dA_loss_fake = D_A.train_on_batch(fake_A, fake) # fake_A??? fake??? ??????
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = D_B.train_on_batch(imgs_B, real)
        dB_loss_fake = D_B.train_on_batch(fake_B, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # Total disciminator loss
        d_loss = 0.5 * np.add(dA_loss, dB_loss)

        # ------------------
        #  Generator ??????
        # ------------------

        g_loss = combined.train_on_batch([imgs_A, imgs_B],
                                              [real, real,
                                               imgs_A, imgs_B,
                                               imgs_A, imgs_B])

        elapsed_time = datetime.datetime.now() - start_time
        # Plot the progress
        print("[Epoch %d/%d] [Batch %d] [D loss: %f] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
              % (epoch, epochs,
                 batch_i,
                 d_loss,
                 g_loss[0],
                 np.mean(g_loss[1:3]),
                 np.mean(g_loss[3:5]),
                 np.mean(g_loss[5:6]),
                 elapsed_time))

        # If at save interval => save generated image samples
        # if batch_i % sample_interval == 0:
        #     sample_images(epoch, batch_i)

generator_model.save('./generator_mnist_{}.h5'.format(1))
generator_model2.save('./generator_mnist_{}.h5'.format(2))