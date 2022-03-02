import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# os.mkdir("./image_gan") # 폴더 생성
# os.mkdir("./model") # 폴더 생성
img_shape = (256 , 256, 3) # 이미지 차원을 설정한다. 28*28*3
batch_size = 1

input_image = Input(shape=img_shape)
print(input_image)
conv_init = RandomNormal(0, 0.02)


def Generator():  # 생성자 모델

    def conv_layer_G(input_data, n_filter, kernel_size=3, padding="reflect", strides=2):
        if (padding == "reflect"):
            x = Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT'))(input_data)
            x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides,
                       kernel_initializer=conv_init)(x)
        else:
            x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides, padding="same",
                       kernel_initializer=conv_init)(input_data)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def res_block(input_data, n_filter, kernel_size=3, instance_norm=True, strides=1):
        x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'))(
            input_data)  # height 과 width에만  padding 추가한다.
        x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides,
                   kernel_initializer=conv_init)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'))(x)

        x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides,
                   kernel_initializer=conv_init)(x)

        x = BatchNormalization()(x)

        shortcut = input_data  ## x 잔차
        merged = Add()([x, shortcut])  # 잔차

        return merged

    def deconv_layer_G(input_data, n_filter, kernel_size=3, instance_norm=True, strides=2, padding="same"):

        x = Conv2DTranspose(filters=n_filter, kernel_size=kernel_size, strides=strides, padding=padding,
                            kernel_initializer=conv_init)(input_data)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    input_image = Input(shape=img_shape)
    ## Conv layer
    c7s1_32 = conv_layer_G(input_image, 32, kernel_size=7, strides=1)
    d64 = conv_layer_G(c7s1_32, 64, kernel_size=3, padding="same", strides=2)
    # d128 = conv_layer_G(d64, 128, kernel_size=3, padding="same", strides=2)
    ## resnet_9blocks
    # R128 = res_block(d64, 128)
    # R128 = res_block(R128, 128)
    # R128 = res_block(R128, 128)
    # R128 = res_block(R128, 128)
    # R128 = res_block(R128, 128)
    # R128 = res_block(R128, 128)
    # R128 = res_block(R128, 128)
    # R128 = res_block(R128, 128)
    # R128 = res_block(R128, 128)

    R128 = res_block(d64, 64)
    R128 = res_block(R128, 64)
    R128 = res_block(R128, 64)
    R128 = res_block(R128, 64)
    R128 = res_block(R128, 64)
    R128 = res_block(R128, 64)
    R128 = res_block(R128, 64)
    R128 = res_block(R128, 64)
    R128 = res_block(R128, 64)
    ## deConv layer
    u64 = deconv_layer_G(R128, 64, kernel_size=3, strides=2)  ## shape 버그가 있음
    # u32 = deconv_layer_G(u64, 32, kernel_size=3, strides=2)
    u32 = Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT'))(u64)
    c731_3 = Conv2D(3, kernel_size=7, strides=1, activation='tanh',
                    kernel_initializer=conv_init)(u32)  ## c731_3._keras_shape
    output = c731_3

    model = Model(inputs=input_image, outputs=output)
    print("## Generator ##")
    model.summary()

    return model
# def res_block(input_data, n_filter, kernel_size=3, instance_norm = True, strides=1):
#     x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1] , [1, 1] , [0, 0]], 'REFLECT'))(input_data) # height 과 width에만  padding 추가한다.
#     x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides,
#                kernel_initializer=conv_init)(x)
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
#
#
#     x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1],  [0, 0]], 'REFLECT'))(x)
#
#     x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides,
#                kernel_initializer=conv_init)(x)
#
#     x = BatchNormalization()(x)
#
#     shortcut = input_data  ## x 잔차
#     merged = Add()([x, shortcut]) # 잔차
#
#     return merged
# ########################################### 생성자 모델 ###################################
# # x = Sequential()
# x = Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT'))(input_image)
# x = Conv2D(filters=256, kernel_size=7, strides=1, kernel_initializer=conv_init)(input_image)
#
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
#
# x = Conv2D(filters=64, kernel_size=3, strides=2, padding="same", kernel_initializer=conv_init)(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
#
# x = Conv2D(filters=128, kernel_size=3, strides=2, padding="same", kernel_initializer=conv_init)(x) # 128개의 채널
# x = BatchNormalization()(x)
# x = Activation("relu")(x) # shape none 63, 63, 128
#
#
#
# ####### resnet_9blocks
# R128 = res_block(x, 128)
# R128 = res_block(R128, 128)
# R128 = res_block(R128, 128)
# R128 = res_block(R128, 128)
# R128 = res_block(R128, 128)
# R128 = res_block(R128, 128)
# R128 = res_block(R128, 128)
# R128 = res_block(R128, 128)
# R128 = res_block(R128, 128)
#
# x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same",kernel_initializer=conv_init)(R128)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
#
# x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same",kernel_initializer=conv_init)(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
#
# u32 = Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT'))(x)
# c731_3 = Conv2D(3, kernel_size=3, strides=1, activation='tanh',kernel_initializer=conv_init)(u32)  ## c731_3._keras_shape
#
# output = c731_3
#
# generator_model = Model(inputs=input_image, outputs=output)
# generator_model.summary()

print("## Generator ##")

################################## 판별자 모델 #########################################
# input_image = Input(shape=img_shape)
# x = Conv2D(filters=64, kernel_size=4, strides=2, padding='same', kernel_initializer=conv_init)(input_image)
#
# x = Conv2D(filters=128, kernel_size=4, strides=2, padding='same', kernel_initializer=conv_init)(x)
# x = BatchNormalization()(x)
# x = LeakyReLU(alpha=0.2)(x)
#
# x = Conv2D(filters=256, kernel_size=4, strides=2, padding='same', kernel_initializer=conv_init)(x)
# x = BatchNormalization()(x)
# x = LeakyReLU(alpha=0.2)(x)
#
# x = Conv2D(filters=512, kernel_size=4, strides=2, padding='same', kernel_initializer=conv_init)(x)
# x = BatchNormalization()(x)
# x = LeakyReLU(alpha=0.2)(x)
#
# output = Conv2D(1, kernel_size = 4, strides=1, padding = "same", kernel_initializer=conv_init)(x)
#
# discriminator_model = Model(inputs = input_image, outputs = output)
def Discriminator():
    def conv_layer_D(input_data, n_filter, kernel_size=4, instance_norm=True, strides=2, padding="same"):
        x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides, padding=padding,
                   kernel_initializer=conv_init)(input_data)
        if (instance_norm == True):
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    input_image = Input(shape=img_shape)

    C64 = conv_layer_D(input_image, 64, instance_norm=False)  ## C64
    print('x:', C64.shape)
    C128 = conv_layer_D(C64, 128, instance_norm=True)  ## C128
    print('y:', C128.shape)
    C256 = conv_layer_D(C128, 256, instance_norm=True)  ## C256
    C512 = conv_layer_D(C256, 512, instance_norm=True)  ## C512
    output = Conv2D(1, kernel_size=4, strides=1, padding="same", kernel_initializer=conv_init)(C512)

    model = Model(inputs=input_image, outputs=output)
    print("## Discriminator ##")


    return model
print("## Discriminator ##")
optimizer = Adam(0.0002, 0.5)
D_A = Discriminator()   ## mone class를 구분하는 Discriminator
D_A.compile(loss='mse', optimizer=optimizer)

print('#####################             D_B        #####################################')
D_B = Discriminator()
D_B.compile(loss='mse', optimizer=optimizer)

lambda_cycle = 10
print("#################### 생성자 ##########################")
G_AB = Generator()
G_BA = Generator() ## B class --> mone class  F 함수
G_AB.summary()
############################ 이미지 명명 ########################################

image_A = Input(shape=img_shape)  ## mone class 이미지
image_B = Input(shape=img_shape)  ## B class 이미지

# 가짜 이미지
fake_B = G_AB(image_A)  ## B class 가짜 생성 이미지
fake_A = G_BA(image_B)  ## mone class 가짜 생성 이미지
#복원 이미지
reconstruct_A = G_BA(fake_B)
reconstruct_B = G_AB(fake_A)

#identity
identity_A = G_BA(image_A)
identity_B = G_AB(image_B)
############################## 학습 옵션 ###################################
## Discriminator의 Weight 학습을 안시킴
D_A.trainable = False
D_B.trainable = False

valid_A = D_A(fake_A)  ## 가짜 mone 이미지를 Discriminator_A에 넣었을 때  점수  D_A(F(B)) (0~1)
valid_B = D_B(fake_B)  ## 가짜 B 이미지를 Discriminator_B에 넣었을 때  점수

combined = Model(inputs=[image_A, image_B], outputs = [valid_A, valid_B, reconstruct_A, reconstruct_B, identity_A, identity_B])

combined.compile(loss=["mse", "mse","mae", "mae","mae", "mae"], loss_weights = [1, 1, lambda_cycle, lambda_cycle,1, 1], optimizer = optimizer)


########################### 학습 ###############################################
img_rows = 256
img_cols = 256
channels = 3

# cyclegan.train(epochs=200, batch_size=1, sample_interval=200)
start_time = datetime.datetime.now()
patch = int(img_rows / 16)
D_patch = (patch, patch, 1)

real = np.ones((batch_size,) +  D_patch) # 1로 채운다.
print('real:',real.shape)

fake = np.zeros((batch_size,) + D_patch) # 0으로 채운다.

epochs = 5000
batch_size =1
sample_interval = 200

r, c = 2, 3
fig, axs = plt.subplots(r, c)
titles = ['Original', 'Translated', 'Reconstructed']

X_train, X_test = np.load('./CycleGAN-master/dataset/mone_image_data.npy', allow_pickle = True) # pickle 은 객체의 형태를 그대로 유지하며 저장
Y_train, Y_test = np.load('./CycleGAN-master/dataset/picture_image.npy', allow_pickle = True) # pickle 은 객체의 형태를 그대로 유지하며 저장

X_train = np.expand_dims(X_train, axis=1) #np 확장시켜서 넣는다.
Y_train = np.expand_dims(Y_train, axis=1) #np 확장시켜서 넣는다.
# print('x_train:', X_train[0].shape)
for epoch in range(epochs):
    for batch_i, (imgs_A, imgs_B) in enumerate(zip(X_train, Y_train)):

        fake_B = G_AB.predict(imgs_A)
        fake_A = G_BA.predict(imgs_B)


        #--------------
        # 판별기 학습
        #--------------
        if batch_i % 5 ==0:
            dA_loss_real = D_A.train_on_batch(imgs_A, real) # imgs_A를 real로 학습

            dA_loss_fake = D_A.train_on_batch(fake_A, fake) # fake_A를 fake로 학습

        dA_loss= 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = D_B.train_on_batch(imgs_B, real)
        dB_loss_fake = D_B.train_on_batch(fake_B, fake)

        dB_loss= 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # Total disciminator loss
        d_loss = 0.5 * np.add(dA_loss, dB_loss)
        # d_acc = 0.5 * np.add(dA_acc, dB_acc)
        # ------------------
        #  생성기 학습
        # ------------------
        D_A.trainable = False
        D_B.trainable = False
        g_loss = combined.train_on_batch([imgs_A, imgs_B],
                                              [real, real,
                                               imgs_A, imgs_B,
                                               imgs_A, imgs_B])

        elapsed_time = datetime.datetime.now() - start_time
        # Plot the progress
        print("[Epoch %d/%d] [Batch %d] [D loss: %f]   [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
              % (epoch, epochs,
                 batch_i,
                 d_loss,
                 g_loss[0],
                 np.mean(g_loss[1:3]),
                 np.mean(g_loss[3:5]),
                 np.mean(g_loss[5:6]),
                 elapsed_time))
    if epoch % 1 == 0:
        fake_B = G_AB.predict(X_train[2])
        fake_A = G_BA.predict(Y_train[2])
        reconstr_A = G_BA.predict(fake_B)
        reconstr_B = G_AB.predict(fake_A)
        gen_imgs = np.concatenate([X_train[2], fake_B, reconstr_A, Y_train[2], fake_A, reconstr_B])
        gen_imgs = 0.5 * gen_imgs + 0.4
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("./image_gans/images_%d.png" % (epoch))  ##pyplot figure 스타일로 저장하기
        G_AB.save('./models/cycle_gan_epoch{}.h5'.format(epoch))
