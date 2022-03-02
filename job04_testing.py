import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib
import copy
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# os.makedirs('images/%s' % self.dataset_name, exist_ok=True)


X_train, X_test = np.load('./CycleGAN-master/datasets/mone_image.npy', allow_pickle=True)  # pickle 은 객체의 형태를 그대로 유지하며 저장
Y_train, Y_test = np.load('./CycleGAN-master/datasets/picture_image.npy', allow_pickle=True)
print(X_train[0])

# imgs_A = (X_test +1) * 2
# plt.imshow(imgs_A[5])
# plt.show()
# exit()
imgs_A = X_train
imgs_B = Y_train
imgs_A = np.expand_dims(imgs_A, axis=1) #np 확장시켜서 넣는다.
imgs_B = np.expand_dims(imgs_B, axis=1) #np 확장시켜서 넣는다.
# imgs_A = self.data_loader.load_data(domain="mone", batch_size=1, is_testing=True)
# imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

# Demo (for GIF)
# imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
# imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

# Translate images to the other domain

G_BA = load_model('cycle_gan_epoch200.h5')
G_AB = load_model('cycle_gan_epoch200.h5')
fake_B = G_AB.predict(imgs_A[2])
fake_A = G_BA.predict(imgs_B[2])

fake_B_s = np.squeeze(fake_B, axis=0)
fake_A_s = np.squeeze(fake_A, axis=0)

# Translate back to original domain
reconstr_A = G_BA.predict(fake_B)
reconstr_B = G_AB.predict(fake_A)

reconstr_B_s = np.squeeze(reconstr_B, axis=0)
reconstr_A_s = np.squeeze(reconstr_A, axis=0)

gen_imgs = np.concatenate([imgs_A[2], fake_B, reconstr_A, imgs_B[2], fake_A, reconstr_B])


# Rescale images 0 - 1
gen_imgs = 0.5 * gen_imgs + 0.5



titles = ['Original', 'Translated', 'Reconstructed']
r, c = 2, 3
fig, axs = plt.subplots(r, c)
# plt.imshow(gen_imgs[2])
# plt.imshow(gen_imgs[1])
# plt.imshow(gen_imgs[2])
# plt.imshow(gen_imgs[3])
# plt.show()
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i, j].imshow(gen_imgs[cnt])
        axs[i, j].set_title(titles[j])
        axs[i, j].axis('off')
        cnt += 1
plt.show()
plt.close()