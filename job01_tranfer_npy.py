import scipy
from glob import glob
import numpy as np
import scipy.misc
from PIL import Image # pillow 설치


batch_size=2
is_testing=False

path_A = glob('./CycleGAN-master/dataset/A/*.jpg')
print(path_A)
path_B = glob('./CycleGAN-master/dataset/B/*.jpg')

# n_batches = int(min(len(path_A), len(path_B)) / batch_size)  # 실행할 배치의 개수
# total_samples = n_batches * batch_size  # 총 샘플수
# print('totlal:',total_samples) # 300개
# # Sample n_batches * batch_size from each path list so that model sees all
# # samples from both domains
# path_A = np.random.choice(path_A, total_samples, replace=False)  # 랜덤으로 골라 path_A로 만든다. 비복원 추출
# print(path_A.shape)
# print('d:1:',path_A[0:1])
# path_B = np.random.choice(path_B, total_samples, replace=False)  # 랜덤으로  골라 path_B로 만든다. 비복원 추출
# print(path_B.shape)
# img_res=(256, 256)
#
# image_w = 256
# image_h = 256
# for i in range(n_batches - 1):  #
#     batch_A = path_A[i * batch_size:(i + 1) * batch_size]
#     print('batch:',batch_A)
#     batch_B = path_B[i * batch_size:(i + 1) * batch_size]
#     imgs_A, imgs_B = [], []
#     for img_A, img_B in zip(batch_A, batch_B):
#         # img_A = imread(img_A)
#         # img_B = imread(img_B)
#         print('img_A:',img_A)
#         img_A = img_A.resize((image_w, image_h))
#         img_B = img_B.resize(img_res)
#         print(img_A)
#         if not is_testing and np.random.random() > 0.5:
#             img_A = np.fliplr(img_A)
#             img_B = np.fliplr(img_B)
#
#         imgs_A.append(img_A)
#         imgs_B.append(img_B)
#
#     imgs_A = np.array(imgs_A) / 127.5 - 1.
#     imgs_B = np.array(imgs_B) / 127.5 - 1.
#     print(imgs_A)
#     print(imgs_B)


from PIL import Image # pillow 설치
import glob
import numpy as np
from sklearn. model_selection import train_test_split


image_w = 256
image_h = 256

pixel = image_h * image_w * 3 # 칼라는 3칼라
X=[]
Y=[]

files = None


files = glob.glob('./CycleGAN-master/dataset/B/*.jpg')
for i, f in enumerate(files):
    try:
        img = Image.open(f)
        img = img.convert('RGB') # RGB모드로 변환
        img = img.resize((image_w, image_h)) #사이즈는 튜플로
        data = np.asarray(img) # 이미지를 어레이로 바꾼다.
        data = np.fliplr(data)

        X.append(data)

        if i % 300 == 0:
            print( ':', f)

    except: #에러가 나도 처리
        print( i, '')





X = np.array(X)
Y = np.array(Y)

X = np.array(X) / 127.5 - 1

print(X[0])
print(Y[:5])
Y_train, Y_test = train_test_split(X, test_size=0.1)
xy= (Y_train, Y_test)
np.save('./CycleGAN-master/dataset/picture_image.npy', xy)
print(len(X))