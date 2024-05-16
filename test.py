import tensorflow as tf
import matplotlib.pyplot as plt
import keras

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import os,PIL,random,pathlib

# 设置随机种子尽可能使结果可以重现
import numpy as np
np.random.seed(123)

# 设置随机种子尽可能使结果可以重现
tf.random.set_seed(123)

### 必须得是绝对路径
data_dir = './data/train'
data_dir = pathlib.Path(data_dir)

all_image_paths = list(data_dir.glob('*'))
all_image_paths = [str(path) for path in all_image_paths]

# 打乱数据
#random.shuffle(all_image_paths)

# 获取数据标签
all_label_names = [path.split("\\")[-1].split("_")[0] for path in all_image_paths]

image_count = len(all_image_paths)
print("图片总数为：",image_count)

### 显示部分图片及其标签
plt.figure(figsize=(10, 5))
for i in range(20):
    plt.subplot(5, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    # 显示图片
    images = plt.imread(all_image_paths[i])
    plt.imshow(images)
    # 显示标签
    plt.xlabel(all_label_names[i])

plt.show()

### 用于图片对应标签的转换
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
char_set = number + alphabet
char_set_len = len(char_set)
label_name_len = len(all_label_names[0])

# 将字符串数字化
def text2vec(text):
   vector = np.zeros([label_name_len, char_set_len])
   for i, c in enumerate(text):
       idx = char_set.index(c)
       vector[i][idx] = 1.0
   return vector



all_labels = [text2vec(i) for i in all_label_names]


## 处理图片
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [71, 224])
    return image / 255.0


## 获取图片并转换
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


AUTOTUNE = tf.data.experimental.AUTOTUNE

path_data = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_data = path_data.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_data = tf.data.Dataset.from_tensor_slices(all_labels)
image_label_data = tf.data.Dataset.zip((image_data, label_data))
print(image_label_data)

##打乱数据
image_label_data = image_label_data.shuffle(buffer_size=image_count)

## 获取训练数据和验证数据
train_data = image_label_data.take(2800)  # 前1000个batch
val_data = image_label_data.skip(200)  # 跳过前1000，选取后面的

## 设置训练集和验证集的相关属性
batch_size = 64

train_data = train_data.batch(batch_size)
train_data = train_data.prefetch(buffer_size=AUTOTUNE)

val_data = val_data.batch(batch_size)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)
print(val_data)

# from tensorflow.keras import datasets, layers, models
# import  keras._tf_keras.keras.models
# import keras._tf_keras.keras.layers
# import keras._tf_keras.keras.datasets

model = keras.models.Sequential([

    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(71, 224, 3)),  # 卷积层1，卷积核3*3
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),  # 卷积层2，卷积核3*3
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),  # 卷积层2，卷积核3*3
    keras.layers.MaxPooling2D((2, 2)),  # 池化层2，2*2采样

    keras.layers.Flatten(),  # Flatten层，连接卷积层与全连接层
    # mobile_net,
    keras.layers.Dropout(0.5),  # 防止过拟合
    keras.layers.Dense(1000, activation='relu'),  # 全连接层，特征进一步提取

    keras.layers.Dense(label_name_len * char_set_len),
    keras.layers.Reshape([label_name_len, char_set_len]),
    keras.layers.Softmax()  # 输出层，输出预期结果
])
# 打印网络结构
# count=4
# device='cpu'
# model = model(count).to(device)
# model.train()
# print(model)
# print("")
model.summary()
epochs=70

# 模型编译
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# # 模型训练
# epochs = 20

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc,'b', label='Training Accuracy')
plt.plot(epochs_range, val_acc,'r',label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss,'b', label='Training Loss')
plt.plot(epochs_range, val_loss,'r', label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()




