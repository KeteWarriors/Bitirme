import tensorflow as tf
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from mtcnn.mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# cpu - gpu ayarı
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(
    per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

num_features = 64
num_classes = 7
batch_size = 32
epochs = 8


def extract_face_from_image(image_path, required_size=(48, 48)):

    image = plt.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)
    face_images = ""
    for face in faces:
        # extract the bounding box from the requested face
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face_boundary = image[y1:y2, x1:x2]

        # resize pixels to the model size
        face_image = Image.fromarray(face_boundary)
        face_image = ImageOps.grayscale(face_image)
        face_image = face_image.resize(required_size)
        face_images = face_image
        """face_array = np.asarray(face_image)
        face_images.append(face_array)"""

    return face_images


with open("./data/Veriseti2.csv") as f:
    content = f.readlines()
lines = np.array(content)
num_of_instances = lines.size
print("number of instances: ", num_of_instances)
print("instance length: ", len(lines[1].split(",")[1].split(" ")))
x_train, y_train, x_test, y_test = [], [], [], []
for i in range(1, num_of_instances):
    try:
        emotion, img, usage = lines[i].split(",")
        val = img.split(" ")
        pixels = np.array(val, 'float32')
        emotion = keras.utils.to_categorical(emotion, num_classes)
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
        print("", end="")
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')
x_train /= 255
x_test /= 255
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
model = Sequential()
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(
    48, 48, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(
    3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(2*num_features, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9,
                             beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss', factor=0.9, patience=3, verbose=1)
tensorboard = TensorBoard(log_dir='./logs')
early_stopper = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(
    './models', monitor='val_loss', verbose=1, save_best_only=True)
fit = False
if fit == True:
    model.fit(
        x_train, y_train, epochs=100, batch_size=32, shuffle=True, verbose=1, validation_data=(x_test, y_test), callbacks=[lr_reducer, tensorboard, early_stopper, checkpointer])
    model.save_weights(save_format='h5', filepath='./models/mymodel')
else:
    model.load_weights(
        './models/mymodel')
score = model.evaluate(x_test, y_test, verbose=0)
"""print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])"""
train_score = model.evaluate(x_train, y_train, verbose=0)
"""print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])"""


def emotion_analysis(emotions):
    objects = ('Kızgın', 'İğrenmiş', 'Korku', 'Mutlu',
               'Üzgün', 'Şaşırmış', 'Normal')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    emotions = emotions.tolist()
    print(objects[emotions.index(max(emotions))])
    return objects[emotions.index(max(emotions))]


"""monitor_testset_results = False

if monitor_testset_results == True:
    model=train_func()
    predictions = model.predict(x_test)

    index = 0
    for i in predictions:
        if index < 30 and index >= 20:
            # print(i)
            # print(y_test[index])

            testing_img = np.array(x_test[index], 'float32')

            plt.gray()
            plt.imshow(testing_img)
            plt.show()

            print(i)

            emotion_analysis(i)
            print("----------------------------------------------")
        index = index + 1"""


def karsilastir(img_path):

    foto = extract_face_from_image(img_path)
    x = image.img_to_array(foto)
    x = np.expand_dims(x, axis=0)
    x /= 255
    custom = model.predict(x)
    print(custom)
    sonuc = emotion_analysis(custom[0])
    return sonuc
