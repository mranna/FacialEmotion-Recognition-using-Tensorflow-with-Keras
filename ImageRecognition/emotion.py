# Author: Mr. Hemanth Anil
# Project: Emotion Recognition.
# Student id: T00564146
# Date: 20th November 2019


import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import callbacks
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras.optimizers import *

"""
Data directory
ckplus -> CK+48 The directory that contains images in different folder
https://stackoverflow.com/questions/44209071/python-os-listdir-reading-subfodlers
Print the check if they are in the correct order
"""
data_path = 'ckplus/CK+48'
data_dir_list = os.listdir(data_path)
print(data_dir_list)

img_datalist = []
for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset- ' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        input_img_resize = cv2.resize(input_img, (128, 128))
        img_datalist.append(input_img_resize)

# Use numpy array
# https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
img_data = np.array(img_datalist)
img_data = img_data.astype('float32')

# Normalize the value
img_data = img_data / 255

print(img_data.shape)

# Number of classes ('ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE')
num_classes = 7
num_of_samples = img_data.shape[0]
print(num_of_samples)
# Return a new array of given shape and type, filled with ones.
labels = np.ones((num_of_samples,), dtype='int64')

labels[0:207] = 0  # 207
labels[207:261] = 1  # 54
labels[261:336] = 2  # 179
labels[336:585] = 3  # 76
labels[585:669] = 4  # 208
labels[669:804] = 5  # 85
labels[804:] = 6  # 238

# The labels - Change names according to the OS you use
# I found that they change in Windows. So make sure to print the label and
# follow that pattern.
names = ['happy', 'contempt', 'fear', 'surprise', 'sadness', 'anger', 'disgust']
#names = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'CONTEMPT', 'SAD', 'SURPRISE']



def getLabel(id):
    return names[id]

"""
np_utils.to_categorical
Converts a class vector (integers) to binary class matrix.
https://keras.io/utils/
"""
Y = np_utils.to_categorical(labels, num_classes)

x, y = shuffle(img_data, Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# test_size = 0.2
# test_size=0.25, random_state=42
x_test = X_test
print(x_test.shape)

# input_shape = (48, 48, 3) <---- Images are 48x48
input_shape = (128, 128, 3)
model = Sequential()


model.add(Conv2D(128, (5, 5), input_shape=input_shape, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])


"""
Checkpoint callback usage
https://keras.rstudio.com/articles/tutorial_save_and_restore.html
It is useful to automatically save checkpoints during and at the end of training. 
This way you can use a trained model without having to retrain it, 
or pick-up training where you left of, in case the training process was interrupted.
"""
filename = 'model_train_new.csv'
filepath = '{0}/Result/checkpoints/checkpoint-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5'
csv_log = callbacks.CSVLogger(filename, separator=',', append=False)
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [csv_log, checkpoint]
callbacks_list = [csv_log]

hist = model.fit(X_train, y_train, epochs=20, verbose=1, validation_data=(X_test, y_test),
                 callbacks=callbacks_list)
model.save_weights('model_weights.h5')
model.save('model_keras.h5')

"""
#Access Model Training History in Keras
https://stackoverflow.com/questions/36952763/how-to-return-history-of-validation-loss-in-keras
The history object is returned from calls to the fit() function used to train the model. 
Metrics are stored in a dictionary in the history member of the object returned.
"""
print("The keys in history:", hist.history.keys())

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

epochs = range(len(train_acc))

#  "Accuracy"
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# "Loss"
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# "Evaluate"

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print(test_image.shape)

print("The test_image: ", model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])

result = model.predict_classes(X_test[0:6])
print("Result ->(Emotions):-", result)
plt.figure(figsize=(10, 10))

# get the images to predict

for i in range(0, 6):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_test[i], cmap=plt.get_cmap('gray'))
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.ylabel('Prediction = %s' % getLabel(result[i]), fontsize=14)
# show the plot
plt.show()
