# IMPORTING REQUIRED PAKAGES
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Conv2D,Flatten
from keras.layers import MaxPool2D
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import warnings

warnings.filterwarnings('ignore')

# IMPORTING DATASETS
Train_ds=tf.keras.utils.image_dataset_from_directory("Dataset/train",batch_size=32,image_size=(180,180),seed=56)
Test_ds=tf.keras.utils.image_dataset_from_directory("Dataset/test",batch_size=32,image_size=(180,180),seed=56)
Valid_ds=tf.keras.utils.image_dataset_from_directory("Dataset/valid",batch_size=32,image_size=(180,180),seed=56)
Class_Names=Train_ds.class_names
print(Class_Names)

# HAVING A LOOK AT THE IMAGES OF TEST DATA
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for images, labels in Test_ds.take(1):
    for i in range(9):
        ax=plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(Class_Names[labels[i]])
        plt.axis("off")
plt.show()
for image_batch, labels_batch in Train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# FAST PROCESSING
AUTOTUNE = tf.data.AUTOTUNE
Train_ds = Train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
Valid_ds = Valid_ds.cache().prefetch(buffer_size=AUTOTUNE)
num_classes=len(Class_Names)

# BUILDING THE CNN,ANN MODEL
model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(128,3,padding='same',activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(256,3,padding='same',activation='relu'),
    layers.MaxPool2D(),
    layers.Dropout(.2),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(num_classes,activation='softmax')
])

# COMPILING THE MODEL
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print(model.summary())
history =model.fit(Train_ds,validation_data=Valid_ds,epochs=10)
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
print(acc,val_acc,loss,val_loss)
epochs_range = range(10)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# /kaggle/input/cards-image-datasetclassification/train
# /kaggle/input/cards-image-datasetclassification/valid
# /kaggle/input/cards-image-datasetclassification/test

# import pickle
# pickle.dump(SS,open('scalar.pkl','wb'))
# ssc=pickle.load(open('scalar.pkl','rb'))
from keras.models import load_model
model.save('model.h5')
model_final = load_model('model.h5')

