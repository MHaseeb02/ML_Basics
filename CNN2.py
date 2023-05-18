#training models with CNN using small databases
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models 
# import matplotlib.pyplot as plt
# from keras.preprocessing import image
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import ImageDataGenerator 

#Data Augmentataio ka old method

# (train_images , train_labels),(test_images , test_labels) = datasets.cifar10.load_data()  #loadng and spliting the data

# train_images, test_images = train_images/255.0 ,test_images/255.0   #makining sure the values are between 0 and 1

# #now we will create a data generator object that will transfer images
# datagen = ImageDataGenerator(
#     rotation_range = 40,
#     width_shift_range= 0.2,
#     height_shift_range = 0.2,
#     shear_range= 0.2,
#     zoom_range = 0.2,
#     horizontal_flip = True,
#     fill_mode= 'nearest'
# )

# test_img = train_images[14]
# img = image.img_to_array(test_img)
# img=img.reshape((1,) + img.shape)

# i = 0

# for batch in datagen.flow(img, save_prefix ='test', save_format = "jpeg" ):
#     plt.figure(i)
#     plot = plt.imshow(img.img_to_array(batch[0]))
#     i += 1
#     if i >= 4:
#         break
# plt.show()

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
keras = tf.keras

URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

class_names = train_dataset.class_names
    
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)  #this is the 20% data from the validation dataset
validation_dataset = validation_dataset.skip(val_batches // 5) # subtracting the 20% data

AUTOTUNE = tf.data.AUTOTUNE                                      #autotuining the datsets for better prefrmance
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([                      #using data augmentation for better data analyzing
  tf.keras.layers.RandomFlip('horizontal'), 
  tf.keras.layers.RandomRotation(0.2),
])

# for image, _ in train_dataset.take(1):                     # ploting/print the images/data
#     plt.figure(figsize=(10, 10))
#     first_image = image[0]
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
#         plt.imshow(augmented_image[0] / 255)
#         plt.axis('off')
        
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input     #setting up the base model that is mobilenet that accept valuse between -1 and 1
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)                   # rescaling the values

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False    #here we are freezing the Base

#adding our own classifier
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

#now we are going to compile the model
base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# now lets train our model
history = model.fit(
    train_dataset,
    epochs = 10,
    validation_data= validation_dataset
)

#saving the model so we dont have to train it again and again
model.save('cats_v_dogs.h5')
new_model = tf.keras.models.load_model('cats_v_dogs.h5')