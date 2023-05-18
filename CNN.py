import tensorflow as tf
from tensorflow.keras import datasets, layers, models 
import matplotlib.pyplot as plt

(train_images , train_labels),(test_images , test_labels) = datasets.cifar10.load_data()  #loadng and spliting the data

train_images, test_images = train_images/255.0 ,test_images/255.0   #makising sure the values are between 0 and 1

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

img_index = 5
plt.imshow(train_images[img_index])
plt.xlabel(class_names[train_labels[img_index][0]])
plt.show()

# making the model of CNN which wil basically extract all the features of differet objects
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3) , activation = 'relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))

#adding the dense layers
# these layers will see the pictures and based upon the data and features the CNN have provided, will determine what is the picture of
# this doce is simallar to dense NN
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10))

#now we are going to train our model
model.compile(
    optimizer='adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(train_images,train_labels,epochs=10,validation_data=(test_images,test_labels))

test_loss , test_acc = model.evaluate(test_images,test_labels,verbose=2)
print(test_loss,test_acc)