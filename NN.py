import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnest = keras.datasets.fashion_mnist #load data
(train_images , train_labels ) , (test_images , test_labels) = fashion_mnest.load_data()  # this will split data into traing and testing

train_images.shape
train_images[0,23,23] #here 1st index represents the picture number,2nd the row number and thrid the column number hence we are pointing at a single pixel.
train_labels[:12]     #it is giving us the categories we can say we have and as we can see the values ranges from 0 to 9 meaning we have 10 categories.

class_names = ['Tshirt','Trousers','Pullover','Dress','Coat','Sandal','Shirt','sneakers','Bag','Ankle Boot']

# plt.figure()
# plt.imshow(train_images[59999])
# plt.colorbar()
# plt.grid(False)
# plt.show()

#presporcessing the data inwhich we are converting all the values between 0 and 1
train_images = train_images/255   #/ing by 255 as we know the pixel value is between 0 and 255
test_images = test_images/255     #samething here...

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),         #input layer
    keras.layers.Dense(128,activation = 'relu'),       #hidden layer
    keras.layers.Dense(10,activation = 'softmax')      #output layer
])

model.compile(                                        #here we are compileing the model by adding activation and loss functions
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics =['accuracy']
)

model.fit(train_images,train_labels,epochs = 10)       #here we are training the model by setting up the epochs and giving it the required training data 

test_loss , test_acc = model.evaluate(test_images,test_labels,verbose=1)    #here we are assigning the test values to test variables
print('test_accaracy: ',test_acc)

predictions = model.predict(test_images)                      #here ae are calculating the predictions of each image with each case
print(class_names[np.argmax(predictions[0])])
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()


#this is basicaaly a simple python program that will ask the user to endter a number and the program will take the inge on the index and then retuen the predicted case
def predict(model,image,correct_label):
    class_names = ['Tshirt','Trousers','Pullover','Dress','Coat','Sandal','Shirt','sneakers','Bag','Ankle Boot']
    predictions = model.predict(test_images)
    predicted_class = class_names[np.argmax(predictions[0])]
    show_image(image,class_names[correct_label],predicted_class)
    
def show_image(img,label,guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Expected" + label)
    plt.xlabel("Guess" + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0<= num <= 1000:
                return int(num)
        else:
            print("Try Again..")

num = get_number()
image= test_images[num]
label = test_labels[num]
predict(model,image,label)                