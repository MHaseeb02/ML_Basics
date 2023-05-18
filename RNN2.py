from keras.preprocessing import sequence
import tensorflow as tf
import keras
import os
import numpy as np

#loading the dataset
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

#reading the contents of the file
text = open(path_to_file,"rb").read().decode(encoding='utf-8')
print('Length of text {} characters'.format(len(text))) 

# print(text[:250])

# encoding the text
vocab = sorted(set(text))
char2idx = { u:i for i,u in enumerate(vocab)} #creating a mapping function from char to indices
idx2char = np.array(vocab)

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)
print("Text" , text[:13])
print("Encoded",text_to_int(text[:13]))

def int_to_text(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return "".join(idx2char[ints])

print(int_to_text(text_as_int[:13]))

#Creating training samples
seq_length = 100                          #length of sequence for training data
example_per_epoc = len(text) // (seq_length + 1)
char_data = tf.data.Dataset.from_tensor_slices(text_as_int) #creating training examples
sequences = char_data.batch(seq_length+1,drop_remainder=True)  #this will basically convert the whole dataset into batches of 101 batch each

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

#Now lets create the training batches
batch_size = 64
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
buffer_size = 1000

dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder = True)

#buidling the model
def build_model(vocab_size,embedding_dim,rnn_units,batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,batch_input_shape=[batch_size,None]),
        tf.keras.layers.LSTM(rnn_units,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size,embedding_dim,rnn_units,batch_size)

for input_example_batch , target_example_batch in dataset.take(1):
    example_batch_prdiction = model(input_example_batch)
    
pred = example_batch_prdiction[0]
# print(len(pred))
# print(pred)    

time_pred = pred[0]
# print(len(time_pred))
# print(time_pred)

sampled_indices = tf.random.categorical(pred, num_samples=1)
sampled_indices = np.reshape(sampled_indices ,(1,-1))[0]
predicted_char = int_to_text(sampled_indices)


#Training the model
def loss(labels,logits):                          #making the loss function
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits= True)

model.compile(optimizer="adam",loss = loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

history = model.fit(dataset, epochs=40,callbacks=[checkpoint_callback])

model = build_model(vocab_size,embedding_dim,rnn_units,batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1,None]))

def generate_text(model,start_string):
    num_generatr = 100

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1

    model.reset_states()
    for i in range(num_generatr):
        prediction = model(input_eval)
        prediction = tf.squeeze(prediction, 0)
        prediction = prediction / temperature
        predicted_id = tf.random.categorical(prediction, num_samples = 1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return(start_string + "".join(text_generated))

inp = input("Type a Starting String: ")
print(generate_text(model,inp))