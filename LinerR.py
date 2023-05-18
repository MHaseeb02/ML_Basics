from __future__ import absolute_import, division , print_function, unicode_literals
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from IPython.display import clear_output
import six
# import tensorflow.compat.v2 as fc
import tensorflow as tf 

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') #testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived') 

catagorical_columns = ['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
numerical_columns = ['age','fare']

feature_columns = []
for feature_name in catagorical_columns:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary))

for feature_name in numerical_columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype = tf.float32))
    
# print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

# print(result['accuracy'])
# print(result)

result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[3])
print(y_eval.loc[3])
print(result[3]['probabilities'][1])
