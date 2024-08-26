# Notebook: Embeddings using a simple NN model
# Description: build an encoder model that codes the categorical features to embeddings on select categorical columns on XSOAR CrowdStrike Data
# we will train a simple NN model with a softmax layer for multi-class labels with: 
# optimizer=tf.keras.optimizers.Adam(),
# loss=tf.keras.losses.CategoricalCrossentropy(),

# What is word embedding?
# https://www.turing.com/kb/guide-on-word-embeddings-in-nlp#what-is-word-embedding?
 

# How to Implement?
# In this notebook, we will build an encoder model that codes the selected categorical features to embeddings. We train these embeddings in a simple NN model(via backpropagation). After the embedding encoder is trained, we will it as a preprocessor to the input features of our model.
# > source 1: https://towardsdatascience.com/4-ways-to-encode-categorical-features-with-high-cardinality-1bc6d8fd7b13
# > source 2: https://keras.io/examples/structured_data/classification_with_tfdf/#experiment-3-decision-forests-with-trained-embeddings
 

# import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils      
import math
import copy
from sklearn.model_selection import train_test_split
 

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)
 
CATEGORICAL_FEATURE_NAMES = [
    "stringColumn1",
    "stringColumn2"
]
 
# Build an NN model to train the embeddings
def build_input_layers(features):
    input_layers = {}
    for feature in features:
        input_layers[feature] = tf.keras.layers.Input(
            shape=(1,),
            name=feature,
            dtype=tf.string
        )
    return input_layers
 
def build_embeddings(x_train, size=None):
    input_layers = build_input_layers(CATEGORICAL_FEATURE_NAMES)
    embedded_layers = []
   
    for feature in input_layers.keys():
        # Get the vocabulary of the categorical feature
        vocabulary = sorted(
                [str(value) for value in list(x_train[feature].unique())]
            )
        # convert the string input values into integer indices
        cardinality = x_train[feature].nunique()
        pre_processing_layer = tf.keras.layers.StringLookup(
            vocabulary=vocabulary,
            num_oov_indices=cardinality,
            name=feature+"_preprocessed"
        )
        pre_processed_input = pre_processing_layer(input_layers[feature])
        # Create an embedding layer with the specified dimensions
        embedding_size = int(math.sqrt(cardinality))
        print("cardinality",cardinality)
        print("embedding_size",embedding_size)
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=2*cardinality+1,
            output_dim=embedding_size,
            name=feature+"_embedded",
 
        )
        embedded_layers.append(embedding_layer(pre_processed_input))  
 
   
    # Concatenate all the encoded features.
    encoded_features = tf.keras.layers.Concatenate()([
                tf.keras.layers.Flatten()(layer) for layer in embedded_layers
            ])
   
    # Apply dropout.
    encoded_features = tf.keras.layers.Dropout(rate=0.25)(encoded_features)
 
    # Perform non-linearity projection.
    encoded_features = tf.keras.layers.Dense(
        units=size if size else encoded_features.shape[-1], activation="gelu"
    )(encoded_features)
    return tf.keras.Model(inputs=input_layers, outputs=encoded_features)
 
def build_neural_network_model(embedding_encoder):
    input_layers = build_input_layers(CATEGORICAL_FEATURE_NAMES)
    embeddings = embedding_encoder(input_layers)
    output = tf.keras.layers.Dense(units=3, activation="softmax")(embeddings)
 
    model = keras.Model(inputs=input_layers,
                        outputs=output)
   
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        #loss=weighted_categorical_crossentropy(weights),
        metrics=[tf.keras.metrics.AUC()]
    )
   
    return model
 

# Build the neural network model
embedding_encoder = build_embeddings(x_train, size=64)
neural_network_model = build_neural_network_model(embedding_encoder)
 
# Prepare the datasets: train, test, validation
def build_dataset(x, y, features):
    x_output = {}
    y_output = utils.to_categorical(y, 3)
    for feat in features:
        x_output[feat] = np.array(x[feat]).reshape(-1,1).astype(str)   
    return x_output, y_output
   
x_train, y_train = build_dataset(x_train, y_train, CATEGORICAL_FEATURE_NAMES)
x_test, y_test = build_dataset(x_test, y_test, CATEGORICAL_FEATURE_NAMES)
x_val, y_val = build_dataset(x_val, y_val, CATEGORICAL_FEATURE_NAMES)



# Train and evaluate the model
history = neural_network_model.fit(x_train, y_train, batch_size=64, epochs=31, validation_data=(x_val, y_val))
history.history
 

# Test the Model
results = neural_network_model.evaluate(x_test, y_test)
print("test loss, test acc:", results)
predictions = neural_network_model.predict(x_test)
print(predictions)
 
# visualized the classification
predict_class = np.argmax(predictions, axis=1)
print(predict_class)
true_class = np.argmax(y_test, axis=1)
print(true_class)
unique, counts = np.unique(predict_class, return_counts=True)
print(unique, counts)
unique, counts = np.unique(true_class, return_counts=True)
print(unique, counts)
print(predict_class - true_class)
 
# encode the feature
X_train_encoded = embedding_encoder.predict(x_train, batch_size=128)
X_test_encoded = embedding_encoder.predict(x_test, batch_size=128)
print(X_train_encoded)