import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from official.nlp import optimization

def build_classifier_model(tfhub_handle_preprocess,tfhub_handle_encoder):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['sequence_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Conv1D(32, 16, activation='tanh',)(net)
    net = tf.keras.layers.AveragePooling1D()(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(128,activation='relu',)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
    return tf.keras.Model(text_input, net)

