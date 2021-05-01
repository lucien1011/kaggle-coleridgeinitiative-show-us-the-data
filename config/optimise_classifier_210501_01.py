import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from official.nlp import optimization

from utils.objdict import ObjDict

# __________________________________________________________________ ||
def build_classifier_model(tfhub_handle_preprocess,tfhub_handle_encoder):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=False, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['sequence_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Conv1D(32, 16, activation='tanh',)(net)
    net = tf.keras.layers.AveragePooling1D()(net)
    net = tf.keras.layers.Conv1D(16, 16, activation='tanh',)(net)
    net = tf.keras.layers.AveragePooling1D()(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(32,activation='tanh',)(net)
    net = tf.keras.layers.Dense(32,activation='tanh',)(net)
    net = tf.keras.layers.Dense(32,activation='tanh',)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
    return tf.keras.Model(text_input, net)

# __________________________________________________________________ ||
config = ObjDict(

    name = "optimise_classifier_210501_01",

    input_df = "data/train_lite.csv",

    tfhub_handle_preprocess='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2',
    
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics = tf.keras.metrics.AUC(),
    optimizer = tf.keras.optimizers.Adam(),

    epochs = 5,

    saved_model_path = 'saved_model/optimise_classifier_210501_01',
)

# __________________________________________________________________ ||
config.input_np_dir = "data/"+config.name+"/"
config.model = build_classifier_model(config.tfhub_handle_preprocess,config.tfhub_handle_encoder)
