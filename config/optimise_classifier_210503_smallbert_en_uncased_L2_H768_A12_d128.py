import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

from utils.objdict import ObjDict

# __________________________________________________________________ ||
name = "optimise_classifier_210503_smallbert_en_uncased_L2_H768_A12_d128"
input_df = "data/train_lite.csv"

# __________________________________________________________________ ||
def build_classifier_model(tfhub_handle_preprocess,tfhub_handle_encoder):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=False, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(128, activation='relu',)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
    return tf.keras.Model(text_input, net)

# __________________________________________________________________ ||
config = ObjDict(

    name = name,

    input_df = input_df,

    x_train_path = os.path.join(input_df,"x_train.npy"),
    y_train_path = os.path.join(input_df,"y_train.npy"),
    sample_weight_train_path = os.path.join(input_df,"sample_weight_train.npy"),

    x_val_path = os.path.join(input_df,"x_val.npy"),
    y_val_path = os.path.join(input_df,"y_val.npy"),
    sample_weight_val_path = os.path.join(input_df,"sample_weight_val.npy"),

    x_test_path = os.path.join(input_df,"x_test.npy"),
    y_test_path = os.path.join(input_df,"y_test.npy"),
    sample_weight_test_path = os.path.join(input_df,"sample_weight_test.npy"),

    tfhub_handle_preprocess='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/2',
    
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics = tf.keras.metrics.AUC(),
    optimizer = tf.keras.optimizers.Adam(),
    batch_size = 32,
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=3, restore_best_weights=True, mode='max'),
        ],

    epochs = 100,

    saved_model_path = 'saved_model/'+name,
    saved_history_path = 'saved_model/'+name+'/history.p',
)

# __________________________________________________________________ ||
config.input_np_dir = "data/optimise_classifier_210501_01/" 
config.model = build_classifier_model(config.tfhub_handle_preprocess,config.tfhub_handle_encoder)
config.checkpoint = None

# __________________________________________________________________ ||
config.slurm_cfg_name = 'submit.cfg'
config.slurm_job_dir = os.path.join('job/',config.name+'/')
config.slurm_commands = """echo \"{job_name}\"
cd {base_path}
source setup_hpg.sh
python3 {pyscript} {cfg_path}
""".format(
            job_name=config.name,
            pyscript="optimise_classifier.py",
            cfg_path="config/"+config.name+".py",
            base_path=os.environ['BASE_PATH'],
            output_path=config.slurm_job_dir,
            )

