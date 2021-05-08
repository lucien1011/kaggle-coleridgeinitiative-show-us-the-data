import os
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
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
    return tf.keras.Model(text_input, net)

# __________________________________________________________________ ||
config = ObjDict(

    name = "optimise_classifier_210502_smallbert_en_uncased_L6_H128_A2",

    input_df = "data/train_lite.csv",

    tfhub_handle_preprocess='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2',
    
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics = tf.keras.metrics.AUC(),
    optimizer = tf.keras.optimizers.Adam(),

    epochs = 5,

    saved_model_path = 'saved_model/optimise_classifier_210502_smallbert_en_uncased_L6_H128_A2',
    saved_history_path = 'saved_model/optimise_classifier_210502_smallbert_en_uncased_L6_H128_A2/history.p',
)

# __________________________________________________________________ ||
config.input_np_dir = "data/optimise_classifier_210501_01/" 
config.model = build_classifier_model(config.tfhub_handle_preprocess,config.tfhub_handle_encoder)

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
