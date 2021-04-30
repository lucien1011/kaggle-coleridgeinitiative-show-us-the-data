import os
import json
import string
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from official.nlp import optimization
from sklearn.model_selection import train_test_split

from model import build_classifier_model

tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2'

df = pd.read_csv("data/train_lite.csv",index_col=0)

x = df['sentence'].to_numpy()
y = df['hasDataset'].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

print("train: ",x_train.shape,y_train.shape)
print("val: ",x_val.shape,y_val.shape)
print("train: ",x_test.shape,y_test.shape)

model = build_classifier_model(tfhub_handle_preprocess,tfhub_handle_encoder)

loss = tf.keras.losses.BinaryCrossentropy()
metrics = tf.keras.metrics.AUC()

epochs = 1
num_train_steps = 1 * epochs
num_warmup_steps = 1

init_lr = 3e-4
optimizer = optimization.create_optimizer(
    init_lr=init_lr,
    num_train_steps=num_train_steps,
    num_warmup_steps=1,
    optimizer_type='adamw',
)
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)

history = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val,y_val),
    epochs=epochs,
    batch_size=512,
    )
