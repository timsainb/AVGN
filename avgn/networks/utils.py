"""
from tqdm.autonotebook import tqdm
import tensorflow as tf


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

#@tf.function
def train_network(model, train_dataset, optimizer):
    for train_x in tqdm(train_dataset): 
        with tf.GradientTape() as tape:
            loss = model.compute_loss(train_x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  
        
#@tf.function
def test_network(model, test_dataset):
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(model.compute_loss(test_x))
    elbo = -loss.result()
    return elbo




%load_ext tensorboard.notebook
import os
logs_base_dir = "./logs"
os.makedirs(logs_base_dir, exist_ok=True)
# Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
%tensorboard --logdir {logs_base_dir} --port 8196
"""