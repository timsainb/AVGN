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