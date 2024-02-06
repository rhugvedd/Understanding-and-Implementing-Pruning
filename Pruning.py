import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot

import tempfile

# Define the initial setup model

input_shape = [20]
x_train = np.random.randn(1, 20).astype(np.float32)
y_train = tf.keras.utils.to_categorical(np.random.randn(1), num_classes=20)

def setup_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(20, input_shape=input_shape),
      tf.keras.layers.Flatten()
  ])
  return model

def setup_pretrained_weights():
  model = setup_model()

  model.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer='adam',
      metrics=['accuracy']
  )

  model.fit(x_train, y_train)

  _, pretrained_weights = tempfile.mkstemp('.tf')

  model.save_weights(pretrained_weights)

  return pretrained_weights

def get_gzipped_model_size(model):
  import os
  import zipfile

  _, keras_file = tempfile.mkstemp('.h5')
  model.save(keras_file, include_optimizer=False)

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(keras_file)

  return os.path.getsize(zipped_file)

setup_model()
pretrained_weights = setup_pretrained_weights()

#########################################################################################################################

# Prune whole model (Sequential and Functional)

base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended.

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

model_for_pruning.summary()

#########################################################################################################################

# Prune some layers (Sequential and Functional)

base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended for model accuracy

def apply_pruning_to_dense(layer):
  if isinstance(layer, tf.keras.layers.Dense):
    return tfmot.sparsity.keras.prune_low_magnitude(layer)
  return layer

model_for_pruning = tf.keras.models.clone_model(
    base_model,
    clone_function=apply_pruning_to_dense,
)

model_for_pruning.summary()

#########################################################################################################################

# While this example used the type of the layer to decide what to prune, the easiest way to prune a particular layer is to set its `name` property, and look for that name in the `clone_function`.

print('=================================================================================================================')
print(base_model.layers[0].name)
print('=================================================================================================================')

# #### More readable but potentially lower model accuracy

# This is not compatible with fine-tuning with pruning, which is why it may be less accurate than the above examples which
# support fine-tuning.
# 

#########################################################################################################################
# In the below examples, while `prune_low_magnitude` can be applied during the initial model definition,-
# -loading the weights afterward does not work.

# Functional API

i = tf.keras.Input(shape=(20,))
x = tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(10))(i)
o = tf.keras.layers.Flatten()(x)
model_for_pruning = tf.keras.Model(inputs=i, outputs=o)

model_for_pruning.summary()


# Sequential example API

model_for_pruning = tf.keras.Sequential([
  tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(20, input_shape=input_shape)),
  tf.keras.layers.Flatten()
])

model_for_pruning.summary()

#########################################################################################################################

# Prune custom Keras layer or modify parts of layer to prune

class MyDenseLayer(tf.keras.layers.Dense, tfmot.sparsity.keras.PrunableLayer):

  def get_prunable_weights(self):
    return [self.kernel, self.bias]

model_for_pruning = tf.keras.Sequential([
  tfmot.sparsity.keras.prune_low_magnitude(MyDenseLayer(20, input_shape=input_shape)),
  tf.keras.layers.Flatten()
])

model_for_pruning.summary()

#########################################################################################################################
# Train model

base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended for model accuracy
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

log_dir = tempfile.mkdtemp()
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
]

model_for_pruning.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer='adam',
      metrics=['accuracy']
)

model_for_pruning.fit(
    x_train,
    y_train,
    callbacks=callbacks,
    epochs=2,
)

#docs_infra: no_execute
# get_ipython().run_line_magic('tensorboard', '--logdir={log_dir}')

#########################################################################################################################

# Custom training loop

# Define the model.
base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended for model accuracy
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

# Boilerplate
loss = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam()
log_dir = tempfile.mkdtemp()
unused_arg = -1
epochs = 2
batches = 1

# Non-boilerplate.
model_for_pruning.optimizer = optimizer
step_callback = tfmot.sparsity.keras.UpdatePruningStep()
step_callback.set_model(model_for_pruning)
log_callback = tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir) # Log sparsity and other metrics in Tensorboard.
log_callback.set_model(model_for_pruning)

step_callback.on_train_begin() # run pruning callback
for _ in range(epochs):
  log_callback.on_epoch_begin(epoch=unused_arg) # run pruning callback
  for _ in range(batches):
    step_callback.on_train_batch_begin(batch=unused_arg) # run pruning callback

    with tf.GradientTape() as tape:
      logits = model_for_pruning(x_train, training=True)
      loss_value = loss(y_train, logits)
      grads = tape.gradient(loss_value, model_for_pruning.trainable_variables)
      optimizer.apply_gradients(zip(grads, model_for_pruning.trainable_variables))

  step_callback.on_epoch_end(batch=unused_arg) # run pruning callback

#docs_infra: no_execute
# get_ipython().run_line_magic('tensorboard', '--logdir={log_dir}')

#########################################################################################################################

# Improve pruned model accuracy

# Define the model.
base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended for model accuracy
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

_, keras_model_file = tempfile.mkstemp('.h5')

# Checkpoint: save the optimizer 
model_for_pruning.save(keras_model_file, include_optimizer=True)

# The code below is only needed for the HDF5 model format (not HDF5 weights and other formats).

# Deserialize model.
with tfmot.sparsity.keras.prune_scope():
  loaded_model = tf.keras.models.load_model(keras_model_file)

loaded_model.summary()

#########################################################################################################################
# Deploy pruned model

# Export model with size compression

# Both `strip_pruning` and applying a standard compression algorithm (e.g. via gzip) are necessary to see the compression
# benefits of pruning.

# Define the model.
base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended for model accuracy
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

print("final model")
model_for_export.summary()

print("\n")
print("Size of gzipped pruned model without stripping: %.2f bytes" % (get_gzipped_model_size(model_for_pruning)))
print("Size of gzipped pruned model with stripping: %.2f bytes" % (get_gzipped_model_size(model_for_export)))
