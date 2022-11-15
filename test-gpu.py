import tensorflow as tf

tf.config.list_physical_devices("GPU")
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
