import tensorflow as tf
print("TensorFlow 版本:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
