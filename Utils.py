import tensorflow as tf

5,4,3,2,1

def normalize(data):
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)

    train_data = (data - min_val) / (max_val - min_val)
    return tf.cast(train_data, tf.float32)