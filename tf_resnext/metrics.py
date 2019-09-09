import tensorflow as tf


def sparse_cross_entropy_loss(y_true, y_pred):
    y_true = tf.cast(tf.squeeze(y_true, axis=1), tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(loss)


def accuracy(y_true, y_pred):
    y_true = tf.cast(tf.squeeze(y_true, axis=1), tf.int32)
    y_pred = tf.argmax(y_pred, axis=1, output_type=tf.int32)
    return tf.reduce_mean(tf.to_float(tf.equal(y_true, y_pred)))
