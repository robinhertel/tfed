import tensorflow as tf

# tensorflow weighted custom loss function
def tf_wclf(y_true, y_hat):
    y_plus = tf.add(10., y_true)
    weights = tf.truediv(1., y_plus)
    loss = tf.where(tf.less(y_hat - y_true, 0), \
                    tf.math.exp(tf.math.negative(y_hat - y_true)\
                                /tf.constant([13.])) - 1,
                    tf.math.exp((y_hat - y_true)\
                                /tf.constant([10.])) - 1)
    w_loss = weights * loss
    return(tf.reduce_sum(w_loss))