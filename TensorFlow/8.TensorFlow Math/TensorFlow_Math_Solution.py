import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
z = tf.sub(tf.div(x,y),tf.constant(1))

# TODO: Print z from a session
with tf.Session() as sess:
    output = sess.run(z)
    print(output)
