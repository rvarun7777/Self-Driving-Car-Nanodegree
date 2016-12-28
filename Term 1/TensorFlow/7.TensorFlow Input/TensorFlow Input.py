Quiz

Let's see how well you understand tf.placeholder() and feed_dict. 
The code below throws an error, but I want you to make it return the number 123. 
Change line 14, so that the code returns the number 123.import tensorflow as tf


def run():
    output = None
    x = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        # TODO: Feed the x tensor 123
        output = sess.run(x)

    return output
