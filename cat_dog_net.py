import tensorflow as tf
import dataset


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.zeros(shape))


def conv_layer(x, w_shape, stride_shape, b_shape, name):
    with tf.name_scope(name):
        W = weight_variable(w_shape)
        b = bias_variable(b_shape)
        conv = tf.nn.conv2d(x, W, stride_shape, padding='SAME')
        h = conv + b
        relu = tf.nn.relu(h)
        return relu


def conv_pool(x, w_shape, stride_shape, b_shape, name):
    with tf.name_scope(name):
        W = weight_variable(w_shape)
        b = bias_variable(b_shape)
        conv = tf.nn.conv2d(x, W, stride_shape, padding='SAME')
        h = conv + b
        relu = tf.nn.relu(h)
        pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pool


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        weights = weight_variable([input_dim, output_dim])
        biases = bias_variable([output_dim])
        pre_activate = tf.matmul(input_tensor, weights) + biases
        if act is not None:
            activations = act(pre_activate, name='activation')
            return activations
        else:
            return pre_activate


def accuracy(y_estimate, y_real):
    correct_prediction = tf.equal(tf.argmax(y_estimate, 1), tf.argmax(y_real, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


x = tf.placeholder(tf.float32, [None, 208, 208, 3], name='input')
y_ = tf.placeholder(tf.float32, [None, 2], name='label')
h1 = conv_pool(x, [11, 11, 3, 48], [1, 4, 4, 1], [48], 'conv_pool_1')
h2 = conv_pool(h1, [5, 5, 48, 128], [1, 1, 1, 1], [128], 'conv_pool_2')
h3 = conv_layer(h2, [3, 3, 128, 192], [1, 1, 1, 1], [192], 'conv_3')
h4 = conv_layer(h3, [3, 3, 192, 192], [1, 1, 1, 1], [192], 'conv_4')
h5 = conv_pool(h4, [3, 3, 192, 128], [1, 1, 1, 1], [128], 'conv_pool_5')
h5r = tf.reshape(h5, [-1, 7 * 7 * 128])
h6 = nn_layer(h5r, 7 * 7 * 128, 2048, "nn_6")
h7 = nn_layer(h6, 2048, 2048, "nn_7")
result = nn_layer(h7, 2048, 2, "result")

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(result,1e-10,1.0)))
loss = tf.reduce_mean(cross_entropy)
accuracy = accuracy(result, y_)
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

saver = tf.train.Saver()

batch_size = 128

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# ckpt = tf.train.get_checkpoint_state('checkpoint')
# if ckpt and ckpt.model_checkpoint_path:
#     print(ckpt.model_checkpoint_path)
#     saver.restore(sess, ckpt.model_checkpoint_path)
#     print( 'data restored')

data_url_list, label_list = dataset.get_data_url_set()
index = 0
for i in range(1, 10000):
    print(index)
    batch_data, batch_label, index = dataset.get_next_batch_data(data_url_list, label_list, batch_size, index)
    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_data, y_: batch_label})
        print("step %d,training accuracy = %g" % (i, train_accuracy))
    _, loss_val = sess.run([train_step, loss], feed_dict={x: batch_data, y_: batch_label})
    print("step %d,loss = %g" % (i, loss_val))
    if i % 500 == 0 and i != 0:
        saver_path = saver.save(sess, 'checkpoint/%05d.ckpt' % i)
        print("Model saved in file:", saver_path)
