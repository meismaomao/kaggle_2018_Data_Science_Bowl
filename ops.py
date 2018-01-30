import tensorflow as tf

def lrelu(x, leak=0.2):
	return tf.maximum(x, leak * x)

def conv2d(input, output_channles, is_train, k_h=5, k_w=5, stddev=0.02, name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_channles],
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		biases = tf.get_variable("biases", [output_channles], initializer=tf.constant_initializer(0.0))
		conv = tf.nn.conv2d(input, w, strides=[1, 2, 2, 1], padding='SAME')
		conv = lrelu(tf.nn.bias_add(conv, biases))
		bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True,
										  decay=0.9, is_training=is_train, updates_collections=None)
		return bn

def deconv2d(input, output_channles, is_train, k, s, name="deconv2d", stddev=0.02, activation_fn=None):
	with tf.variable_scope(name):
		deconv = tf.contrib.layers.conv2d_transpose(input, num_outputs=output_channles,
										 weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
										 biases_initializer=tf.zeros_initializer(),
										 kernel_size=[k, k], stride=[s, s], padding='VALID')
		if not activation_fn:
			deconv = tf.nn.relu(deconv)
			deconv = tf.contrib.layers.batch_norm(deconv, center=True, scale=True,
												  decay=0.9, is_training=is_train, updates_collections=None)
		else:
			deconv = activation_fn(deconv)
		return deconv

def deconv2d_1(input, height, width, output_channels, input_channels, is_train,
			   k, s, batch_size, name='deconv2d', stddev=0.02, activation_fn=None):
	with tf.variable_scope(name):
		weight = tf.Variable(tf.truncated_normal([k, k, output_channels, input_channels], stddev=stddev), name='weight')
		bias = tf.Variable(tf.constant(0.0, shape=[output_channels]), name='bias')
		deconv = tf.nn.conv2d_transpose(input, filter=weight,
										output_shape=tf.stack([batch_size, height, width, output_channels]), strides=[1, s, s, 1],
										padding='SAME')
		deconv = tf.nn.bias_add(deconv, bias)

		if not activation_fn:
			# deconv = tf.nn.relu(deconv)
			deconv = tf.contrib.layers.batch_norm(deconv, center=True, scale=True,
												  decay=0.9, is_training=is_train, updates_collections=None)
		else:
			deconv = activation_fn(deconv)

		return deconv
