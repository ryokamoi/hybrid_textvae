import sys

import numpy as np
import tensorflow as tf

sys.path.append("../")

from config import FLAGS

class Encoder(object):
    def __init__(self, encoder_input_embedded, is_training=True, ru=False):
        with tf.name_scope("encoder_input"):
            self.encoder_input_embedded = encoder_input_embedded
            self.is_training = is_training

        with tf.variable_scope("encoder_cnn"):
            with tf.variable_scope("encoder_cnn1"):
                filter1 = tf.get_variable(name="encoder_cnn1_filter1",
                                          shape=(3, FLAGS.EMBED_SIZE, 128),
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(stddev=0.1))

                with tf.name_scope("h1"):
                    h1 = tf.nn.conv1d(self.encoder_input_embedded,
                                      filter1,
                                      stride=2,
                                      padding='SAME',
                                      name="encoder_conv1")

                    normed1 = tf.nn.relu(tf.contrib.layers.batch_norm(
                                                    h1,
                                                    decay=0.99,
                                                    center=True,
                                                    scale=True,
                                                    updates_collections=None,
                                                    is_training=self.is_training,
                                                    reuse=ru,
                                                    scope="bn1"))

                    assert normed1.shape == (FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN/2, 128)

            with tf.variable_scope("encoder_cnn2"):
                filter2 = tf.get_variable(name="encoder_cnn1_filter2",
                                          shape=(3, 128, FLAGS.ENCODER_CNN_OUTPUT_NUM),
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(stddev=0.1))

                with tf.name_scope("h2"):
                    h2 = tf.nn.conv1d(normed1,
                                      filter2,
                                      stride=2,
                                      padding='SAME',
                                      name="encoder_conv2")

                    normed2 = tf.nn.relu(tf.contrib.layers.batch_norm(
                                                    h2,
                                                    decay=0.99,
                                                    center=True,
                                                    scale=True,
                                                    updates_collections=None,
                                                    is_training=self.is_training,
                                                    reuse=ru,
                                                    scope="bn2"))

                    assert normed2.shape == (FLAGS.BATCH_SIZE, int(FLAGS.SEQ_LEN/4), FLAGS.ENCODER_CNN_OUTPUT_NUM)

            with tf.name_scope("encoder_rnn_output"):
                encoder_cnn_output = tf.reshape(normed2,
                                                [FLAGS.BATCH_SIZE,
                                                 FLAGS.ENCODER_CNN_OUTPUT_NUM * int(FLAGS.SEQ_LEN/4)])

        with tf.variable_scope("encoder_linear"):
            context_to_mu_W = tf.get_variable(name="context_to_mu_W",
                                              shape=[FLAGS.ENCODER_CNN_OUTPUT_NUM * int(FLAGS.SEQ_LEN/4),
                                                     FLAGS.LATENT_VARIABLE_SIZE],
                                              dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(stddev=0.1))

            context_to_mu_b = tf.get_variable(name="context_to_mu_b",
                                              shape=[FLAGS.LATENT_VARIABLE_SIZE],
                                              dtype=tf.float32)

            context_to_logvar_W = tf.get_variable(
                                              name="context_to_logvar_W",
                                              shape=[FLAGS.ENCODER_CNN_OUTPUT_NUM * int(FLAGS.SEQ_LEN/4),
                                                     FLAGS.LATENT_VARIABLE_SIZE],
                                              dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(stddev=0.1))

            context_to_logvar_b = tf.get_variable(
                                              name="context_to_logvar_b",
                                              shape=[FLAGS.LATENT_VARIABLE_SIZE],
                                              dtype=tf.float32)

        with tf.name_scope("mu"):
            self.mu = tf.matmul(encoder_cnn_output, context_to_mu_W) + context_to_mu_b
        with tf.name_scope("log_var"):
            self.logvar = tf.matmul(encoder_cnn_output, context_to_logvar_W) + context_to_logvar_b

        with tf.name_scope("z"):
            z = tf.truncated_normal((FLAGS.BATCH_SIZE, FLAGS.LATENT_VARIABLE_SIZE), stddev=1.0)

        with tf.name_scope("latent_variables"):
            self.latent_variables = self.mu + tf.exp(0.5 * self.logvar) * z
