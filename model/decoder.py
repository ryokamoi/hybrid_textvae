import sys

import tensorflow as tf

sys.path.append("../")

from config import FLAGS

class Decoder(object):
    def __init__(self, decoder_input_list, latent_variables, embedding,
                 batchloader, is_training=True, ru=False):
        with tf.name_scope("decoder_input"):
            self.decoder_input_list = decoder_input_list
            self.latent_variables = latent_variables
            self.embedding = embedding

            self.batchloader = batchloader
            self.go_input = tf.constant(self.batchloader.go_input,
                                        dtype=tf.int32)

            self.is_training = is_training

        with tf.variable_scope("lv2decoder"):
            lv2decoder_W = tf.get_variable(name="lv2decoder_W",
                                           shape=(FLAGS.LATENT_VARIABLE_SIZE,
                                                  FLAGS.ENCODER_CNN_OUTPUT_NUM * \
                                                  int(int(FLAGS.SEQ_LEN/4))))
            lv2decoder_b = tf.get_variable(name="lv2decoder_b",
                                           shape=(FLAGS.ENCODER_CNN_OUTPUT_NUM * \
                                                  int(int(FLAGS.SEQ_LEN/4))))

        with tf.name_scope("decoder_cnn_input"):
            decoder_cnn_input = tf.nn.relu(tf.matmul(self.latent_variables, lv2decoder_W) \
                                                + lv2decoder_b)
            decoder_cnn_input = tf.reshape(decoder_cnn_input,
                                           [FLAGS.BATCH_SIZE,
                                            1,
                                            int(FLAGS.SEQ_LEN/4),
                                            FLAGS.ENCODER_CNN_OUTPUT_NUM])

        with tf.variable_scope("decoder_cnn"):
            with tf.variable_scope("decoder_cnn1"):
                filter1 = tf.get_variable(name="decoder_cnn1_filter1",
                                          shape=(1, 3, 128, FLAGS.ENCODER_CNN_OUTPUT_NUM),
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(stddev=0.1))

                with tf.name_scope("h1"):
                    h1 = tf.nn.conv2d_transpose(decoder_cnn_input,
                                                filter1,
                                                output_shape=[FLAGS.BATCH_SIZE,
                                                              1,
                                                              int(FLAGS.SEQ_LEN/2),
                                                              128],
                                                strides=[1, 1, 2, 1],
                                                padding='SAME',
                                                name="decoder_conv1")

                    normed1 = tf.nn.relu(tf.contrib.layers.batch_norm(
                                                h1,
                                                decay=0.99,
                                                center=True,
                                                scale=True,
                                                updates_collections=None,
                                                is_training = self.is_training,
                                                reuse=ru,
                                                scope="decoder_bn1"))

            with tf.variable_scope("decoder_cnn2"):
                filter2 = tf.get_variable(name="decoder_cnn1_filter2",
                                          shape=(1, 3, FLAGS.DECODER_CNN_OUTPUT_NUM, 128),
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(stddev=0.1))

                with tf.name_scope("h2"):
                    h2 = tf.nn.conv2d_transpose(normed1,
                                                filter2,
                                                output_shape=[FLAGS.BATCH_SIZE,
                                                              1,
                                                              FLAGS.SEQ_LEN,
                                                              FLAGS.DECODER_CNN_OUTPUT_NUM],
                                                strides=[1, 1, 2, 1],
                                                padding='SAME',
                                                name="decoder_conv2")

                    normed2 = tf.nn.relu(tf.contrib.layers.batch_norm(
                                                h2,
                                                decay=0.99,
                                                center=True,
                                                scale=True,
                                                updates_collections=None,
                                                is_training=self.is_training,
                                                reuse=ru,
                                                scope="decoder_bn2"))

            with tf.name_scope("decoder_cnn_output"):
                decoder_cnn_output = tf.reshape(normed2,
                                                [FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN, FLAGS.DECODER_CNN_OUTPUT_NUM])
                decoder_cnn_output_t = tf.transpose(decoder_cnn_output, perm=[1, 0, 2])

                self.decoder_cnn_output_list = []
                for i in range(FLAGS.SEQ_LEN):
                    self.decoder_cnn_output_list.append(decoder_cnn_output_t[i])
                    assert self.decoder_cnn_output_list[i].shape == (FLAGS.BATCH_SIZE, FLAGS.DECODER_CNN_OUTPUT_NUM)

            with tf.variable_scope("decoder_cnn2vocab"):
                cnn2vocab_W = tf.get_variable(name="cnn2vocab_W",
                                              shape=(FLAGS.DECODER_CNN_OUTPUT_NUM, FLAGS.VOCAB_SIZE),
                                              dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(stddev=0.1))

                cnn2vocab_b = tf.get_variable(name="decoder_linear_b",
                                              shape=(FLAGS.VOCAB_SIZE),
                                              dtype=tf.float32)

            with tf.name_scope("aux_logits"):
                self.aux_logits = []
                for cnn_output in self.decoder_cnn_output_list:
                    aux_logit = tf.matmul(cnn_output, cnn2vocab_W) + cnn2vocab_b
                    assert aux_logit.shape == (FLAGS.BATCH_SIZE, FLAGS.VOCAB_SIZE)

                    self.aux_logits.append(aux_logit)

        # rnn
        with tf.variable_scope("decoder_rnn"):
            with tf.variable_scope("rnn_input_weight"):
                self.rnn_input_W = tf.get_variable(name="rnn_input_W",
                                                   shape=(FLAGS.EMBED_SIZE + FLAGS.DECODER_CNN_OUTPUT_NUM,
                                                          FLAGS.RNN_SIZE),
                                                   dtype=tf.float32,
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                self.rnn_input_b = tf.get_variable(name="rnn_input_b",
                                                   shape=(FLAGS.RNN_SIZE),
                                                   dtype=tf.float32)

            with tf.variable_scope("decoder_rnn"):
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(FLAGS.RNN_SIZE)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     output_keep_prob=FLAGS.DROPOUT_KEEP)
                self.cell = tf.contrib.rnn.MultiRNNCell([cell] * FLAGS.RNN_NUM)

                self.init_states = [cell.zero_state(FLAGS.BATCH_SIZE, tf.float32)
                                    for _ in range(FLAGS.RNN_NUM)]
                self.states = [tf.placeholder(tf.float32,
                                              (FLAGS.BATCH_SIZE),
                                              name="state")
                               for _ in range(FLAGS.RNN_NUM)]

            with tf.variable_scope("decoder_rnn2vocab"):
                self.rnn2vocab_W = tf.get_variable(name="rnn2vocab_W",
                                                   shape=(FLAGS.RNN_SIZE, FLAGS.VOCAB_SIZE),
                                                   dtype=tf.float32,
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                self.rnn2vocab_b = tf.get_variable(name="rnn2vocab_b",
                                                   shape=(FLAGS.VOCAB_SIZE),
                                                   dtype=tf.float32)

            with tf.name_scope("decoder_rnn_output"):
                if self.is_training:
                    self.rnn_logits = self.rnn_train_predict()
                else:
                    self.rnn_logits = self.rnn_valid_predict()


    # input text from dataset
    def rnn_train_predict(self):
        pred = []
        state = self.init_states
        for i in range(FLAGS.SEQ_LEN):
            with tf.name_scope("decoder_input_embedding"):
                cnn_output = self.decoder_cnn_output_list[i]
                decoder_input = self.decoder_input_list[i]
                decoder_input_embedding = tf.nn.embedding_lookup(self.embedding, decoder_input)
                rnn_input = tf.concat([cnn_output, decoder_input_embedding], axis=1)
                assert rnn_input.shape == (FLAGS.BATCH_SIZE,
                                           FLAGS.DECODER_CNN_OUTPUT_NUM + FLAGS.EMBED_SIZE)

            with tf.name_scope("rnn_input"):
                rnn_input = tf.matmul(rnn_input, self.rnn_input_W) + self.rnn_input_b
                assert rnn_input.shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)

            with tf.name_scope("rnn_predict"):
                step_pred, state = self.cell(rnn_input, state)
                assert step_pred.shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)

                step_word = tf.matmul(step_pred, self.rnn2vocab_W) + self.rnn2vocab_b
                assert step_word.shape == (FLAGS.BATCH_SIZE, FLAGS.VOCAB_SIZE)

                pred.append(step_word)

        return pred

    # input text from previous output
    def rnn_valid_predict(self):
        pred = []
        state = self.init_states
        next_input = tf.nn.embedding_lookup(self.embedding, self.go_input)
        for i in range(FLAGS.SEQ_LEN):
            with tf.name_scope("decoder_input_embedding"):
                cnn_output = self.decoder_cnn_output_list[i]
                rnn_input = tf.concat([cnn_output, next_input], axis=1)
                assert rnn_input.shape == (FLAGS.BATCH_SIZE,
                                           FLAGS.DECODER_CNN_OUTPUT_NUM + FLAGS.EMBED_SIZE)

            with tf.name_scope("rnn_input"):
                rnn_input = tf.matmul(rnn_input, self.rnn_input_W) + self.rnn_input_b
                assert rnn_input.shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)

            with tf.name_scope("rnn_input"):
                step_pred, state = self.cell(rnn_input, state)
                assert step_pred.shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)

            with tf.name_scope("rnn_predict"):
                step_word = tf.matmul(step_pred, self.rnn2vocab_W) + self.rnn2vocab_b
                assert step_word.shape == (FLAGS.BATCH_SIZE, FLAGS.VOCAB_SIZE)

                pred.append(step_word)

                next_symbol = tf.stop_gradient(tf.argmax(step_word, 1))
                next_input = tf.nn.embedding_lookup(self.embedding, next_symbol)

        return pred
