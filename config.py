from datetime import datetime

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('DATA_PATH', "dataset/mc_review2/data_mc.pkl", "")
flags.DEFINE_string('LABEL_PATH', "dataset/mc_review2/label_mc.pkl", "")
flags.DEFINE_string('DICT_PATH', "dictionary/mc_review2/dict_mc.pkl", "")

flags.DEFINE_integer('VOCAB_SIZE', 20000, '')
flags.DEFINE_integer('BATCH_SIZE', 32, '')
flags.DEFINE_integer('SEQ_LEN', 60, '')
flags.DEFINE_integer('EPOCH', 80, '')
flags.DEFINE_integer('BATCHES_PER_EPOCH', 1000, '')
flags.DEFINE_integer('LABELED_NUM', 30000, '')

flags.DEFINE_integer('DROPOUT_KEEP', 1.0, '')
flags.DEFINE_integer('LEARNING_RATE', 0.001, '')
flags.DEFINE_integer('LR_DECAY_START', 20, '')
flags.DEFINE_integer('ALPHA', 0.2, '')
flags.DEFINE_integer('MAX_GRAD', 5.0, '')

flags.DEFINE_integer('EMBED_SIZE', 80, '')
flags.DEFINE_integer('ENCODER_CNN_OUTPUT_NUM', 256, '')
flags.DEFINE_integer('DECODER_CNN_OUTPUT_NUM', 200, '')
flags.DEFINE_integer('LATENT_VARIABLE_SIZE', 50, '')

flags.DEFINE_integer('RNN_NUM', 1, '')
flags.DEFINE_integer('RNN_SIZE', 500, '')

flags.DEFINE_integer('KLD_ANNEAL_START', 10 * 1000, '')
flags.DEFINE_integer('KLD_ANNEAL_END', 17 * 1000, '')

flags.DEFINE_string('LOG_DIR', "log/log" + datetime.now().strftime("%y%m%d-%H%M"), "")

FLAGS = flags.FLAGS
