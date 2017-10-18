import os
import sys
import shutil

sys.path.append("model")
sys.path.append("utils")

import numpy as np
import tensorflow as tf

from vae import VAE
from config import FLAGS
from batchloader import BatchLoader

def log_and_print(log_file, logstr, br=True):
    print(logstr)

    if(br):
        logstr = logstr + "\n"
    with open(log_file, 'a') as f:
        f.write(logstr)

def main():
    os.mkdir(FLAGS.LOG_DIR)
    os.mkdir(FLAGS.LOG_DIR + "/model")
    log_file = FLAGS.LOG_DIR + "/log.txt"
    shutil.copyfile("config.py", FLAGS.LOG_DIR + "/config.py")

    # gpu memory
    sess_conf = tf.ConfigProto(
        gpu_options = tf.GPUOptions(
            # allow_growth = True
        )
    )

    with tf.Graph().as_default():
        with tf.Session(config=sess_conf) as sess:
            batchloader = BatchLoader(with_label=False)

            with tf.variable_scope("VAE"):
                vae = VAE(batchloader, is_training=True, ru=False)

            with tf.variable_scope("VAE", reuse=True):
                vae_test = VAE(batchloader, is_training=False, ru=True)

            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(FLAGS.LOG_DIR, sess.graph)

            sess.run(tf.global_variables_initializer())

            log_and_print(log_file, "start training")

            loss_sum = []
            rnn_loss_sum = []
            aux_loss_sum = []
            kld_sum = []

            lr = FLAGS.LEARNING_RATE
            step = 0
            for epoch in range(FLAGS.EPOCH):
                log_and_print(log_file, "epoch %d" % (epoch+1))
                if epoch >= FLAGS.LR_DECAY_START:
                    lr *= 0.95
                for batch in range(FLAGS.BATCHES_PER_EPOCH):

                    step += 1

                    encoder_input, decoder_input, target = \
                                        batchloader.next_batch(FLAGS.BATCH_SIZE, "train")
                    feed_dict = {vae.encoder_input: encoder_input,
                                 vae.decoder_input: decoder_input,
                                 vae.target: target,
                                 vae.step: step,
                                 vae.lr: lr}

                    aux_logits, rnn_logits, loss, rnn_loss, aux_loss, kld, merged_summary, _ \
                        = sess.run([vae.aux_logits, vae.rnn_logits, vae.loss, vae.rnn_loss,
                                    vae.aux_loss, vae.kld, vae.merged_summary, vae.train_op],
                                   feed_dict = feed_dict)

                    rnn_loss_sum.append(rnn_loss)
                    aux_loss_sum.append(aux_loss)
                    kld_sum.append(kld)
                    loss_sum.append(loss)
                    summary_writer.add_summary(merged_summary, step)

                    if(batch % 50 == 49):
                        log_and_print(log_file, "epoch %d batch %d" % \
                                                ((epoch+1), (batch+1)), br=False)

                        ave_loss = np.average(loss_sum)
                        log_and_print(log_file, "\tloss: %f" % ave_loss, br=False)
                        ave_rnnloss = np.average(rnn_loss_sum)
                        log_and_print(log_file, "\trnn_loss: %f" % ave_rnnloss, br=False)
                        ave_auxloss = np.average(aux_loss_sum)
                        log_and_print(log_file, "\taux_loss: %f" % ave_auxloss, br=False)
                        ave_kld = np.average(kld_sum)
                        log_and_print(log_file, "\tkld %f" % ave_kld, br=False)

                        loss_sum = []
                        rnn_loss_sum = []
                        aux_loss_sum = []
                        kld_sum = []

                        # train input, output
                        # output input and logits
                        sample_train_input, sample_train_input_list \
                            = sess.run([vae.encoder_input, vae.encoder_input_list],
                                       feed_dict = feed_dict)
                        encoder_input_texts = batchloader.logits2str(sample_train_input_list,
                                                                     1,
                                                                     onehot=False,
                                                                     numpy=True)

                        log_and_print(log_file, "\ttrain input: %s" % encoder_input_texts[0])
                        sample_train_rnn_outputs = batchloader.logits2str(rnn_logits, 1)
                        sample_train_aux_outputs = batchloader.logits2str(aux_logits, 1)
                        log_and_print(log_file, "\ttrain rnn output: %s" % sample_train_rnn_outputs[0])
                        log_and_print(log_file, "\ttrain aux output: %s" % sample_train_aux_outputs[0])


                        # validation output
                        sample_input, _, sample_target = batchloader.next_batch(FLAGS.BATCH_SIZE, "test")
                        sample_input_list, sample_latent_variables = \
                            sess.run([vae_test.encoder_input_list, vae_test.encoder.latent_variables],
                                     feed_dict = {vae_test.encoder_input: sample_input})
                        sample_aux_logits, sample_rnn_logits, valid_aux_loss, \
                            valid_rnn_loss, merged_summary = \
                                sess.run([vae_test.aux_logits, vae_test.rnn_logits,
                                          vae_test.aux_loss, vae_test.rnn_loss, vae_test.merged_summary],
                                          feed_dict = {vae_test.target: sample_target,
                                                       vae_test.latent_variables: sample_latent_variables})

                        log_and_print(log_file, "\tvalid rnn loss: %f" % valid_rnn_loss)
                        log_and_print(log_file, "\tvalid aux loss: %f" % valid_aux_loss)
                        sample_input_texts = batchloader.logits2str(sample_input_list,
                                                                    1, onehot=False, numpy=True)
                        sample_rnn_samples = batchloader.logits2str(sample_rnn_logits, 1)
                        sample_aux_samples = batchloader.logits2str(sample_aux_logits, 1)
                        log_and_print(log_file, "\tsample input: %s" % sample_input_texts[0])
                        log_and_print(log_file, "\tsample rnn output: %s" % sample_rnn_samples[0])
                        log_and_print(log_file, "\tsample aux output: %s" % sample_aux_samples[0])

                        summary_writer.add_summary(merged_summary, step)

                # save model
                save_path = saver.save(sess, FLAGS.LOG_DIR + ("/model/model%d.ckpt" % (epoch+1)))
                log_and_print(log_file, "Model saved in file %s" % save_path)

if __name__ == "__main__":
    main()
