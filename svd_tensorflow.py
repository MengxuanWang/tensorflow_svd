import csv
import os
import datetime
import random
import numpy as np
import tensorflow as tf
from utils import *

#Parameters
# =================================================================
tf.flags.DEFINE_integer("embedding_dim", 30, "Dimensionality of embedding")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.flags.DEFINE_integer("num_epochs", 25, "Number of training epoches")
tf.flags.DEFINE_float("lrate", 0.1, "Learning rate")
tf.flags.DEFINE_float("l2_reg_lambda", 0.005, "L2 regularization lambda")
tf.flags.DEFINE_string("data_path", "train.csv", "file path where training data are")

FLAGS = tf.flags.FLAGS

# Model
# ==================================================================
class SVD:
    def __init__(self, nuser, nitem, avg_rat, dim, regular):

        self.input_user = tf.placeholder(tf.int32, shape=[None], name="id_user")
        self.input_item = tf.placeholder(tf.int32, shape=[None], name="id_item")
        self.input_rats = tf.placeholder(tf.float32, shape=[None])

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            w_user = tf.get_variable("embd_user", shape=[nuser, dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
            w_item = tf.get_variable("embd_item", shape=[nitem, dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
            embd_user = tf.nn.embedding_lookup(w_user, self.input_user, name="embedding_user")
            embd_item = tf.nn.embedding_lookup(w_item, self.input_item, name="embedding_item")

            w_bias_user = tf.get_variable("embd_bias_user", shape=[nuser])
            w_bias_item = tf.get_variable("embd_bias_item", shape=[nitem])
            bias_user = tf.nn.embedding_lookup(w_bias_user, self.input_user, name="bias_user")
            bias_item = tf.nn.embedding_lookup(w_bias_item, self.input_item, name="bias_item")
            bias_global = tf.constant(avg_rat, dtype=tf.float32, shape=[], name="bias_global")

        with tf.name_scope("infer"):
            self.infer = tf.reduce_sum(tf.mul(embd_user, embd_item), 1)
            self.infer = tf.add(self.infer, bias_global)
            self.infer = tf.add(self.infer, bias_user)
            self.infer = tf.add(self.infer, bias_item)

        with tf.name_scope("loss"):
            regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")
            cost = tf.nn.l2_loss(tf.sub(self.infer, self.input_rats))
            panelty = tf.constant(regular, dtype=tf.float32, shape=[], name="l2")
            self.cost = tf.reduce_mean(tf.add(cost, tf.mul(regularizer, panelty)))

# Training
# ==================================================
trainrats, nuser, nitem, mu = load_ratings(FLAGS.data_path, nsample=300000)
with tf.Graph().as_default():
    session = tf.Session()
    with session.as_default():
        svd = SVD(
            nuser = nuser,
            nitem = nitem,
            avg_rat = mu,
            dim = FLAGS.embedding_dim,
            regular = FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #train_op = tf.train.FtrlOptimizer(FLAGS.lrate).minimize(svd.cost)
        optimizer = tf.train.FtrlOptimizer(FLAGS.lrate)
        grads_and_vars = optimizer.compute_gradients(svd.cost)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        session.run(tf.initialize_all_variables())
        # A single training step
        def train_step(user_ids, item_ids, rats):
            feed_dict = {
                svd.input_user: user_ids,
                svd.input_item: item_ids,
                svd.input_rats: rats
            }
            # train_op returns nothing
            _, step, loss = session.run([train_op, global_step, svd.cost], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step % 1000 == 0:
                print "{}: step {}, loss {:g}".format(time_str, step, loss)

        def predicts_step(user_ids, item_ids):
            feed_dict = {
                svd.input_user: user_ids,
                svd.input_item: item_ids
            }
            predicts = session.run([svd.infer], feed_dict)
            return predicts

        batches = batch_iter(trainrats, FLAGS.batch_size, FLAGS.num_epochs)
        for batch in batches:
            user_ids, item_ids, rats = batch[:, 0], batch[:, 1], batch[:, 2]
            train_step(user_ids, item_ids, rats)
