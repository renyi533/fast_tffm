# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from py.fm_ops import fm_ops

# Basic model parameters as external flags.
FLAGS = None
binary_parser_op = fm_ops

def inference(train_data, keep_prob=0.84, input_dimension=3996, sparse_input=False, ps_num=1, with_sparse_weights=False):
  # flatten
  # flatten = tf.reshape(data, [-1, 1])
  # flatten = tf.contrib.layers.flatten(inputs = data)

  with tf.variable_scope("layer1"):
    if not sparse_input:
      train_layer = full_connect_relu(train_data, [input_dimension, 256], [1,256], True, ps_num)
    else:
      train_layer = full_connect_sparse_relu(train_data, [input_dimension, 256], [1,256], True, ps_num, with_sparse_weights)
    train_layer = tf.nn.dropout(train_layer, keep_prob)


  with tf.variable_scope("layer2"):
    train_layer= full_connect_relu(train_layer, [256, 128], [1,128], ps_num = ps_num)
    train_layer = tf.nn.dropout(train_layer, keep_prob)

  with tf.variable_scope("layer3"):
    train_layer= full_connect_relu(train_layer, [128, 64], [1,64], ps_num = ps_num)
    train_layer = tf.nn.dropout(train_layer, keep_prob)

  with tf.variable_scope("layer4"):
    train_layer= full_connect_relu(train_layer, [64, 32], [32], ps_num = ps_num)
    train_layer = tf.nn.dropout(train_layer, keep_prob)

  with tf.variable_scope("layer5"):
    train_layer= full_connect(train_layer, [32, 1], [1], ps_num = ps_num)

  return train_layer

def fm_infer(train_data, input_dimension):
  weights = tf.get_variable("weights",
                             [input_dimension, 6],
                             #regularizer = tf.nn.l2_loss,
                             initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.01))
  segment_ids = train_data[0].indices[:, 0]
  if segment_ids.dtype != tf.int32:
    segment_ids = tf.cast(segment_ids, tf.int32)
  ori_ids, feature_ids, feature_poses = fm_ops.fm_sparse_tensor_parser(segment_ids, train_data[0].values, input_dimension)
  local_params = tf.nn.embedding_lookup(weights, ori_ids)
  pred_score, reg_score = fm_ops.fm_scorer(feature_ids, local_params, train_data[1].values, feature_poses, 1.0, 1.0)
  return pred_score


def full_connect(train_inputs, weights_shape, biases_shape, no_biases=False, ps_num=1):
  # weights
  if ps_num > 1:
    weights = tf.get_variable("weights",
                              weights_shape,
                              initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.01),
                              regularizer = tf.nn.l2_loss,
                              partitioner = tf.min_max_variable_partitioner(max_partitions=ps_num))
  else:
    weights = tf.get_variable("weights",
                              weights_shape,
                              regularizer = tf.nn.l2_loss,
                              initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.01))

  if no_biases:
      # matmul
      train = tf.matmul(train_inputs, weights)
      return train
  else:
      # biases
      biases = tf.get_variable("biases",
        biases_shape,
        initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.1))
      # matmul
      train = tf.matmul(train_inputs, weights) + biases
      return train

def full_connect_sparse(train_inputs, weights_shape, biases_shape, no_biases=False, ps_num=1, with_sparse_weights=False):
  # weights
  if ps_num > 1:
    weights = tf.get_variable("weights",
                              weights_shape,
                              initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.01),
                              #regularizer = tf.nn.l2_loss,
                              partitioner = tf.min_max_variable_partitioner(max_partitions=ps_num))
  else:
    weights = tf.get_variable("weights",
                              weights_shape,
                              #regularizer = tf.nn.l2_loss,
                              initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.01))
  if with_sparse_weights:
    train = tf.nn.embedding_lookup_sparse(weights,sp_ids=train_inputs[0],sp_weights=train_inputs[1],combiner="sum")
  else:
    train = tf.nn.embedding_lookup_sparse(weights,sp_ids=train_inputs[0],sp_weights=None, combiner="sum")

  if not no_biases:
      # biases
      biases = tf.get_variable("biases",
        biases_shape,
        initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.1))
      train = train + biases
  return train

def full_connect_sparse_relu(train_inputs,  weights_shape, biases_shape, no_biases=False, ps_num=1, with_sparse_weights=False):
  train= full_connect_sparse(train_inputs, weights_shape, biases_shape, no_biases, ps_num, with_sparse_weights)
  return tf.nn.relu(train)

# relu full connect
def full_connect_relu(train_inputs,  weights_shape, biases_shape, no_biases=False, ps_num=1):
  train= full_connect(train_inputs, weights_shape, biases_shape, no_biases, ps_num)
  return tf.nn.relu(train)


def loss_fn(logits, labels, weight_decay):
  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')

  reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  return tf.reduce_mean(cross_entropy, name='xentropy_mean') + weight_decay * tf.add_n(reg_losses)


def training(loss, global_step, worker_num, FLAGS):
  learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                           FLAGS.lr_decay_steps, FLAGS.lr_decay_rate, staircase=True)
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  #optimizer2 = tf.train.GradientDescentOptimizer(learning_rate)
  #optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
  #optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
  optimizer = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999, 1e-8)
  #optimizer = tf.train.QuantizationOptimizerWrapper(optimizer, FLAGS.task_index, True, True)

  '''
  if not FLAGS.async_training:
    if FLAGS.sync_mode ==1:
      optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                               replicas_to_aggregate=FLAGS.sync_agg_num if FLAGS.sync_agg_num > 0 else worker_num,
                                               total_num_replicas=worker_num,
                                               use_locking = FLAGS.use_lock)
    else:
      optimizer = tf.train.SemiSyncReplicasOptimizer(optimizer,
                                               batch_sync_num=FLAGS.sync_agg_num if FLAGS.sync_agg_num > 0 else worker_num,
                                               use_locking = FLAGS.use_lock)
  '''

  train_op = optimizer.minimize(loss, global_step=global_step)
  #train_op2 = optimizer2.minimize(loss)
  #train_op = tf.group(*[train_op, train_op2])
  return train_op, optimizer


def evaluation(logits, labels):
  labels = tf.to_int64(labels)
  labels = tf.reshape(labels, [-1])
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))


def auc(predictions, labels):
  _, auc_op = tf.contrib.metrics.streaming_auc(predictions, labels)
  return auc_op

def read_inputs(scope, filelist):
  with tf.variable_scope(scope):

    filename_queue = tf.train.string_input_producer(filelist, num_epochs=FLAGS.num_epochs)

    _, batch_example = tf.TFRecordReader().read_up_to(filename_queue, FLAGS.batch_size)

    labels, features = binary_parser_op.dense_binary_parser(batch_example, FLAGS.feature_dimension)
    labels.set_shape([None, 1])
    features.set_shape([None, FLAGS.feature_dimension])
    return labels, features

def read_sparse_inputs(scope, filelist):
  with tf.variable_scope(scope):
    filename_queue = tf.train.string_input_producer(filelist, num_epochs=FLAGS.num_epochs)
    _, batch_example = tf.TFRecordReader().read_up_to(filename_queue, FLAGS.batch_size)
    labels, indices, fvals, fids, dense_shape = binary_parser_op.sparse_binary_parser(batch_example)
    sparse_fids = tf.SparseTensor(indices, fids, dense_shape)
    sparse_fvals = tf.SparseTensor(indices, fvals, dense_shape)
    labels.set_shape([None, 1])
    return labels, sparse_fids, sparse_fvals

def get_input_file_list(data_dir):
  if (data_dir.startswith("hdfs://")):
      cmd = "`which hadoop` fs -ls %s | awk '{print $8}'" % data_dir
      p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
      return [x for x in p.stdout.read().split("\n") if x != '']
  else:
      return [os.path.join(data_dir, x) for x in os.listdir(data_dir)]

def run_training():
  # Tell TensorFlow that the model will be built into the default Graph.
  FLAGS.async_training = True

  with tf.Graph().as_default():
    train_list = get_input_file_list(FLAGS.data_dir)
    print(train_list)
    labels = None
    features = None
    if not FLAGS.sparse_input:
      labels, features = read_inputs("train", train_list)
    else:
      labels, sparse_fids, sparse_fvals = read_sparse_inputs("train", train_list)
      features = [sparse_fids, sparse_fvals]
    # Build a Graph that computes predictions from the inference model.
    train_logits= inference(features, FLAGS.keep_prob, FLAGS.feature_dimension, FLAGS.sparse_input)
    if FLAGS.sparse_input:
      fm_pred = fm_infer(features,  FLAGS.feature_dimension)
      fm_pred = tf.reshape(fm_pred, [-1,1])
      if FLAGS.mode == 1:
        train_logits = fm_pred
      else:
        train_logits = fm_pred + train_logits
    # Add to the Graph the Ops for loss calculation.
    loss = loss_fn(train_logits, labels, 0.0)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op,_ = training(loss, global_step, 1, FLAGS)
    train_predictions = tf.nn.sigmoid(train_logits, name='sigmoid')
    auc_op = auc(train_predictions, labels)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    # sess.run(init)
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        # Start the training loop.
        #for step in xrange(FLAGS.max_steps):
        step = 0
        while not coord.should_stop():

          start_time = time.time()

          # Run one step of the model.  The return values are the activations
          # from the `train_op` (which is discarded) and the `loss` Op.  To
          # inspect the values of your Ops or variables, you may include them
          # in the list passed to sess.run() and the value tensors will be
          # returned in the tuple from the call.

          _, loss_value, step, auc_value = sess.run([train_op, loss, global_step, auc_op])
          #print(py_labels)
          #print(py_preds)

          duration = time.time() - start_time

          # Write the summaries and print an overview fairly often.
          if step % 10 == 0:
            # Update the events file.
            summary_str = sess.run(summary)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            #auc_value = 0.5
            speed = FLAGS.batch_size / duration

            print('Step %d: loss = %.2f (%.2f /s) AUC %.3f' % (step, loss_value, speed, auc_value))

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    coord.join(threads)
    sess.close()

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--lr_decay_rate',
      type=float,
      default=1.0,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--lr_decay_steps',
      type=int,
      default=60000,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--keep_prob',
      type=float,
      default=1.0,
      help='drop out keep prob'
  )
  parser.add_argument(
      '--feature_dimension',
      type=int,
      default=100000,
      help='shape of features .'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=1,
      help='Number of epochs.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--mode',
      type=int,
      default=0,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--data_dir',
      type=str,
      default='hdfs://dnn1.dev.et2:9000/dnn_all_1_dol/sample/',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/tmp/tensorflow/kgbnet/logs/fully_connected_feed',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--sparse_input',
      default=False,
      help='If true, uses sparse input.',
      action='store_true'
  )
  parser.add_argument(
      '--weight_decay',
      type=float,
      default=0,
      help='weigth decay'
  )
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.95,
      help='weigth decay'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
