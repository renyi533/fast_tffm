import tensorflow as tf
import os, time
from tensorflow.python.framework import ops

eve_alg_op = tf.load_op_library(os.path.dirname(os.path.realpath(__file__)) + '/libeve_alg_op.so')

def eve_rate(curr_target, decay_rate, lower, upper, is_local_mode, worker_idx):
  device_str = "/cpu:0"
  var_collection = None
  if not is_local_mode:
    if worker_idx >= 0:
      device_str = "/job:worker/task:%d" % worker_idx
      var_collection = [tf.GraphKeys.LOCAL_VARIABLES]
    else:
      device_str = "/job:ps/task:0"
  print('set local eve device to:'+device_str)
  with tf.device(device_str), tf.name_scope("eve_scope"):
    d = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="d", collections=var_collection)
    f = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="f", collections=var_collection)
    step = tf.Variable(0, trainable=False, dtype=tf.int64, name="step", collections=var_collection)

  up = tf.convert_to_tensor(upper)
  low = tf.convert_to_tensor(lower)
  decay = tf.convert_to_tensor(decay_rate)

  new_f, new_d = eve_alg_op.eve_alg(f, d, curr_target, step, decay, low, up)

  f_assign = tf.assign(f, new_f)
  step_inc = tf.assign_add(step, 1)
  with tf.control_dependencies([f_assign, step_inc]):
    d_assign = tf.assign(d, new_d)
  return d_assign

