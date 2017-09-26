import tensorflow as tf
import os, time
from tensorflow.python.framework import ops

sparse_merge_op = tf.load_op_library(os.path.dirname(os.path.realpath(__file__)) + '/libsparse_merge_op.so')
fm_ops = sparse_merge_op
@ops.RegisterGradient("FmScorer")
def _fm_scorer_grad(op, pred_grad, reg_grad):
  feature_ids = op.inputs[0]
  feature_params = op.inputs[1]
  feature_vals = op.inputs[2]
  feature_poses = op.inputs[3]
  factor_lambda = op.inputs[4]
  bias_lambda = op.inputs[5]
  with ops.control_dependencies([pred_grad.op, reg_grad.op]):
    return None, fm_ops.fm_grad(feature_ids, feature_params, feature_vals, feature_poses, factor_lambda, bias_lambda, pred_grad, reg_grad), None, None, None, None

def sparse_merge(sp_inputs, max_fids, mode=0):
  max_fids = tf.convert_to_tensor(max_fids, dtype=tf.int64)

  if len(sp_inputs) == 1:
    return sp_inputs[0]

  inds = [sp_input.indices for sp_input in sp_inputs]
  vals = [sp_input.values for sp_input in sp_inputs]
  shapes = [sp_input.dense_shape for sp_input in sp_inputs]

  output_ind, output_val, output_shape = (sparse_merge_op.sparse_merge(
      inds, vals, shapes, max_fids, mode))

  return tf.SparseTensor(output_ind, output_val, output_shape)


def sparse_merge_val(sp_inputs, max_fids, mode=0):
  max_fids = tf.convert_to_tensor(max_fids, dtype=tf.int64)

  if len(sp_inputs) == 1:
    return sp_inputs[0]

  inds = [sp_input.indices for sp_input in sp_inputs]
  vals = [sp_input.values for sp_input in sp_inputs]
  shapes = [sp_input.dense_shape for sp_input in sp_inputs]

  output_ind, output_val, output_shape = (sparse_merge_op.sparse_merge_val(
      inds, vals, shapes, max_fids, mode))

  return tf.SparseTensor(output_ind, output_val, output_shape)

def merged_embedding_lookup_sparse(params,
                            sp_ids,
                            embedding_dim,
                            max_fids,
                            partition_strategy="div",
                            name=None,
                            combiner=None,
                            max_norm=None):
  input_len = len(sp_ids)
  sp_ids = sparse_merge(sp_ids, max_fids)

  result = tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights=None,
                            partition_strategy=partition_strategy,
                            name=name,
                            combiner=combiner,
                            max_norm=max_norm)
  result = tf.reshape(result, [-1, embedding_dim * input_len])
  return tf.split(result, num_or_size_splits=input_len, axis=1)

def sparse_concat(sp_inputs):
  if len(sp_inputs) == 1:
    return sp_inputs[0]

  inds = [sp_input.indices for sp_input in sp_inputs]
  vals = [sp_input.values for sp_input in sp_inputs]
  shapes = [sp_input.dense_shape for sp_input in sp_inputs]

  output_ind, output_val, output_shape = (sparse_merge_op.sparse_concatenate(inds, vals, shapes))

  return tf.SparseTensor(output_ind, output_val, output_shape)

def step_merged_embedding_lookup_sparse(params,
                            sp_ids,
                            embedding_dim,
                            max_fids,
                            partition_strategy="div",
                            name=None,
                            combiner=None,
                            max_norm=None):
  step_len = len(sp_ids)
  print('step_len:%d' % step_len)
  input_len = 0
  sp_tensor_vec = []
  for i in range(step_len):
    if (input_len != 0) and (input_len != len(sp_ids[i])):
      raise Exception('inconsisten input_len across steps')
    input_len = len(sp_ids[i])
    print('input_len:%d' % input_len)
    sp_tensor_vec.append(sparse_merge(sp_ids[i], max_fids))

  concated_tensor = sparse_concat(sp_tensor_vec)

  result = tf.nn.embedding_lookup_sparse(params, concated_tensor, sp_weights=None,
                            partition_strategy=partition_strategy,
                            name=name,
                            combiner=combiner,
                            max_norm=max_norm)
  result = tf.reshape(result, [-1, embedding_dim * input_len])

  result = tf.split(result, num_or_size_splits=step_len, axis=0)
  result_vec = []

  for i in range(step_len):
    result_vec.append(tf.split(result[i], num_or_size_splits=input_len, axis=1))

  return result_vec
