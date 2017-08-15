import tensorflow as tf
import os, time
from tensorflow.python.framework import ops
from merged_sparse_embedding.operations import sparse_merge
from merged_sparse_embedding.operations import sparse_merge_val
from merged_sparse_embedding.operations import merged_embedding_lookup_sparse

sess = tf.InteractiveSession()
a_indices = tf.constant([[0,1],[0,2],[1,0],[1,1],[1,3]], dtype=tf.int64)
a_vals = tf.constant([0,1,5,4,6], dtype=tf.int64)
a_f_vals = tf.constant([0,1,5,4,6], dtype=tf.float32)
a_shape = tf.constant([2, 4], dtype=tf.int64)
sp_tensor_a = tf.SparseTensor(indices = a_indices, values= a_vals, dense_shape=a_shape)
sp_tensor_fa = tf.SparseTensor(indices = a_indices, values= a_f_vals, dense_shape=a_shape)


b_indices = tf.constant([[0,0],[1,0],[1,1]], dtype=tf.int64)
b_vals = tf.constant([0,0,1], dtype=tf.int64)
b_f_vals = tf.constant([0,0,1], dtype=tf.float32)
b_shape = tf.constant([2, 2], dtype=tf.int64)
sp_tensor_b = tf.SparseTensor(indices = b_indices, values= b_vals, dense_shape=b_shape)
sp_tensor_fb = tf.SparseTensor(indices = b_indices, values= b_f_vals, dense_shape=b_shape)

print('original sparse tensors')

print(sp_tensor_a.eval())
print(sp_tensor_b.eval())

print('test special case of merge 1 single tensor')

max_fids = [6]
print('mode 0')
result = sparse_merge([sp_tensor_a], max_fids, 0)
print(result.eval())


print('mode 1')
result = sparse_merge([sp_tensor_a], max_fids, 1)
print(result.eval())


print('test merge 2 tensors')
max_fids = [6, 1]
print('mode 0')

result = sparse_merge([sp_tensor_a, sp_tensor_b], max_fids, 0)

print(result.eval())
print('mode 1')

result = sparse_merge([sp_tensor_a, sp_tensor_b], max_fids, 1)

print(result.eval())

print('test merge 2 tensor vals')
max_fids = [6, 1]
print('mode 0')

result = sparse_merge_val([sp_tensor_fa, sp_tensor_fb], max_fids, 0)

print(result.eval())
print('mode 1')

result = sparse_merge_val([sp_tensor_fa, sp_tensor_fb], max_fids, 1)

print(result.eval())

print('test merge 3 tensors')
max_fids = [6, 1, 6]
print('mode 0')

result = sparse_merge([sp_tensor_a, sp_tensor_b, sp_tensor_a], max_fids,0)

print(result.eval())
print('mode 1')

result = sparse_merge([sp_tensor_a, sp_tensor_b, sp_tensor_a], max_fids,1)

print(result.eval())


print('test whole merged embedding process')
var = tf.get_variable('embedding_var', [16, 8], initializer=tf.constant_initializer(0.1))
sess.run(tf.global_variables_initializer())

result1, result2, result3 = merged_embedding_lookup_sparse(var, [sp_tensor_a, sp_tensor_b, sp_tensor_a], 8, max_fids, name='merged_embedding', combiner='sum')

print(result1.eval())
print(result2.eval())
print(result3.eval())

sess.close()
