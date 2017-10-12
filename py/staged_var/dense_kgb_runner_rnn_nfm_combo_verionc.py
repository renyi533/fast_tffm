import argparse
import json
import os.path
import subprocess
import sys
import time
import numpy as np

import dense_kgbnet_rnn_nfm_combo_versionc as kgbnet
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tf_client import base_runner
from tensorflow.python.client import timeline
from datetime import *
import base_summary
from staged_var import StagedVariableGetter 
from StagedVarOptimizer import StagedVarOptimizer
# Basic model parameters as external flags.
FLAGS = None

def run_training():
    class LoggerHook(tf.train.SessionRunHook):
        def __init__(self, vars):
            #self.vars = vars
            self.vars = []
            for i in range(len(vars)):
                self.vars.append(vars[i].op)
            self._iter_cnt = 0
        def before_run(self, run_context):
            if self._iter_cnt > 0:
                return tf.train.SessionRunArgs(self.vars)
        def after_run(self, run_context, run_values):
            self._iter_cnt = self._iter_cnt + 1

    class StagingAreaHook(tf.train.SessionRunHook):
        def __init__(self, put_ops):
            self._put_ops = put_ops
            self._iter_cnt = 0
        def after_create_session(self, session, coord):
            print('put ops:')
            print(self._put_ops)
            session.run(self._put_ops)
        def before_run(self, run_context):
            #if self._iter_cnt == 0:
            #    session = run_context.session
            #    session.run(self._put_ops)
            #else:
            return tf.train.SessionRunArgs(self._put_ops)

        def after_run(self, run_context, run_values):
            self._iter_cnt = self._iter_cnt + 1

    class KGBRunner(base_runner.BaseRunner):

      def wrap_optimizer(self, optimizer, props):
        optimizer = StagedVarOptimizer(optimizer, self._staging_ops)
        return optimizer 

      def inference(self, labels, features):
        step_num = self.get_window_size()  # can be changed!
        self._staging_ops = {}
        self._put_ops = []
        self._staged_vars = {}
        custom_getter = StagedVariableGetter(self._staging_ops, self._staged_vars, self._put_ops)
        labels = features.data[0]
        train_logits, vars = kgbnet.inference(features, step_num, self.get_job_conf('keep_prob'), self.get_ps_num(), custom_getter)
        train_logits = tf.reshape(train_logits, [-1, step_num])
        
        #real_size_dense = features.data[38]
        real_size_dense = features.data[len(features.data) - 1]
        batch_size = tf.shape(real_size_dense)[0]
        seqlen = tf.reshape(real_size_dense, [-1])
        seqlen = tf.cast(seqlen, tf.int32)

        with tf.variable_scope("mask"):
          #-- each loss
          lower_triangular_ones = tf.constant(np.tril(np.ones([step_num,step_num])), dtype=tf.float32)
          mask = tf.gather(lower_triangular_ones, seqlen - 1)

          #-- last loss
          #diagonal_ones = tf.constant(np.eye(step_num), dtype=tf.float32)
          #mask = tf.gather(diagonal_ones, seqlen - 1)
        
        with tf.variable_scope("idx"):
          #-- each loss
          #lower_triangular_bools = tf.constant(np.tril(np.ones([step_num,step_num], dtype=bool)))
          #idx = tf.gather(lower_triangular_bools, seqlen - 1)

          #-- last loss
          #idx = tf.range(0, batch_size) * step_num + (seqlen - 1)
          diagonal_ones = tf.constant(np.eye(step_num), dtype=tf.bool)
          idx = tf.gather(diagonal_ones, seqlen - 1)

        loss = kgbnet.loss(train_logits, labels, mask, self.get_job_conf('weight_decay'))
        
        train_predictions = tf.nn.sigmoid(train_logits, name='sigmoid_auc')
        base_runner.add_trace_variable('train_predictions', train_predictions)
        auc_op = kgbnet.auc(train_predictions, labels, idx, self.get_job_conf('auc_bucket_num'))
        #base_summary.f_summary_variables() #output all globle variables
        #base_summary.f_summary_gradients(loss) #compute and  output gradients for all trainable variables
        #base_summary.f_summary_collection() #output other tensors collected by  base_summary.f_summary_add_to_collection()
        #vars.append(loss)
        #vars.append(train_logits)
        self.append_session_run_hook(LoggerHook(vars))
        self.append_session_run_hook(StagingAreaHook(self._put_ops))
        
        return loss, auc_op
        
    KGBRunner(FLAGS.config).run()
def main(_):
  run_training()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument(
      '--config',
      type=str,
      help='drop out keep prob'
  )
    
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
