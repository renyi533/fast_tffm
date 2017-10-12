import collections as pycoll
import operator
import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops

class StagedModelVariable(object):
  """Staging variable wrapper that decouples reads and updates.
  This class represents a variable through a staging buffer. Reads from this
  variable directly gets from the staging buffer. Updates are stacked into
  another staging buffer, and will be processed later.
  """

  def __init__(self, real_var, var_stage_get, put_ops):
    """Initializer for the model variables through a staging buffer.
    Args:
      real_var: the underlying real variable.
      var_stage_get: the read op from the staging buffer.
      variable_mgr: the parent variable-manager.
    """
    self.real_var = real_var
    self.var_stage_get = var_stage_get
    self.put_ops = put_ops

  def _value(self):
    """The read access of this variable. The content from the staging buffer."""
    return self.var_stage_get

  def _ref(self):
    """Return the underlying variable ref, required by tf.colocate_with."""
    return self.real_var._ref()  # pylint: disable=protected-access

  def read_value(self):
    """Mimics tf.Variable.read_value()."""
    return tf.identity(self.var_stage_get, name='read')

  @property
  def dtype(self):
    """Return the non-reference dtype."""
    return self.var_stage_get.dtype

  def assign_sub(self, delta, name=None):
    """Mimic the updates to the variable.
    Args:
      delta: is pushed into a staging buffer and will be pumped later.
      name: currently ignored; names of ops and the StagingArea are
            computed without using this pass name.
    Returns:
      The actual updates. The colocation constraint will be reapplied.
    """
    # This parameter is ignored: the StagingArea only supports setting
    # the shared name, not the names of individual ops it uses.
    del name

    # colocate_with(None, True) clears the colocation constraints.
    # Push the delta into a staging buffer.
    with ops.colocate_with(None, True), tf.device(self.var_stage_get.device):
      delta_staging_area = data_flow_ops.StagingArea(
          [self.var_stage_get.dtype], shapes=[self.var_stage_get.shape])
      delta_put_op = delta_staging_area.put([delta])
      self.put_ops.append(delta_put_op)
      delta_get_op = delta_staging_area.get()[0]
    # Return the actual updates. The colocation constraint will be reapplied.
    return self.real_var.assign_sub(delta_get_op)

  @staticmethod
  # pylint: disable=bad-staticmethod-argument,invalid-name
  def _TensorConversionFunction(self, dtype=None, name=None, as_ref=False):
    """Utility function for converting a StagedModelVariable to a Tensor."""
    del dtype, name  # unused: this function returns the cached ref or value.
    if as_ref:
      return self._ref()
    else:
      return self._value()


ops.register_tensor_conversion_function(
    StagedModelVariable, StagedModelVariable._TensorConversionFunction)  # pylint: disable=protected-access


class StagedVariableGetter(object):
  """A variable getter through staging buffers on devices.
  Instead of a caching device, this getter tracks where the variable is used.
  And on each device, it goes through a staging buffer.
  """

  def __init__(self, staging_ops, staged_vars,put_ops):
    """Initializer for StagedVariableGetter.
    Args:
      staging_ops: the staging put ops array
      staged_vars: the staged vars dict
    """
    self._staging_ops = staging_ops
    self._staged_vars = staged_vars
    self._put_ops = put_ops

  def __call__(self, getter, name, *args, **kwargs):
    real_var = getter(name, *args, **kwargs)
    staging_ops = self._staging_ops
    if real_var in staging_ops:
      put_op, get_op = staging_ops[real_var]
      return get_op
    shape = kwargs['shape']
    print(name+':')
    print(shape)
    dtype = kwargs['dtype']
    print(dtype)
    trainable = kwargs['trainable']
    # This helps copying the weights from the parameter to this server only
    # once.
    if real_var in self._staged_vars:
      cpu_var = self._staged_vars[real_var]
    else:
      cpu_var = tf.identity(real_var)
      self._staged_vars[real_var] = cpu_var
    var_to_stage = cpu_var

    staging_area = data_flow_ops.StagingArea([dtype], shapes=[shape])
    put_op = staging_area.put([var_to_stage])
    get_op = staging_area.get()
    staging_ops[real_var] = (put_op, get_op)
    self._put_ops.append(put_op)
    if trainable:
      # For trainable variables, they are managed separatedly through
      # apply_gradients
      #return real_var
      return get_op
    else:
      # For other shadow variables, the access is decoupled through a wrapper
      # class.
      return StagedModelVariable(real_var, get_op, self._put_ops)
