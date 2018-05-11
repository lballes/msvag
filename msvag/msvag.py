# -*- coding: utf-8 -*-
"""
This module contains a TensorFlow implementation of the M-SVAG and SVAG
optimization methods of [1]. ``MSVAGOptimizer`` and ``SVAGOptimizer`` inherit
from ``tf.train.Optimizer`` and can be used as direct drop-in replacement for
TensorFlow's built-in optimizers.

[1]: Lukas Balles and Philipp Hennig. Dissecting Adam: The Sign, Magnitude and
Variance of Stochastic Gradients. (https://arxiv.org/abs/1705.07774)
"""

import tensorflow as tf


class MSVAGOptimizer(tf.train.Optimizer):
  
  """This class implements the M-SVAG optimizer of [1].
  
  The M-SVAG optimizer maintains moving averages of past gradients (``m``) and
  their element-wise square (``v``) to obtain an estimate of the gradient
  variance ``s=(v-m**2)/(1-rho)``, where ``rho`` is a scalar factor (see [1] for
  details). It then updates along the direction
  
      (m**2/(m**2 + rho*s)) * m
  
  [1]: Lukas Balles and Philipp Hennig. Dissecting Adam: The Sign, Magnitude and
  Variance of Stochastic Gradients. (https://arxiv.org/abs/1705.07774)
  """
  
  def __init__(self,
               learning_rate,
               beta=0.9,
               use_locking=False,
               name="MSVAG"):
    """Construct a new MSVAGOptimizer optimizer.
    
    Args:
      :learning_rate: Learning rate (scalar tensor or float value).
      :beta: Moving average (momentum) constant (scalar tensor or float value).
      :use_locking: If True use locks for update operations.
      :name: Optional name prefix for the created ops (default: "MSVAG").
    """
    
    super(MSVAGOptimizer, self).__init__(use_locking, name=name)
    self._lr = learning_rate
    self._beta = beta
    
    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta_t = None
    
    # Variable track of the (t+1)-th power of beta, created in _prepare().
    self._beta_power = None
  
  def _get_beta_accumulator(self):
    if tf.contrib.eager.executing_eagerly():
      graph = None
    else:
      graph = tf.get_default_graph()
    return self._get_non_slot_variable("beta_power", graph=graph)

  def _create_slots(self, var_list):
    # Create the beta accumulator on the same device as the first variable.
    # Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored). The accumulator is initialized to beta, since it is 
    # updated in _finish(), i.e., after each iteration.
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(initial_value=self._beta,
                                   name="beta_power",
                                   colocate_with=first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m_tilde", self._name)
      self._zeros_slot(v, "v_tilde", self._name)
    
  def _prepare(self):
    # Learning rate and momentum parameter as tensors
    self._lr_t = tf.convert_to_tensor(self._lr, name="lr")
    self._beta_t = tf.convert_to_tensor(self._beta, name="beta")
  
  def _apply_dense(self, grad, var):
    # Add update operations for slot variables
    mt = self.get_slot(var, "m_tilde")
    vt = self.get_slot(var, "v_tilde")
    mt_update = mt.assign(self._beta_t*mt+(1.0-self._beta_t)*grad,
                          use_locking=self._use_locking) 
    vt_update = vt.assign(self._beta_t*vt+(1.0-self._beta_t)*tf.square(grad),
                          use_locking=self._use_locking)
    
    with tf.control_dependencies([mt_update, vt_update]):
      # Initialization-bias-corrected moving averages
      beta_power = self._get_beta_accumulator()
      m = mt/(1.0-beta_power)
      v = vt/(1.0-beta_power)
      
      # Correction factor rho. Cap to avoid issues with rho=1 in first iter
      rho = tf.square(1.0-self._beta_t) * (1.0-tf.square(beta_power))
      rho /= tf.square(1.0-beta_power) * (1.0-tf.square(self._beta_t))
      rho = tf.minimum(rho, 0.9999)
      
      # Variance estimate
      m2 = tf.square(m)
      s = (v-m2)/(1.0-rho)
      
      # Variance adaptation factors
      factor = m2 / (m2 + rho*s)
      
      # Remove NaNs (if m=v=0) and clip to [0,1] (due to round-off errors)
      factor = tf.clip_by_value(factor, 0.0, 1.0)
      factor = tf.where(tf.is_nan(factor), tf.zeros_like(factor), factor)
      
      # Return variable update operation
      return var.assign_sub(self._lr_t*factor*m, use_locking=self._use_locking)
                              
  def _resource_apply_dense(self, grad, var):
    raise NotImplementedError("_resource_apply_dense is not implemented for "
                              "MSVAGOptimizer.")

  def _apply_sparse(self, grad, var):
    tf.logging.info("_apply_sparse is not implemented for MSVAGOptimizer. "
                    "Reverting to _apply_dense.")
    return self._apply_dense(grad, var)

  def _resource_apply_sparse(self, grad, var, indices):
    raise NotImplementedError("_resource_apply_sparse is not implemented for "
                              "MSVAGOptimizer.")
  
  def _finish(self, update_ops, name_scope):
    # Update the power-of-beta accumulator.
    beta_power = self._get_beta_accumulator()
    with tf.control_dependencies(update_ops):
      with tf.colocate_with(beta_power):
        update_beta_power = beta_power.assign(
            beta_power*self._beta_t,
            use_locking=self._use_locking)
    
    return tf.group(*update_ops + [update_beta_power], name=name_scope)


class SVAGOptimizer(tf.train.Optimizer):
  
  """This class implements the SVAG optimizer of [1].
  
  The M-SVAG optimizer maintains moving averages of past gradients (``m``) and
  their element-wise square (``v``) to obtain an estimate of the gradient
  variance ``s=(v-m**2)/(1-rho)``, where ``rho`` is a scalar factor (see [1] for
  details). It then updates along the direction
  
      (m**2/(m**2 + s)) * g
  
  where ``g`` is the local stochastic gradient.
  
  [1]: Lukas Balles and Philipp Hennig. Dissecting Adam: The Sign, Magnitude and
  Variance of Stochastic Gradients. (https://arxiv.org/abs/1705.07774)
  """
  
  def __init__(self,
               learning_rate,
               beta=0.9,
               use_locking=False,
               name="SVAG"):
    """Construct a new SVAGOptimizer optimizer.
    
    Args:
      :learning_rate: Learning rate (scalar tensor or float value).
      :beta: Moving average constant (scalar tensor or float value).
      :use_locking: If True use locks for update operations.
      :name: Optional name prefix for the created ops (default: "SVAG").
    """
    
    super(SVAGOptimizer, self).__init__(use_locking, name=name)
    self._lr = learning_rate
    self._beta = beta
    
    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta_t = None
    
    # Variable track of the (t+1)-th power of beta, created in _prepare().
    self._beta_power = None
  
  def _create_slots(self, var_list):
    for v in var_list:
      self._zeros_slot(v, "m_tilde", self._name)
      self._zeros_slot(v, "v_tilde", self._name)
  
  def _prepare(self):
    # Learning rate and momentum parameter as tensors
    self._lr_t = tf.convert_to_tensor(self._lr, name="lr")
    self._beta_t = tf.convert_to_tensor(self._beta, name="beta")
    
    # Variable to maintain (t+1)-th power of beta. Initialize to beta. It is
    # updated in _finish, i.e. *after* each iteration.
    self._beta_power = tf.Variable(self._beta, trainable=False, name="beta_power")
    
    # Correction factor rho
    self._rho = tf.square(1.0-self._beta_t) * (1.0-tf.square(self._beta_power))
    self._rho /= tf.square(1.0-self._beta_power) * (1.0-tf.square(self._beta_t))
    
    # To avoid issues with rho=1 in the first iteration, we add this minimum
    # operation. It will only take effect in the first iteration (where the
    # exact value of rho is irrelevant, since v-m**2, leading to s=0).
    self._rho = tf.minimum(self._rho, 0.999)
  
  def _apply_dense(self, grad, var):
    # Add update operations for slot variables
    mt = self.get_slot(var, "m_tilde")
    vt = self.get_slot(var, "v_tilde")
    mt_update = mt.assign(self._beta_t*mt+(1.0-self._beta_t)*grad,
                          use_locking=self._use_locking) 
    vt_update = vt.assign(self._beta_t*vt+(1.0-self._beta_t)*tf.square(grad),
                          use_locking=self._use_locking)
    
    with tf.control_dependencies([mt_update, vt_update]):
      # initialization-bias-corrected moving averages
      m = mt/(1.0-self._beta_power)
      v = vt/(1.0-self._beta_power)
      
      # Variance estimate
      m2 = tf.square(m)
      s = (v-m2)/(1.0-self._rho)
      
      # Compute variance adaptation factors
      factor = m2 / (m2 + s)
      
      # Remove possible NaNs and clip to be between 0 and 1. NaNs occur only in
      # the case that m=v=0. Values outside [0,1] might occur due to numerical
      # imprecision
      factor = tf.clip_by_value(factor, 0.0, 1.0)
      factor = tf.where(tf.is_nan(factor), tf.zeros_like(factor), factor)
      
      # Return variable update operation
      return var.assign_sub(self._lr_t*factor*grad, use_locking=self._use_locking)
                              
  def _resource_apply_dense(self, grad, var):
    raise NotImplementedError("_resource_apply_dense is not implemented for "
                              "SVAGOptimizer.")

  def _apply_sparse(self, grad, var):
    tf.logging.info("_apply_sparse is not implemented for SVAGOptimizer. "
                    "Reverting to _apply_dense.")
    return self._apply_dense(grad, var)

  def _resource_apply_sparse(self, grad, var, indices):
    raise NotImplementedError("_resource_apply_sparse is not implemented for "
                              "SVAGOptimizer.")
  
  def _finish(self, update_ops, name_scope):
    # Update the power-of-beta accumulator.
    with tf.control_dependencies(update_ops):
      with tf.colocate_with(self._beta_power):
        update_beta_power = self._beta_power.assign(
            self._beta_power*self._beta_t,
            use_locking=self._use_locking)
    
    return tf.group(*update_ops + [update_beta_power], name=name_scope)