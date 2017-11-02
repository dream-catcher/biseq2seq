# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Collection of RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn

from seq2seq.encoders.encoder import Encoder, EncoderOutput
from seq2seq.training import utils as training_utils


def _unpack_cell(cell):
  """Unpack the cells because the stack_bidirectional_dynamic_rnn
  expects a list of cells, one per layer."""
  if isinstance(cell, tf.contrib.rnn.MultiRNNCell):
    return cell._cells  #pylint: disable=W0212
  else:
    return [cell]


def _default_rnn_cell_params():
  """Creates default parameters used by multiple RNN encoders.
  """
  return {
      "cell_class": "BasicLSTMCell",
      "cell_params": {
          "num_units": 128
      },
      "dropout_input_keep_prob": 1.0,
      "dropout_output_keep_prob": 1.0,
      "num_layers": 1,
      "residual_connections": False,
      "residual_combiner": "add",
      "residual_dense": False
  }


def _toggle_dropout(cell_params, mode):
  """Disables dropout during eval/inference mode
  """
  cell_params = copy.deepcopy(cell_params)
  if mode != tf.contrib.learn.ModeKeys.TRAIN:
    cell_params["dropout_input_keep_prob"] = 1.0
    cell_params["dropout_output_keep_prob"] = 1.0
  return cell_params


class Biseq2seqEncoder(Encoder):
  """
  A biseq2seq encoder. uses query and retrieved answer as encoder inputs
  respectively.
  reference: https://arxiv.org/abs/1610.07149

  Args:
    cell: An instance of tf.contrib.rnn.RNNCell
    name: A name for the encoder
  """

  def __init__(self, params, mode, name="biseq2seq_encoder"):
    super(Biseq2seqEncoder, self).__init__(params, mode, name)
    self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)

  @staticmethod
  def default_params():
    return {
        "rnn_cell": _default_rnn_cell_params(),
        "init_scale": 0.04,
    }

  def encode(self, inputs, sequence_length, **kwargs):
    scope = tf.get_variable_scope()
    scope.set_initializer(tf.random_uniform_initializer(
        -self.params["init_scale"],
        self.params["init_scale"]))

    query_inputs, retrieved_inputs = tf.split(inputs, 2)

    query_cell_fw = training_utils.get_rnn_cell(**self.params["rnn_cell"])
    query_cell_bw = training_utils.get_rnn_cell(**self.params["rnn_cell"])
    query_outputs, query_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=query_cell_fw,
        cell_bw=query_cell_bw,
        inputs=query_inputs,
        sequence_length=sequence_length,
        dtype=tf.float32,
        **kwargs)

    retrieved_cell_fw = training_utils.get_rnn_cell(**self.params["rnn_cell"])
    retrieved_cell_bw = training_utils.get_rnn_cell(**self.params["rnn_cell"])
    retrieved_outputs, retrieved_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=retrieved_cell_fw,
        cell_bw=retrieved_cell_bw,
        inputs=retrieved_inputs,
        sequence_length=sequence_length,
        dtype=tf.float32,
        **kwargs)

    outputs = tf.concat([query_outputs, retrieved_outputs], 0)
    # Concatenate outputs and states of the forward and backward RNNs
    outputs_concat = tf.concat(outputs, 2)

    states = tf.concat([query_states, retrieved_states], 0)

    return EncoderOutput(
        outputs=outputs_concat,
        final_state=states,
        attention_values=outputs_concat,
        attention_values_length=sequence_length)

