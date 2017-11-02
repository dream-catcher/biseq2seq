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
Definition of a basic seq2seq model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pydoc import locate
import tensorflow as tf
from seq2seq.contrib.seq2seq import helper as tf_decode_helper

from seq2seq import graph_utils
from seq2seq import decoders
from seq2seq.data import vocab
from seq2seq.models.basic_seq2seq import BasicSeq2Seq
from seq2seq.graph_utils import templatemethod
from seq2seq.models import bridges
from seq2seq.encoders.encoder import Encoder, EncoderOutput


class BasicBiSeq2Seq(BasicSeq2Seq):
  """Basic Sequence2Sequence model with a unidirectional encoder and decoder.
  The last encoder state is used to initialize the decoder and thus both
  must share the same type of RNN cell.

  Args:
    source_vocab_info: An instance of `VocabInfo`
      for the source vocabulary
    target_vocab_info: An instance of `VocabInfo`
      for the target vocabulary
    params: A dictionary of hyperparameters
  """

  def __init__(self, params, mode, name="basic_biseq2seq"):
    super(BasicBiSeq2Seq, self).__init__(params, mode, name)
    # add candidate answer part
    self.source_candidate_vocab_info = None
    if "vocab_source_candidate" in self.params and self.params["vocab_source_candidate"]:
        self.source_candidate_vocab_info = vocab.get_vocab_info(self.params["vocab_source_candidate"])
    self.encoder_class = locate(self.params["encoder.class"])
    self.decoder_class = locate(self.params["decoder.class"])

  @staticmethod
  def default_params():
    params = BasicSeq2Seq.default_params().copy()
    params.update({
        "bridge.class": "seq2seq.models.bridges.InitialStateBridge",
        "bridge.params": {},
        "encoder.class": "seq2seq.encoders.UnidirectionalRNNEncoder",
        "encoder.params": {},  # Arbitrary parameters for the encoder
        "decoder.class": "seq2seq.decoders.BasicDecoder",
        "decoder.params": {},  # Arbitrary parameters for the decoder
        "source_candidate.max_seq_len": 20,
        "source_candidate.reverse": True,
        "vocab_source_candidate": None
    })
    return params

  def _create_bridge(self, encoder_outputs, decoder_state_size):
    """Creates the bridge to be used between encoder and decoder"""
    bridge_class = locate(self.params["bridge.class"]) or \
      getattr(bridges, self.params["bridge.class"])
    return bridge_class(
        encoder_outputs=encoder_outputs,
        decoder_state_size=decoder_state_size,
        params=self.params["bridge.params"],
        mode=self.mode)

  def _create_decoder(self, _encoder_output, _features, _labels):
    """Creates a decoder instance based on the passed parameters."""
    return self.decoder_class(
        params=self.params["decoder.params"],
        mode=self.mode,
        vocab_size=self.target_vocab_info.total_size)

  def _decode_train(self, decoder, bridge, _encoder_output, _features, labels):
    """Runs decoding in training mode"""
    target_embedded = tf.nn.embedding_lookup(self.target_embedding,
                                             labels["target_ids"])
    helper_train = tf_decode_helper.TrainingHelper(
        inputs=target_embedded[:, :-1],
        sequence_length=labels["target_len"] - 1)
    decoder_initial_state = bridge()
    print("basic_seq2seq decoder_initial_state:{}".format(decoder_initial_state))
    return decoder(decoder_initial_state, helper_train)

  def _decode_infer(self, decoder, bridge, _encoder_output, features, labels):
    """Runs decoding in inference mode"""
    batch_size = self.batch_size(features, labels)
    if self.use_beam_search:
      batch_size = self.params["inference.beam_search.beam_width"]

    target_start_id = self.target_vocab_info.special_vocab.SEQUENCE_START
    helper_infer = tf_decode_helper.GreedyEmbeddingHelper(
        embedding=self.target_embedding,
        start_tokens=tf.fill([batch_size], target_start_id),
        end_token=self.target_vocab_info.special_vocab.SEQUENCE_END)
    decoder_initial_state = bridge()
    return decoder(decoder_initial_state, helper_infer)

  @templatemethod("encode")
  def encode(self, features, labels):
    # source_embedded = tf.nn.embedding_lookup(self.source_embedding,
    #                                          features["source_ids"])
    # encoder_fn = self.encoder_class(self.params["encoder.params"], self.mode)
    # return encoder_fn(source_embedded, features["source_len"])

    # 1. query source encoder sequence output
    query_embedded = tf.nn.embedding_lookup(self.source_embedding,
                                            features["source_ids"])
    query_encoder_fn = self.encoder_class(self.params["encoder.params"], self.mode)
    query_output = query_encoder_fn(query_embedded, features["source_len"])
    # return query_output
    # 2. candidate source encoder sequence output
    candidate_embedded = tf.nn.embedding_lookup(self.source_candidate_embedding,
                                                features["source_candidate_ids"])
    candidate_encoder_fn = self.encoder_class(self.params["encoder.params"], self.mode)
    candidate_output = candidate_encoder_fn(candidate_embedded, features["source_candidate_len"])

    print("query_output:{}".format(query_output))
    print("candidate_output:{}".format(candidate_output))
    # 3. merge two encoder generated output
    # outputs = tf.concat([query_output.outputs, candidate_output.outputs], 0)
    # #final_state = tf.reshape(tf.concat([query_output.final_state, candidate_output.final_state], 0), [-1, 128])
    # final_state = tf.concat([query_output.final_state, candidate_output.final_state], 0)
    # attention_values = tf.concat([query_output.attention_values, candidate_output.attention_values], 0)
    # att_v_len = tf.concat([query_output.attention_values_length, candidate_output.attention_values_length], 0)

    outputs = query_output.outputs + candidate_output.outputs
    final_state = (query_output.final_state[0] + candidate_output.final_state[0],
                   query_output.final_state[1] + candidate_output.final_state[1])
    attention_values = query_output.attention_values + candidate_output.attention_values
    att_v_len = query_output.attention_values_length + candidate_output.attention_values_length
    
    encoderOutput = EncoderOutput(outputs=outputs,
                                  final_state=final_state,
                                  attention_values=attention_values,
                                  attention_values_length=att_v_len)
    print("encoderOutput:{}".format(encoderOutput))
    return encoderOutput

  @templatemethod("decode")
  def decode(self, encoder_output, features, labels):
    decoder = self._create_decoder(encoder_output, features, labels)
    if self.use_beam_search:
      decoder = self._get_beam_search_decoder(decoder)

    bridge = self._create_bridge(
        encoder_outputs=encoder_output,
        decoder_state_size=decoder.cell.state_size)
    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      return self._decode_infer(decoder, bridge, encoder_output, features,
                                labels)
    else:
      return self._decode_train(decoder, bridge, encoder_output, features,
                                labels)
  @property
  @templatemethod("source_candidate_embedding")
  def source_candidate_embedding(self):
    """Returns the embedding used for the source sequence.
    """
    return tf.get_variable(
        name="W_candidate",
        shape=[self.source_candidate_vocab_info.total_size, self.params["embedding.dim"]],
        initializer=tf.random_uniform_initializer(
            -self.params["embedding.init_scale"],
            self.params["embedding.init_scale"]))

  def _preprocess(self, features, labels):
    """Model-specific preprocessing for features and labels:

    - Creates vocabulary lookup tables for source and target vocab
    - Converts tokens into vocabulary ids
    """

    # Create vocabulary lookup for source
    source_vocab_to_id, source_id_to_vocab, source_word_to_count, _ = \
      vocab.create_vocabulary_lookup_table(self.source_vocab_info.path)

    source_candidate_vocab_to_id, source_candidate_id_to_vocab, source_candidate_word_to_count, _ = \
        vocab.create_vocabulary_lookup_table(self.source_candidate_vocab_info.path)

    # Create vocabulary look for target
    target_vocab_to_id, target_id_to_vocab, target_word_to_count, _ = \
      vocab.create_vocabulary_lookup_table(self.target_vocab_info.path)

    # Add vocab tables to graph colection so that we can access them in
    # other places.
    graph_utils.add_dict_to_collection({
        "source_vocab_to_id": source_vocab_to_id,
        "source_id_to_vocab": source_id_to_vocab,
        "source_word_to_count": source_word_to_count,
        "source_candidate_vocab_to_id": source_candidate_vocab_to_id,
        "source_candidate_id_to_vocab": source_candidate_id_to_vocab,
        "source_candidate_word_to_count": source_candidate_word_to_count,
        "target_vocab_to_id": target_vocab_to_id,
        "target_id_to_vocab": target_id_to_vocab,
        "target_word_to_count": target_word_to_count
    }, "vocab_tables")

    # Slice source to max_len
    if self.params["source.max_seq_len"] is not None:
      features["source_tokens"] = features["source_tokens"][:, :self.params[
          "source.max_seq_len"]]
      features["source_len"] = tf.minimum(features["source_len"],
                                          self.params["source.max_seq_len"])
    # Slice source_candidate to max_len
    if self.params["source_candidate.max_seq_len"] is not None:
        features["source_candidate_tokens"] = features["source_candidate_tokens"][:, :self.params[
            "source_candidate.max_seq_len"]]
        features["source_candidate_len"] = tf.minimum(features["source_candidate_len"],
                                            self.params["source_candidate.max_seq_len"])

    # Look up the source ids in the vocabulary
    features["source_ids"] = source_vocab_to_id.lookup(features[
        "source_tokens"])
    features["source_candidate_ids"] = source_candidate_vocab_to_id.lookup(features[
                                                           "source_candidate_tokens"])
    # Maybe reverse the source
    if self.params["source.reverse"] is True:
      features["source_ids"] = tf.reverse_sequence(
          input=features["source_ids"],
          seq_lengths=features["source_len"],
          seq_dim=1,
          batch_dim=0,
          name=None)
      features["source_candidate_ids"] = tf.reverse_sequence(
          input=features["source_candidate_ids"],
          seq_lengths=features["source_candidate_len"],
          seq_dim=1,
          batch_dim=0,
          name=None)

    features["source_len"] = tf.to_int32(features["source_len"])
    tf.summary.histogram("source_len", tf.to_float(features["source_len"]))
    features["source_candidate_len"] = tf.to_int32(features["source_candidate_len"])
    tf.summary.histogram("source_candidate_len", tf.to_float(features["source_candidate_len"]))

    if labels is None:
      return features, None

    labels = labels.copy()

    # Slices targets to max length
    if self.params["target.max_seq_len"] is not None:
      labels["target_tokens"] = labels["target_tokens"][:, :self.params[
          "target.max_seq_len"]]
      labels["target_len"] = tf.minimum(labels["target_len"],
                                        self.params["target.max_seq_len"])

    # Look up the target ids in the vocabulary
    labels["target_ids"] = target_vocab_to_id.lookup(labels["target_tokens"])

    labels["target_len"] = tf.to_int32(labels["target_len"])
    tf.summary.histogram("target_len", tf.to_float(labels["target_len"]))

    # Keep track of the number of processed tokens
    num_tokens = tf.reduce_sum(labels["target_len"])
    num_tokens += tf.reduce_sum(features["source_len"])
    num_tokens += tf.reduce_sum(features["source_candidate_len"])
    token_counter_var = tf.Variable(0, "tokens_counter")
    total_tokens = tf.assign_add(token_counter_var, num_tokens)
    tf.summary.scalar("num_tokens", total_tokens)

    with tf.control_dependencies([total_tokens]):
      features["source_tokens"] = tf.identity(features["source_tokens"])
      features["source_candidate_tokens"] = tf.identity(features["source_candidate_tokens"])

    # Add to graph collection for later use
    graph_utils.add_dict_to_collection(features, "features")
    if labels:
      graph_utils.add_dict_to_collection(labels, "labels")

    print("attention_biseqseq features:{} labels:{}".format(features, labels))
    return features, labels
