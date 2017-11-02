# -*- coding:utf-8 -*-
import os
import codecs
import shutil
from pydoc import locate
import tensorflow as tf
import numpy as np
from seq2seq import tasks, models, graph_utils
from seq2seq.training import utils as training_utils
from seq2seq.tasks.inference_task import InferenceTask, unbatch_dict

SEQUENCE_LEN_MAX = 50

class DecodeOnce(InferenceTask):
  '''
  Similar to tasks.DecodeText, but for one input only.
  Source fed via features.source_tokens and features.source_len
  '''
  def __init__(self, params, callback_func):
    super(DecodeOnce, self).__init__(params)
    self.callback_func=callback_func
  
  @staticmethod
  def default_params():
    return {}

  def before_run(self, _run_context):
    fetches = {}
    fetches["predicted_tokens"] = self._predictions["predicted_tokens"]
    fetches["logits"] = self._predictions["logits"]
    fetches["features.source_tokens"] = self._predictions["features.source_tokens"]
    fetches["features.source_candidate_tokens"] = self._predictions["features.source_candidate_tokens"]
    fetches["beam_parent_ids"] = self._predictions["beam_search_output.beam_parent_ids"]
    return tf.train.SessionRunArgs(fetches)

  def after_run(self, _run_context, run_values):
    fetches_batch = run_values.results
    for fetches in unbatch_dict(fetches_batch):
      # Convert to unicode
      fetches["predicted_tokens"] = np.char.decode(
          fetches["predicted_tokens"].astype("S"), "utf-8")
      predicted_tokens = fetches["predicted_tokens"]

      # If we're using beam search we take the first beam
      # TODO: beam search top k
      if np.ndim(predicted_tokens) > 2:
        predicted_tokens = predicted_tokens[:, 0]

      fetches["features.source_tokens"] = np.char.decode(
          fetches["features.source_tokens"].astype("S"), "utf-8")
      source_tokens = fetches["features.source_tokens"]
      source_candidate_tokens = fetches["features.source_candidate_tokens"]
      key = list(source_tokens) + ["#"] + [s.decode("UTF-8") for s in source_candidate_tokens]
      self.callback_func(key, predicted_tokens)



print("start to test")
# TODO: pass via args
# MODEL_DIR = "model_fin_qa"
MODEL_DIR = "model_fin_qa"
checkpoint_path = tf.train.latest_checkpoint(MODEL_DIR)
print("checkpoint:" + checkpoint_path)

# Load saved training options
train_options = training_utils.TrainOptions.load(MODEL_DIR)

# Create the model
model_cls = locate(train_options.model_class) or \
  getattr(models, train_options.model_class)
model_params = train_options.model_params
model_params.update({"inference.beam_search.beam_width": 3})

print("create model class")
model = model_cls(
    params=model_params,
    mode=tf.contrib.learn.ModeKeys.INFER)


# first dim is batch size
# source_query_tokens_ph = tf.placeholder(dtype=tf.string, shape=(1, None), name="source_tokens")
# source_query_len_ph = tf.placeholder(dtype=tf.int32, shape=(1,), name="source_len")
# source_candidate_tokens_ph = tf.placeholder(dtype=tf.string, shape=(1, None), name="source_candidate_tokens")
# source_candidate_len_ph = tf.placeholder(dtype=tf.int32, shape=(1,), name="source_candidate_len")
source_query_tokens_ph = tf.placeholder(dtype=tf.string, shape=(), name="source_tokens")
source_query_len_ph = tf.placeholder(dtype=tf.int32, shape=(), name="source_len")
source_candidate_tokens_ph = tf.placeholder(dtype=tf.string, shape=(), name="source_candidate_tokens")
source_candidate_len_ph = tf.placeholder(dtype=tf.int32, shape=(), name="source_candidate_len")

# source_query_tokens_input = tf.expand_dims(source_query_tokens_ph, 0)
# source_query_len_input = tf.expand_dims(source_query_len_ph, 0)
# source_candidate_tokens_input = tf.expand_dims(source_candidate_tokens_ph, 0)
# source_candidate_len_input = tf.expand_dims(source_candidate_len_ph,0)

# source_query_tokens_ph = tf.string_split(source_query_tokens_ph, " ")
# source_query_len_ph = tf.string_split(source_query_len_ph, " ")
# source_candidate_tokens_ph = tf.string_split(source_candidate_tokens_ph, " ")
# source_candidate_len_ph = tf.string_split(source_candidate_len_ph, " ")

source_query_tokens_input = tf.expand_dims(source_query_tokens_ph, 0)
source_query_len_input = tf.expand_dims(source_query_len_ph, 0)
source_candidate_tokens_input = tf.expand_dims(source_candidate_tokens_ph, 0)
source_candidate_len_input = tf.expand_dims(source_candidate_len_ph, 0)

source_query_tokens_input = tf.string_split(source_query_tokens_input)
source_query_tokens_input = tf.sparse_tensor_to_dense(source_query_tokens_input, default_value="")
# source_query_len_input = tf.string_split(source_query_len_input)
source_candidate_tokens_input = tf.string_split(source_candidate_tokens_input)
source_candidate_tokens_input = tf.sparse_tensor_to_dense(source_candidate_tokens_input, default_value="")
# source_candidate_len_input = tf.string_split(source_candidate_len_input)

model(
  features={
    # "source_tokens": source_query_tokens_ph,
    # "source_len": source_query_len_ph,
    # "source_candidate_tokens": source_candidate_tokens_ph,
    # "source_candidate_len": source_candidate_len_ph

    # "source_tokens": [source_query_tokens_ph],
    # "source_len": [source_query_len_ph],
    # "source_candidate_tokens": [source_candidate_tokens_ph],
    # "source_candidate_len": [source_candidate_len_ph]

    "source_tokens": source_query_tokens_input,
    "source_len": source_query_len_input,
    "source_candidate_tokens": source_candidate_tokens_input,
    "source_candidate_len": source_candidate_len_input
  },
  labels=None,
  params={
    "vocab_source_query": "dataset/fin_qa/train/vocab.sources.txt",
    "vocab_source_candidate": "dataset/fin_qa/train/vocab.sources_candidate.txt",
    "vocab_target": "dataset/fin_qa/train/vocab.targets.txt"
  }
)

saver = tf.train.Saver()
# _predictions = graph_utils.get_dict_from_collection("predictions")

def _session_init_op(_scaffold, sess):
  saver.restore(sess, checkpoint_path)
  tf.logging.info("Restored model from %s", checkpoint_path)

scaffold = tf.train.Scaffold(init_fn=_session_init_op)
session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold)


def _tokens_to_str(tokens):
  return " ".join(tokens).split("SEQUENCE_END")[0].strip()

# A hacky way to retrieve prediction result from the task hook...
prediction_dict = {}
def _save_prediction_to_dict(source_tokens, predicted_tokens):
  prediction_dict[_tokens_to_str(source_tokens)] = _tokens_to_str(predicted_tokens)

# print("create session")
# sess = tf.train.MonitoredSession(
#   session_creator=session_creator,
#   hooks=[DecodeOnce({}, callback_func=_save_prediction_to_dict)])

_predictions = graph_utils.get_dict_from_collection("predictions")
fetches = {}
fetches["predicted_tokens"] = _predictions["predicted_tokens"]
fetches["predicted_ids"] = _predictions["predicted_ids"]
fetches["features.source_tokens"] = _predictions["features.source_tokens"]
fetches["features.source_candidate_tokens"] = _predictions["features.source_candidate_tokens"]
#fetches["beam_parent_ids"] = _predictions["beam_search_output.beam_parent_ids"]
tf.train.SessionRunArgs(fetches)
sess = tf.Session()
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
saver.restore(sess, checkpoint_path)

print("start to export model:")
saved_model_path = "fin_biseq2seq_model"
if os.path.exists(saved_model_path):
  print("removing old saved_model:" + saved_model_path)
  shutil.rmtree(saved_model_path)
builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)
# Relevant change to the linked example is here!
builder.add_meta_graph_and_variables(sess, ["fin_biseq2seq"], legacy_init_op=tf.tables_initializer())
builder.save()
print("finish")

# The main API exposed
def query_once(source_tokens):
  tf.reset_default_graph()
  source_tokens = source_tokens.split() + ["SEQUENCE_END"]
  sess.run([], {
      source_query_tokens_ph: [source_tokens],
      source_query_len_ph: [len(source_tokens)],
      source_candidate_tokens_ph: [source_tokens],
      source_candidate_len_ph: [len(source_tokens)]
    })
  return prediction_dict.pop(_tokens_to_str(source_tokens))

def query_biseq2seq_once(source_query_tokens, source_candidate_tokens):
  tf.reset_default_graph()
  source_query_tokens_temp = source_query_tokens.split() + ["SEQUENCE_END"]
  source_query_len = len(source_query_tokens_temp)
  if SEQUENCE_LEN_MAX > source_query_len:
    source_query_tokens_temp += ["SEQUENCE_END"] * (SEQUENCE_LEN_MAX - source_query_len)
  source_query_tokens = ' '.join(source_query_tokens_temp)
  source_candidate_tokens_temp = source_candidate_tokens.split() + ["SEQUENCE_END"]
  source_candidate_len = len(source_candidate_tokens_temp)
  if SEQUENCE_LEN_MAX > source_candidate_len:
    source_candidate_tokens_temp += ["SEQUENCE_END"] * (SEQUENCE_LEN_MAX - source_candidate_len)
  source_candidate_tokens = ' '.join(source_candidate_tokens_temp)

  # source_query_tokens = source_query_tokens.split() + ["SEQUENCE_END"]
  # source_query_len = len(source_query_tokens)
  # source_candidate_tokens = source_candidate_tokens.split() + ["SEQUENCE_END"]
  # source_candidate_len = len(source_candidate_tokens)
  # slen = max(source_query_len, source_candidate_len)
  # if SEQUENCE_LEN_MAX > source_query_len:
  #   source_query_tokens += ["SEQUENCE_END"] * (SEQUENCE_LEN_MAX - source_query_len)
  # if SEQUENCE_LEN_MAX > source_candidate_len:
  #   source_candidate_tokens += ["SEQUENCE_END"] * (SEQUENCE_LEN_MAX - source_candidate_len)

  # source_query_tokens += "SEQUENCE_END"
  source_len = len(source_candidate_tokens.split())
  predictions, = sess.run([fetches], {
      # source_query_tokens_ph: [source_query_tokens],
      # source_query_len_ph: [slen],
      # source_candidate_tokens_ph: [source_candidate_tokens],
      # source_candidate_len_ph: [slen]
      source_query_tokens_ph: source_query_tokens,
      source_query_len_ph: source_len,
      source_candidate_tokens_ph: source_candidate_tokens,
      source_candidate_len_ph: source_len
    })
  # key = list(source_query_tokens) + ["#"] + list(source_candidate_tokens)
  # return prediction_dict.pop(_tokens_to_str(key))
  # return predictions
  print(predictions["predicted_tokens"])
  print("predicted_ids:")
  print(predictions["predicted_ids"])
  print("beam_parent_ids:")
  print(predictions["beam_parent_ids"])
  predicted_tokens = predictions["predicted_tokens"]
  if np.ndim(predicted_tokens) == 3:
    predicted_tokens = predicted_tokens[:, :, 0]
  elif np.ndim(predicted_tokens) == 2:
    predicted_tokens = predicted_tokens[:, 0]
  predicted_tokens = [b.decode("UTF-8") for b in predicted_tokens[0]]
  predicted = ' '.join(predicted_tokens).split("SEQUENCE_END")[0]
  predicted = ''.join(predicted.split())
  return predicted

def gather_tree(values, parents):
  """Gathers path through a tree backwards from the leave nodes. Used
  to reconstruct beams given their parents."""
  beam_length = values.shape[0]
  num_beams = values.shape[1]
  res = np.zeros_like(values)
  res[-1, :] = values[-1, :]
  for beam_id in range(num_beams):
    parent = parents[-1][beam_id]
    for level in reversed(range(beam_length - 1)):
      res[level, beam_id] = values[level][parent]
      parent = parents[level][parent]
  return np.array(res).astype(values.dtype)



