import numpy as np
import cupy as cp

import random
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from model import *
from updater import *
from iterator import *

################### Configuration #####################
LARGE = False

if LARGE:
  TRAIN_SIZE = 180000
  VALID_SIZE = 200
  MAX_STRING_LEN = 25
  NUM_ENCODE_LAYERS = 2
  NUM_DECODE_LAYERS = 2
  NUM_HIDDEN = 128
  BATCH_SIZE = 128
else:
  TRAIN_SIZE = 3000
  VALID_SIZE = 200
  MAX_STRING_LEN = 15
  NUM_ENCODE_LAYERS = 1
  NUM_DECODE_LAYERS = 1
  NUM_HIDDEN = 64
  BATCH_SIZE = 64

NUM_EPOCHS = 20
ATTEND = True
DROPOUT = 0.5
DEVICE = -1

GRADIENT_CLIP = 5
LOG = "./log"

word_id_map = {"a" : 0, "b" : 1, "c" : 2, "d" : 3, "<eos>" : 4, "<pad>" : 5}
VOCAB_SIZE = len(word_id_map)

np.random.seed(234)
random.seed(543)
####################################################


################## Sample data #####################

def sample_data(min_length, max_length):
  x = np.random.randint(0, VOCAB_SIZE - 2, size=(random.randint(min_length,
      max_length),)).tolist()
  y = x[::-1]
  return x, y

train_set = [sample_data(1, MAX_STRING_LEN) for _ in xrange(TRAIN_SIZE)]
val_set = [sample_data(1, MAX_STRING_LEN) for _ in xrange(VALID_SIZE)]


##################### MAIN ########################

def compute_loss(result, size):
  result["train_loss"] = result["main/loss"]
  result["train_acc"] = result["main/accuracy"]
  if "validation/main/loss" in result:
    result["val_loss"] = result["validation/main/loss"]
  if "validation/main/accuracy" in result:
    result["val_acc"] = result["validation/main/accuracy"]

if ATTEND:
  attention_model = GlobalAttention(NUM_HIDDEN)
  model = AttentionalEncoderDecoder(attention_model, NUM_ENCODE_LAYERS,
      NUM_DECODE_LAYERS, NUM_HIDDEN, VOCAB_SIZE, DROPOUT)
else:
  model = EncoderDecoder(NUM_ENCODE_LAYERS, NUM_DECODE_LAYERS, NUM_HIDDEN,
      VOCAB_SIZE, DROPOUT)
model = Seq2SeqClassifier(model)

# Set up for using GPU. Very easy!
xp = np
if DEVICE >= 0:
  chainer.cuda.get_device(DEVICE).use()
  model.to_gpu()
  xp = cp

# Optimizer.
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(GRADIENT_CLIP))

# Data iterators.
train_iter = Seq2SeqIterator(xp, train_set, BATCH_SIZE, word_id_map["<eos>"],
    word_id_map["<pad>"], shuffle=False, repeat=True)
val_iter = Seq2SeqIterator(xp, val_set, BATCH_SIZE, word_id_map["<eos>"],
    word_id_map["<pad>"], shuffle=False, repeat=False)

# Updater (training loop).
updater = Seq2SeqUpdater(train_iter, optimizer, DEVICE)

# Trainer
trainer = training.Trainer(updater, stop_trigger=(NUM_EPOCHS, "epoch"), out=LOG)

# Attach monitoring and evaluating extensions.
trainer.extend(extensions.Evaluator(val_iter, model, device=DEVICE,
    eval_func=lambda (x_batch, y_batch) : model(x_batch, y_batch, train=False)))
interval = 1
trainer.extend(extensions.LogReport(
    postprocess=lambda result: compute_loss(result, TRAIN_SIZE),
    trigger=(interval, "epoch")))
trainer.extend(extensions.PrintReport(
    ["epoch", "iteration", "train_loss", "train_acc", "val_loss", "val_acc"]),
    trigger=(interval, "epoch"))
trainer.extend(extensions.ProgressBar(update_interval=1))

trainer.extend(extensions.snapshot(trigger=(interval, "epoch")))
trainer.extend(
    extensions.snapshot_object(model, "model_epoch_{.updater.epoch}"))

trainer.run()
















