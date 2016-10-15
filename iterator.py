import numpy as np
import cupy as cp
import random
import chainer

class Seq2SeqIterator(chainer.dataset.Iterator):
  def __init__(self, xp, dataset, batch_size, eos, pad, shuffle=True, repeat=True):
    self.xp = xp
    self.dataset = dataset
    self.size = len(dataset)
    self.batch_size = batch_size
    assert batch_size <= self.size
    self.eos = eos
    self.pad = pad
    self.shuffle = shuffle
    self.repeat = repeat

    self.epoch = 0
    self.is_new_epoch = False
    self.iteration = 0
    self.offset = 0

  def __next__(self):
    self.is_new_epoch = (self.offset == 0)
    if self.is_new_epoch:
      self.epoch += 1
      if self.shuffle:
        random.shuffle(self.dataset)
      if not self.repeat and self.epoch > 1:
        raise StopIteration

    next_offset = min(self.size, self.offset + self.batch_size)
    batch = self.dataset[self.offset : next_offset]
    assert len(batch) > 0
    assert len(batch) == self.batch_size or (next_offset == self.size and
        len(batch) == self.size - self.offset)
    self.offset = next_offset if next_offset < self.size else 0

    # Padding
    max_x_length = max([len(pair[0]) for pair in batch])
    max_y_length = max([len(pair[1]) for pair in batch])

    x_batch = []
    y_batch = []
    for x, y in batch:
      x_batch.append(x + [self.eos] + [self.pad] * (max_x_length - len(x)))
      y_batch.append(y + [self.eos] + [self.pad] * (max_y_length - len(y)))
    x_batch = self.xp.array(x_batch, dtype=np.int32)
    y_batch = self.xp.array(y_batch, dtype=np.int32)
    return x_batch, y_batch

  @property
  def epoch_detail(self):
    return self.epoch + (self.offset * 1.0 / self.size)

  def serialize(self, serializer):
    self.iteration = serializer("iteration", self.iteration)
    self.epoch = serializer("self.epoch", self.epoch)
    self.offset = serializer("self.offset", self.offset)
