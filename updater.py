import chainer
from model import *

class Seq2SeqUpdater(chainer.training.StandardUpdater):
  def __init__(self, data_iter, optimizer, device):
    super(Seq2SeqUpdater, self).__init__(data_iter, optimizer, device=device)

  def update_core(self):
    data_iter = self.get_iterator("main")
    optimizer = self.get_optimizer("main")

    x_batch, y_batch = data_iter.__next__()
    loss = optimizer.target(x_batch, y_batch, train=True)
    optimizer.target.cleargrads()
    loss.backward()
    optimizer.update()

