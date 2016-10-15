import chainer
import chainer.functions as F
import chainer.links as L

class MultiRNNCell(chainer.ChainList):
  def __init__(self, num_layers, num_hidden, dropout_ratio=0.5):
    self.num_layers = num_layers
    self.dropout_ratio = dropout_ratio

    layers = [L.LSTM(num_hidden, num_hidden) for _ in xrange(num_layers)]

    super(MultiRNNCell, self).__init__(*layers)

  def reset_state(self):
    for i in xrange(self.num_layers):
      self[i].reset_state()

  def __call__(self, x, train):
    output = x
    for i in xrange(self.num_layers):
      output = self[i](F.dropout(output, ratio=self.dropout_ratio, train=train))
    return output

