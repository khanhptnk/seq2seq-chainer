import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from cell import MultiRNNCell

# Vanilla multi-layer LSTM RNN.
class SimpleRNN(chainer.Chain):
  def __init__(self, num_rnn_layers, num_hidden, vocab_size, dropout_ratio=0.5):
    super(SimpleRNN, self).__init__(
      embed = L.EmbedID(vocab_size, num_hidden),
      cell = MultiRNNCell(num_rnn_layers, num_hidden, dropout_ratio),
    )

  def __call__(self, xs, train):
    self.cell.reset_state()
    embeddings = self.embed(xs)
    return [self.cell(embeddings[:, i, :], train) for i in
        xrange(embeddings.shape[1])]

  def predict(self, xs, ys, train):
    outputs = self.__call__(xs, ys, train)
    return self.linear(F.concat(outputs, axis=0))


# Vanilla seq2seq model.
class EncoderDecoder(chainer.Chain):
  def __init__(self, num_encode_rnn_layers, num_decode_rnn_layers, num_hidden,
      vocab_size, dropout_ratio=0.5):
    super(EncoderDecoder, self).__init__(
      encoder = SimpleRNN(num_encode_rnn_layers, num_hidden, vocab_size,
          dropout_ratio),
      decoder_cell = MultiRNNCell(num_decode_rnn_layers, num_hidden,
          dropout_ratio),
      linear = L.Linear(num_hidden, vocab_size),
    )

  def __call__(self, xs, ys, train):
    encoder_hidden = self.encoder(xs, train)[-1]

    self.decoder_cell.reset_state()
    ys_embeddings = self.encoder.embed(ys)
    length = ys_embeddings.shape[1]
    outputs = []
    output = self.xp.zeros((xs.shape[0], self.decoder_cell[0].state_size),
        dtype=self.xp.float32)
    for i in xrange(length):
      if i > 0:
        decoder_inputs = encoder_hidden + ys_embeddings[:, i - 1, :]
      else:
        decoder_inputs = encoder_hidden
      output = self.decoder_cell(decoder_inputs, train)
      outputs.append(output)
    return outputs

  def predict(self, xs, ys, train):
    outputs = self.__call__(xs, ys, train)
    return self.linear(F.concat(outputs, axis=0))


# Global attention model from https://arxiv.org/pdf/1412.7449v3.pdf
class GlobalAttention(chainer.Chain):
  def __init__(self, num_hidden):
    super(GlobalAttention, self).__init__(
      w1 = L.Linear(num_hidden, num_hidden),
      w2 = L.Linear(num_hidden, num_hidden),
      v = L.Linear(num_hidden, 1),
    )
    self.encoder_hiddens = None
    self.w1hi = None

  def __call__(self, output_hidden, length):
    batch_size = output_hidden.shape[0]
    num_hidden = output_hidden.shape[1]

    w2dt = F.broadcast_to(self.w2(output_hidden),
        shape=(length, batch_size, num_hidden))
    w1hi_plus_w2dt = self.w1hi + w2dt
    w1hi_plus_w2dt = F.swapaxes(w1hi_plus_w2dt, 0, 1)
    w1hi_plus_w2dt = F.reshape(w1hi_plus_w2dt, shape=(batch_size * length, -1))

    logits = F.reshape(self.v(F.tanh(w1hi_plus_w2dt)), shape=(batch_size, -1))

    probs = F.broadcast_to(F.softmax(logits),
        shape=(num_hidden, batch_size, length))
    probs = F.swapaxes(probs, 0, 2)

    return F.sum(self.encoder_hiddens * probs, axis=0)

  def precompute(self, encoder_hiddens):
    length = len(encoder_hiddens)
    batch_size = encoder_hiddens[0].shape[0]
    self.encoder_hiddens = F.stack(encoder_hiddens)
    self.w1hi = F.reshape(
        self.w1(F.reshape(self.encoder_hiddens,
            shape=(length * batch_size, -1))),
        shape=(length, batch_size, -1))

# Attentional seq2seq model with customizeable attention model.
class AttentionalEncoderDecoder(EncoderDecoder):
  def __init__(self, attention_model, num_encode_rnn_layers,
      num_decode_rnn_layers, num_hidden, vocab_size, dropout_ratio=0.5):
    super(AttentionalEncoderDecoder, self).__init__(num_encode_rnn_layers,
        num_encode_rnn_layers, num_hidden, vocab_size, dropout_ratio)
    self.add_link("attention", attention_model)

  def __call__(self, xs, ys, train):
    self.attention.precompute(self.encoder(xs, train))

    self.decoder_cell.reset_state()
    ys_embeddings = self.encoder.embed(ys)
    length = ys_embeddings.shape[1]
    outputs = []
    output = self.xp.zeros((xs.shape[0], self.decoder_cell[0].state_size),
        dtype=self.xp.float32)
    for i in xrange(length):
      decoder_inputs = self.attention(output, length)
      if i > 0:
        decoder_inputs += ys_embeddings[:, i - 1, :]
      output = self.decoder_cell(decoder_inputs, train)
      outputs.append(output)
    return outputs

# Classifier (where loss and metrics are computed).
class Seq2SeqClassifier(chainer.Chain):
  def __init__(self, predictor):
    super(Seq2SeqClassifier, self).__init__(predictor=predictor)

  def __call__(self, xs, ys, train):
    decoder_logits = self.predictor.predict(xs, ys, train)
    labels = F.flatten(F.transpose(ys))
    loss = F.softmax_cross_entropy(decoder_logits, labels)
    accuracy = F.accuracy(decoder_logits, labels)
    reporter.report({"loss" : loss, "accuracy" : accuracy}, self)
    return loss











