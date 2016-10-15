# seq2seq-chainer

Implementation of recurrent neural network (RNN) and seq2seq models in [Chainer](http://docs.chainer.org/en/stable/install.html). This repo is inspired by [Tal Baumel's cnn (now dynet) seq2seq notebook](https://talbaumel.github.io/attention/).

The toy task is learning to reverse a string (i.e. given input "abcde", output "edcba"). Implemented models are:
+ Vanilla multi-layer LSTM RNN model. 
+ Vanilla encoder-decoder model. 
+ Global-attentional encoder-decoder model ([Vinyals et al.](https://arxiv.org/pdf/1412.7449v3.pdf))

To run the code, please install **Chainer** and **CuDNN** first. Then evoke `main.py`:
~~~~
$ python main.py
~~~~

To run with different modes, modify `main.py`. Some notable variables are:
+ DEVICE: the code is set to run on CPU (DEVICE = -1), set DEVICE = 0 to run on single GPU. 
+ LARGE: size of the data set. 
+ ATTEND: whether to use attention or not. 

If there are any problems, email me at nguyenxuankhanhm@gmail.com
