# seq2seq-chainer

Implementation of recurrent neural network (RNN) and seq2seq models in [Chainer](http://docs.chainer.org/en/stable/install.html). 

The toy task is learning to reverse a string (i.e. given input "abcde", output "edcba"). Implemented models are:
+ Vanilla multi-layer LSTM RNN model. 
+ Vanilla encoder-decoder model. 
+ Global-attentional encoder-decoder model ([Vinyals et al.](https://arxiv.org/pdf/1412.7449v3.pdf))

Run the program:
~~~~
$ python main.py
~~~~

To run with different modes, modify `main.py`. Some notable variables are:
+ DEVICE: the code is set to run on 1 GPU (DEVICE = 0), set DEVICE = -1 to run on CPU. 
+ LARGE: size of the data set. 
+ ATTEND: whether to use attention or not. 
