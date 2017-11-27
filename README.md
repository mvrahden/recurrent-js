# recurrent-ts

*recurrent-ts* is a Typescript port of the _Andrej Karpathy's_ RecurrentJS library that implements the following:

- Deep **Recurrent Neural Networks** (RNN) 
- **Long Short-Term Memory networks** (LSTM) 
- In fact, the library is more general because it has functionality to construct arbitrary **expression graphs** over which the library can perform **automatic differentiation** similar to what you may find in Theano for Python, or in Torch etc. Currently, the code uses this very general functionality to implement RNN/LSTM, but one can build arbitrary Neural Networks and do automatic backprop.

For further Information see the [recurrent-js](https://github.com/karpathy/recurrentjs) repository

# License

As of License-File: *MIT*
