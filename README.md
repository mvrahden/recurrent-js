# recurrent-ts

*recurrent-ts* is a Typescript port of the _Andrej Karpathy's_ RecurrentJS library that implements the following:

* Deep **Recurrent Neural Networks** (RNN) 
* **Long Short-Term Memory networks** (LSTM) 
* In fact, the library is more general because it has functionality to construct arbitrary **expression graphs** over which the library can perform **automatic differentiation** similar to what you may find in Theano for Python, or in Torch etc. Currently, the code uses this very general functionality to implement RNN/LSTM, but one can build arbitrary Neural Networks and do automatic backprop.

For further Information see the [recurrent-js](https://github.com/karpathy/recurrentjs) repository.

# Use as `npm`-Project Dependency

## To install as dependency:

```
npm install --save recurrent-ts
```

## To use the Library in Production:

Currently exposed Classes:

* R
* Mat
* RandMat
* Graph
* Net
* LSTM

These classes can be imported from this modules `index` file.

# Contributional Development

1. `Clone` this project to a working directory.
2. `npm install` to setup the development dependencies.
2. To compile the codebase:

```
tsc -p .
```

# License

As of License-File: *MIT*
