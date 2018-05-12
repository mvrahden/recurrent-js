# recurrent-js
[![Build Status](https://travis-ci.org/mvrahden/recurrent-js.svg?branch=master)](https://travis-ci.org/mvrahden/recurrent-js)
[![Build status](https://ci.appveyor.com/api/projects/status/7qkcof8t6b0io44f/branch/master?svg=true)](https://ci.appveyor.com/project/mvrahden/recurrent-js/branch/master)
[![js-google-style](https://img.shields.io/badge/code%20style-google-blue.svg)](https://google.github.io/styleguide/jsguide.html)
[![dependency-free](https://img.shields.io/badge/dependencies-none-brightgreen.svg)]()

[docs-utils]: https://github.com/mvrahden/recurrent-js/blob/master/docs/utils.md
[docs-mat]: https://github.com/mvrahden/recurrent-js/blob/master/docs/mat.md
[docs-graph]: https://github.com/mvrahden/recurrent-js/blob/master/docs/graph.md
[docs-net]: https://github.com/mvrahden/recurrent-js/blob/master/docs/fnn/net.md
[docs-dnn]: https://github.com/mvrahden/recurrent-js/blob/master/docs/fnn/dnn.md
[docs-bnn]: https://github.com/mvrahden/recurrent-js/blob/master/docs/fnn/bnn.md
[docs-rnn]: https://github.com/mvrahden/recurrent-js/blob/master/docs/rnn/rnn.md
[docs-lstm]: https://github.com/mvrahden/recurrent-js/blob/master/docs/rnn/lstm.md

**The recurrent-js library** &ndash; Amazingly simple to build and train various neural networks. The library is an object-oriented neural network approach (baked with [Typescript](https://github.com/Microsoft/TypeScript)), containing stateless and stateful neural network architectures. It is a redesigned and extended version of _Andrej Karpathy's_ RecurrentJS library that implements the following:

* Vanilla Feedforward Neural Network (Net)
* Deep **Recurrent Neural Networks** (RNN)
* Deep **Long Short-Term Memory** Networks (LSTM) 
* **Bonus #1**: Deep **Feedforward Neural Networks** (DNN)
* **Bonus #2**: Deep **Bayesian Neural Networks** (BNN)
* In fact, the library is more general because it has functionality to construct arbitrary **expression graphs** over which the library can perform **automatic differentiation** similar to what you may find in Theano for Python, or in Torch etc. Currently, the code uses this very general functionality to implement RNN/LSTM, but one can build arbitrary Neural Networks and do automatic backprop.

## For Production Use

### What does the Library has to offer?

The following sections show the currently exposed Classes and Interfaces:

#### Utility Classes:

* **[Utils][docs-utils]** - Collection of Utility functions.
* **[Mat][docs-mat]** - Matrix Class with matrix operations for the neural networks.
* **RandMat** - A convenient subclass of `Mat`. `RandMat` objects are automatically populated with random values on their creation.
* **[Graph][docs-graph]** - Graph memorizing the sequences of matrix operations for backpropagation.
* **NetOpts** - Standardized `Interface` for the initial configuration of all Neural Networks.
<!-- * **FNNModel** - Generalized Class containing the Weights (and `Graph`) for stateless `FNN`-models
  * such as `DNN` or `BNN`.
* **RNNModel** - Generalized Class containing the Weights (and `Graph`) for stateful `RNN`-models
  * such as `RNN` or `LSTM`. -->
* **InnerState** - Standardized `Interface` for stateful networks memorizing the previous state of activations.

#### Neural Network Classes:
* stateless:
  * **[Net][docs-net]** - Vanilla Feedforward Neural Network.
  * **[DNN][docs-dnn]** - Deep Feedforward Neural Network.
  * **[BNN][docs-bnn]** - Deep Bayesian Neural Network.
* stateful:
  * **[RNN][docs-rnn]** - Recurrent Neural Network.
  * **[LSTM][docs-lstm]** - Long Short Term Memory Network.

### How to install as dependency

Download available `@npm`: [recurrent-js](https://www.npmjs.com/package/recurrent-js)

Install via command line:
```
npm install --save recurrent-js@latest
```

The project directly ships with the transpiled Javascript code.
For TypeScript development it also contains Map-files and Declaration-files.

### How to import?

The aforementioned classes can be imported from this `npm` module, e.g.:
```typescript
import { NetOpts, DNN } from 'recurrent-js';
```

For JavaScript usage `require` classes from this `npm` module as follows:
```javascript
// NetOpts is an interface (Typescript only), but it gives clues about the required Object-properties (keys)
const DNN = require('recurrent-js').DNN;
```

### How to train?

Training of neural networks is achieved by iteratively reinforcing wanted activation behavior or by suppressing the unwanted activation paths through adjusting the slopes of these activation paths.
The training is achieved via an expression `Graph`, that memorizes the sequence of matrix operations being executed during the forward-pass.
The results of the Matrix operations are contained in `Mat`-objects, which contains the resulting values (`w`).
By manipulating a derivative value of the resulting output-`Mat`, the `Graph`-object can then be used to calculate the resulting gradient and propagate that modified gradient back into the memorized order of matrix operations.
The so called backpropagation will then lead to supporting wanted neural network activity and suppressing unwanted activation behavior.
The described backpropagation can be achieved as follows:

```typescript
import { Graph, DNN, Mat } from 'recurrent-js';

/* define network structure configuration */
const netOpts = {
  inputSize: 3,
  hiddenUnits: [ 6, 2, 6 ],
  outputSize: 4
};

/* instantiate network */
const net = new DNN(netOpts);
/* instantiate a graph with backprop-ability */
const graph = new Graph(true);

/*
1. Before forward pass:
Create a single row of type `Mat`, which holds an observation.
We will refer to this as the state.
Dimensions of state are according to the configuration of the net (rows = inputSize = 3, cols = 1).
*/
const observation = [1, 0, 1]; /* an observation (ideally normalized) */
const state = new Mat(3, 1);
state.setFrom(observation);

/*
2. Decision making:
Forward pass with observed state of type `Mat` and graph.
The resulting decision is of type `Mat` and is holding multiple output values (here: 4).
*/
const decision = net.forward(state, graph);

/* 
3. After forward pass:
Compute the decision errors.
We refer to this as the loss value(s).
Inject that loss value into the derivative of your targeted value.
Here you could also apply e.g. loss clipping before injecting the value.
NOTE: You can also apply multiple loss values, to the respective array fields.
*/
decision.dw[1] = /* some value e.g. 0.5 */;

/*
4. After injecting the loss value:
since graph is keeping a reference of `decision`, it can now perform the backpropagation and therefore calculate a new decision gradient.
*/
graph.backward();

/*
5. After determining a new decision gradient:
The gradient determined with the loss value has now been calculated.
To propagate the slope of the new gradient back into the network and therefore adjust the actual decision gradient, the weights need to be updated accordingly.
With the injected `alpha`-value you can control the degree of the weight update.
The underlying formula is as follows: 
w[i] = w[i] - (dw[i] * alpha)
*/
net.update(0.01);

/* REPEAT numbers 1 to 5 till the loss value(s) reach a certain threshold */
```

The training is somehow identical for `Net`, `DNN`, `BNN`, `RNN`, `LSTM`.
For timely unfolding `RNN` and `LSTM` networks, keep in mind the statefulness of those approximators.

Should you want to get some deeper insights on "how to train the network", it is recommendable to have a look into the source of the DQN-Solver from the [reinforce-js](https://github.com/mvrahden/reinforce-js) library (`learnFromSarsaTuple`-Method).

## Example Applications

This project is an integral part of the `reinforce-js` library.
As such it is vividly demonstrated in the `learning-agents` model.

- [learning-agents](https://mvrahden.github.io/learning-agents) (GitHub Page)
- [reinforce-js](https://github.com/mvrahden/reinforce-js) (GitHub Repository)


## Community Contribution

Everybody is more than welcome to contribute and extend the functionality!

Please feel free to contribute to this project as much as you wish to.

1. clone from GitHub via `git clone https://github.com/mvrahden/treasurer.git`
2. `cd` into the directory and `npm install` for initialization
3. Try to `npm run test`. If everything is green, you're ready to go :sunglasses:

Before triggering a pull-request, please make sure that you've run all the tests via the *testing command*:

```
npm run test
```

This project relies on Visual Studio Codes built-in Typescript linting facilities. It primarily follows the [Google TypeScript Style-Guide](https://github.com/google/ts-style) through the provided *tslint-google.json* configuration file.

## License

As of License-File: [MIT](LICENSE)
