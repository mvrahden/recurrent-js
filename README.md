# recurrent-js
[![js-google-style](https://img.shields.io/badge/code%20style-google-blue.svg)](https://google.github.io/styleguide/jsguide.html)
[![dependency-free](https://img.shields.io/badge/dependencies-none-brightgreen.svg)]()

**The recurrent-js library** &ndash; Amazingly simple to build and train neural networks. The library is an object-oriented neural network approach (baked with [Typescript](https://github.com/Microsoft/TypeScript)). It is a redesigned and extended version of _Andrej Karpathy's_ RecurrentJS library that implements the following:

* Vanilla Feedforward Neural Network (Net)
* Deep **Recurrent Neural Networks** (RNN)
* Deep **Long Short-Term Memory** Networks (LSTM) 
* **Bonus #1**: Deep **Feedforward Neural Networks** (DNN)
* **Bonus #2**: Deep **Bayesian Neural Networks** (BNN)
* In fact, the library is more general because it has functionality to construct arbitrary **expression graphs** over which the library can perform **automatic differentiation** similar to what you may find in Theano for Python, or in Torch etc. Currently, the code uses this very general functionality to implement RNN/LSTM, but one can build arbitrary Neural Networks and do automatic backprop.

## For Production Use

### How to install as dependency

Download available `@npm`: [recurrent-js](https://www.npmjs.com/package/recurrent-js)

Install via command line:
```
npm install --save recurrent-js
```

### What does the Library offer?

Currently exposed Classes:

* Utility Classes:
  * **Utils** - Collection of Utility functions.
  * **Mat** - Sophisticated Matrix Structure for Weights in Networks.
  * **RandMat** - `Mat` populated with random gaussian distributed values.
  * **Graph** - Graph holding the Operation sequence for backpropagation.
  * **NetOpts** - Standardized `Interface` for the initial configuration of all Neural Networks.
  * **FNNModel** - Genralized Class containing the Weights (and `Graph`) for stateless `FNN`-models, such as `DNN` or `BNN`.
  * **RNNModel** - Genralized Class containing the Weights (and `Graph`) for statefull `RNN`-models, such as a `RNN` or `LSTM`.
  * **InnerState** - Standardized `Interface` for parameter injection in forward-pass of `RNNModel`s holding the previous state.

* Neural Network Classes:
  * **Net** - Simple Neural Network.
  * **DNN** - Deep Feedfoward Neural Network. Extends `FNNModel`.
  * **BNN** - Deep Bayesian Neural Network. Extends `FNNModel`.
  * **RNN** - Recurrent Neural Network. Extends `RNNModel`.
  * **LSTM** - Long Short Term Memory Network. Extends `RNNModel`.

### How to import?

These classes can be imported from this `npm` module, e.g.:
```typescript
import { NetOpts, DNN } from 'recurrent-js';
```

For JavaScript usage `require` classes from this `npm` module as follows:
```javascript
// NetOpts is a interface (Typescript only), but it gives clues about the required Object-keys
const DNN = require('recurrent-js').DNN;
```

### How to train?

Training of neural networks is achieved by iteratively reinforcing wanted activation behavior and by surpressing the unwanted.
The training is achieved via an expression `Graph`, that holds the sequences of matrix operations being done during the forwardpass.
This Graph can then be used to propagate a loss value back into the neural network, via a gradient descent approach.
The so called backpropagation will then lead to supporting wanted neural net activity and surpressing unwanted activation behavior.
Backpropagation can be achieved as follows:

```typescript
import { Graph, DNN } from 'recurrent-js';

// define network structure
const netOpts = {
  inputSize: 3,
  hiddenUnits: [6,2,6],
  output: 4
};

// instantiate network
const net = new DNN(netOpts);
// instantiate a graph with backprop-ability
const graph = new Graph(true);

/*
Create a single row `Mat` which holds an observation.
Dimensions of `Mat` are per configuration of the net (rows = inputSize = 3, cols = 1).
*/

// forward pass with observed state and graph
// result is a `Mat` holding multiple output values (here: 4)
const result = net.forward(/* inject some observed state */, graph);

// after forward pass: 
// inject a loss value into the derivative of your targeted value
// here you could also apply e.g. loss clipping before injecting the value
result.dw[1] = /* some value e.g. 0.5 */;


graph.backward(); // since graph is keeping a reference of `result`, it can now perform the backpropagation

/* The loss manipulated gradient has now been propagated back into the network.
In addition you could also marginally discount all existing weights with a gradient on a global scale (meaning: throughout the whole net) as follows:
*/
net.update(0.01);

/*  */
```

The training is identical for `Net`, `DNN`, `BNN`, `RNN`, `LSTM`.
For `RNN` and `LSTM` keep in mind the statefulness of those approximators.

Should you want to get some deeper insights on "how to train the network", it is recommendable to have a look into the source of the DQN-Solver from the [reinforce-js](https://github.com/mvrahden/reinforce-js) library (`learnFromSarsaTuple`-Method).

## Example Projects

This project is an integral part of the `reinforce-js` library.
As such it is vividly demonstrated in the `learning-agents` model.

- [learning-agents](https://mvrahden.github.io/learning-agents) (GitHub Page)
- [reinforce-js](https://github.com/mvrahden/reinforce-js) (GitHub Repository)

## For Contributors

1. `Clone` this project to a working directory.
2. `npm install` to setup the development dependencies.
3. To compile the codebase:

```
tsc -p .
```

This project relies on Visual Studio Codes built-in Typescript linting facilities. It primarily follows the [Google TypeScript Style-Guide](https://github.com/google/ts-style) through the provided *tslint-google.json* configuration file.

## License

As of License-File: [MIT](LICENSE)
