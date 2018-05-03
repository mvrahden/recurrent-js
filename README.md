# recurrent-js
[![Build Status](https://travis-ci.org/mvrahden/recurrent-js.svg?branch=master)](https://travis-ci.org/mvrahden/recurrent-js)
[![Build status](https://ci.appveyor.com/api/projects/status/1gbi2lkll4d48cy6/branch/master?svg=true)](https://ci.appveyor.com/project/mvrahden/recurrent-js/branch/master)
[![js-google-style](https://img.shields.io/badge/code%20style-google-blue.svg)](https://google.github.io/styleguide/jsguide.html)
[![dependency-free](https://img.shields.io/badge/dependencies-none-brightgreen.svg)]()

**The recurrent-js library** &ndash; Amazingly simple to build and train various neural networks. The library is an object-oriented neural network approach (baked with [Typescript](https://github.com/Microsoft/TypeScript)), containing stateless and stateful neural network architectures. It is a redesigned and extended version of _Andrej Karpathy's_ RecurrentJS library that implements the following:

* Vanilla Feedforward Neural Network (Net)
* Deep **Recurrent Neural Networks** (RNN)
* Deep **Long Short-Term Memory** Networks (LSTM) 
* **Bonus #1**: Deep **Feedforward Neural Networks** (DNN)
* **Bonus #2**: Deep **Bayesian Neural Networks** (BNN)
* In fact, the library is more general because it has functionality to construct arbitrary **expression graphs** over which the library can perform **automatic differentiation** similar to what you may find in Theano for Python, or in Torch etc. Currently, the code uses this very general functionality to implement RNN/LSTM, but one can build arbitrary Neural Networks and do automatic backprop.

## For Production Use

### What does the Library offer?

Currently exposed Classes:

* Utility Classes:
  * **Utils** - Collection of Utility functions.
  * **Graph** - Graph holding matrix operation sequences for backpropagation.
  * **Mat** - Sophisticated matrix structure for Weights in Networks.
  * **RandMat** - `Mat` populated with random gaussian distributed values.
  * **NetOpts** - Standardized `Interface` for the initial configuration of all Neural Networks.
  <!-- * **FNNModel** - Genralized Class containing the Weights (and `Graph`) for stateless `FNN`-models
    * such as `DNN` or `BNN`.
  * **RNNModel** - Genralized Class containing the Weights (and `Graph`) for stateful `RNN`-models
    * such as `RNN` or `LSTM`. -->
  * **InnerState** - Standardized `Interface` for parameter injection in forward-pass of stateful networks holding the previous state.

* Neural Network Classes:
  * stateless:
    * **Net** - Simple Neural Network.
    * **DNN** - Deep Feedforward Neural Network.
    * **BNN** - Deep Bayesian Neural Network.
  * stateful:
    * **RNN** - Recurrent Neural Network.
    * **LSTM** - Long Short Term Memory Network.

### How to install as dependency

Download available `@npm`: [recurrent-js](https://www.npmjs.com/package/recurrent-js)

Install via command line:
```
npm install --save recurrent-js@latest
```

The project directly ships with the transpiled Javascript code.
For TypeScript development it also contains Map-files and Declaration-files.

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

Training of neural networks is achieved by iteratively reinforcing wanted activation behavior or by suppressing the unwanted activation paths through adjusting the slopes of activation paths.
The training is achieved via an expression `Graph`, that holds the sequences of matrix operations being done during the forward-pass.
The results of the Matrix operations are contained in a `Mat`-Object, which contains the resulting values (`w`) and their automatically differentiated counterparts (`dw`).
By manipulating a derivative value of the resulting output-`Mat`, the `Graph`-object can then be used to propagate that gradient modification back into the neural network, via a gradient descent approach.
The so called backpropagation will then lead to supporting wanted neural net activity and suppressing unwanted activation behavior.
Backpropagation can be achieved as follows:

```typescript
import { Graph, DNN, Mat } from 'recurrent-js';

// define network structure
const netOpts = {
  inputSize: 3,
  hiddenUnits: [ 6, 2, 6 ],
  outputSize: 4
};

// instantiate network
const net = new DNN(netOpts);
// instantiate a graph with backprop-ability
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
since graph is keeping a reference of `decision`, it can now perform the backpropagation and therefore adjust the decisions gradient.
*/
graph.backward();

/*
5. OPTIONAL:
The loss manipulated gradient has now been propagated back into the network.
In addition you could also marginally discount all existing weights with a gradient on a global scale (meaning: throughout the whole net) as follows:
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
