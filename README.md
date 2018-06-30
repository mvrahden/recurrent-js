# recurrent-js
[![Build Status](https://travis-ci.org/mvrahden/recurrent-js.svg?branch=master)](https://travis-ci.org/mvrahden/recurrent-js)
[![Build status](https://ci.appveyor.com/api/projects/status/7qkcof8t6b0io44f/branch/master?svg=true)](https://ci.appveyor.com/project/mvrahden/recurrent-js/branch/master)
[![js-google-style](https://img.shields.io/badge/code%20style-google-blue.svg)](https://google.github.io/styleguide/jsguide.html)
[![dependency-free](https://img.shields.io/badge/dependencies-none-brightgreen.svg)]()

[docs-utils]: https://github.com/mvrahden/recurrent-js/blob/master/docs/utils.md
[docs-mat]: https://github.com/mvrahden/recurrent-js/blob/master/docs/mat.md
[docs-mat-opts]: https://github.com/mvrahden/recurrent-js/blob/master/docs/mat-opts.md
[docs-graph]: https://github.com/mvrahden/recurrent-js/blob/master/docs/graph.md
[docs-net]: https://github.com/mvrahden/recurrent-js/blob/master/docs/fnn/net.md
[docs-dnn]: https://github.com/mvrahden/recurrent-js/blob/master/docs/fnn/dnn.md
[docs-bnn]: https://github.com/mvrahden/recurrent-js/blob/master/docs/fnn/bnn.md
[docs-rnn]: https://github.com/mvrahden/recurrent-js/blob/master/docs/rnn/rnn.md
[docs-lstm]: https://github.com/mvrahden/recurrent-js/blob/master/docs/rnn/lstm.md

**Call For Volunteers:** Due to my lack of time, I'm desperately looking for voluntary help. Should you be interested in the training of neural networks (even though you're a newbie) and willing to develop this educational project a little further, please contact me :) There are some points on the agenda, that I'd still like to see implemented to make this project a nice library for abstract educational purposes.

> INACTIVE: Due to lack of time and help

**The recurrent-js library** &ndash; Various amazingly simple to build and train neural network architectures. This Library is for **educational purposes** only. The library is an object-oriented neural network approach (baked with [Typescript](https://github.com/Microsoft/TypeScript)), containing stateless and stateful neural network architectures. It is a redesigned and extended version of _Andrej Karpathy's_ RecurrentJS library that implements the following:

* Vanilla Feedforward Neural Network (Net)
* Deep **Recurrent Neural Networks** (RNN)
* Deep **Long Short-Term Memory** Networks (LSTM) 
* **Bonus #1**: Deep **Feedforward Neural Networks** (DNN)
* **Bonus #2**: Deep **Bayesian Neural Networks** (BNN)
* In fact, the library is more general because it has functionality to construct arbitrary **expression graphs** over which the library can perform **automatic differentiation** similar to what you may find in Theano for Python, or in Torch etc. Currently, the code uses this very general functionality to implement RNN/LSTM, but one can build arbitrary Neural Networks and do automatic backprop.

## For Production Use

### What does the Library has to offer?

The following sections provide an overview of the available Classes and Interfaces.
The class names are linked to more detailed descriptions of the specific classes.

#### Utility Classes:

* **[Utils][docs-utils]** - Collection of Utility functions: Array creation & manipulation, Statistical evaluation methods etc.
* **[Mat][docs-mat]** - Matrix Class holding weights and their derivatives for the neural networks.
* **RandMat** - A convenient subclass of `Mat`. `RandMat` objects are automatically populated with random values on their creation.
* **[MatOps][docs-mat-opts]** - Class with matrix operations (add, multiply, sigmoid etc.) and their respective derivative functions.
* **[Graph][docs-graph]** - Graph memorizing the sequences of matrix operations and matching their respective derivative functions for backpropagation.
* **NetOpts** - Standardized `Interface` for the initial configuration of all Neural Networks.
<!-- * **FNNModel** - Generalized Class containing the Weights (and `Graph`) for stateless `FNN`-models
  * such as `DNN` or `BNN`.
* **RNNModel** - Generalized Class containing the Weights (and `Graph`) for stateful `RNN`-models
  * such as `RNN` or `LSTM`. -->
* **InnerState** - Standardized `Interface` for stateful networks memorizing the previous state of activations.

#### Neural Network Classes:
* stateless:
  * **[Net][docs-net]** - shallow Vanilla Feedforward Neural Network (1 hidden layer).
  * **[DNN][docs-dnn]** - Deep Feedforward Neural Network.
  * **[BNN][docs-bnn]** - Deep Bayesian Neural Network.
* stateful (*Still old API!*):
  * **[RNN][docs-rnn]** - Deep Recurrent Neural Network.
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

Training of neural networks is achieved by iteratively reinforcing wanted neural activations or by suppressing unwanted activation paths through adjusting their respective slopes.
The training is achieved via an expression `Graph`, which memorizes the sequence of matrix operations being executed during the forward-pass operation of a neural network.
The results of the Matrix operations are contained in `Mat`-objects, which contain the resulting values (`w`) and their corresponding derivatives (`dw`).
The `Graph`-object can be used to calculate the resulting gradient and propagate a loss value back into the memorized sequence of matrix operations.
The update of the weights of the neural connections will then lead to supporting wanted neural network activity and suppressing unwanted activation behavior.
The described backpropagation can be achieved as follows:

```typescript
import { Graph, DNN } from 'recurrent-js';

/* define network structure configuration */
const netOpts = {
    architecture: { inputSize: 2, hiddenUnits: [2, 3], outputSize: 3 },
    training: { loss: 1e-11 }
  };

/* instantiate network */
const net = new DNN(netOpts);

/* make it trainable */
net.setTrainability(true);

/** 
 * Perform an iterative training by first forward passing an input
 * and second backward propagating the according target output.
 * You'll receive the squared loss, that gives you a hint of the networks
 * approximation quality.
 * Repeat this action until the quality of the output of the forward pass 
 * suits your needs, or the mean squared error is small enough, e.g. < 1.
 */
do {
  const someInput = [0, 1]; /* an array of intput values */
  const someExpectedOutput = [0, 1, 0]; /* an array of target output */

  const someOutput = net.forward(someInput);
  
  net.backward(someExpectedOutput /* , alpha?: number */);
  const squaredLoss = net.getSquaredLoss(someInput, someExpectedOutput);
} while(squaredLoss > 0.1);
/**
 * --> Keep in mind: you actually want a low MEAN squaredLoss, this is
 * left out in this example, to keep the focus on the important parts
 */

```
**HINT #1**: providing an additional *custom learning rate* (`alpha`) for the backpropagation can accelerate the training. For further info please consult the respective`test-examples.spec.ts` file.

**HINT #2**: The *Recurrent Neural Network Architectures* (RNN, LSTM) are not yet updated to this new training API. Due to my current lack of time, this likely won't change for a while... (unless this repo gets some voluntary help). Please consult the README of the [commit v.1.6.2](https://github.com/mvrahden/recurrent-js/tree/4065e644a36a26ae31598070dd0197008fe1a88b) for the details of the former training style. Thanks!

Should you want to get some deeper insights on "how to train the network", it is recommendable to have a look into the source of the DQN-Solver from the [reinforce-js](https://github.com/mvrahden/reinforce-js) library (`learnFromSarsaTuple`-Method).

## Example Applications

This project is an integral part of the `reinforce-js` library.
As such it is vividly demonstrated in the `learning-agents` model.

- [learning-agents](https://mvrahden.github.io/learning-agents) (GitHub Page)
- [reinforce-js](https://github.com/mvrahden/reinforce-js) (GitHub Repository)


## Community Contribution

Everybody is more than welcome to contribute and extend the functionality!

Please feel free to contribute to this project as much as you wish to.

1. clone from GitHub via `git clone https://github.com/mvrahden/recurrent-js.git`
2. `cd` into the directory and `npm install` for initialization
3. Try to `npm run test`. If everything is green, you're ready to go :sunglasses:

Before triggering a pull-request, please make sure that you've run all the tests via the *testing command*:

```
npm run test
```

This project relies on Visual Studio Codes built-in Typescript linting facilities. It primarily follows the [Google TypeScript Style-Guide](https://github.com/google/ts-style) through the provided *tslint-google.json* configuration file.

## License

As of License-File: [MIT](LICENSE)
