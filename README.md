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

For further Information see the [recurrentjs](https://github.com/karpathy/recurrentjs) repository.

## For Production Use

### How to install as dependency

Download available `@npm`: [recurrent-js](https://www.npmjs.com/package/recurrent-js)

Install via command line:
```
npm install --save recurrent-js
```

### How To use the Library in Production

Currently exposed Classes:

* Utility Classes:
  * **Utils** - Collection of Utility functions.
  * **Mat** - Sophisticated Matrix Structure for Weights in Networks.
  * **RandMat** - `Mat` populated with random gaussian distributed values.
  * **Graph** - Graph holding the Operation sequence for backpropagation.
  * **FNNModel** - Genralized Class containing the Weights (and `Graph`) for `FNN`-models, such as `DNN` or `BNN`.
  * **RNNModel** - Genralized Class containing the Weights (and `Graph`) for `RNN`-models, such as a `RNN` or `LSTM`.
  * **InnerState** - Standardized Interface for parameter injection in forward-pass of `RNNModel`s holding the previous state.

* Neural Network Classes:
  * **Net** - Simple Neural Network.
  * **DNN** - Deep Feedfoward Neural Network. Extends `FNNModel`.
  * **BNN** - Deep Bayesian Neural Network. Extends `FNNModel`.
  * **RNN** - Recurrent Neural Network. Extends `RNNModel`.
  * **LSTM** - Long Short Term Memory Network. Extends `RNNModel`.

These classes can be imported from this `npm` module, e.g.:
```typescript
import { Graph, Net } from 'recurrent-js';
```

For JavaScript usage `require` classes from this `npm` module as follows:
```javascript
const Graph = require('recurrent-js').Graph;
const Net = require('recurrent-js').Net;
```

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
