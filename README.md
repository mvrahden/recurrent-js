# RECURRENT-js
[![js-google-style](https://img.shields.io/badge/code%20style-google-blue.svg)](https://google.github.io/styleguide/jsguide.html)

**RECURRENT-js** is an object-oriented Javascript library (baked with [Typescript](https://github.com/Microsoft/TypeScript)). It is a port of _Andrej Karpathy's_ RecurrentJS library that implements the following:

* Deep **Recurrent Neural Networks** (RNN) 
* **Long Short-Term Memory networks** (LSTM) 
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
  * **R** - Collection of Utility functions
  * **Mat** - Sophisticated Matrix Structure for Weights in Networks.
  * **RandMat** - `Mat` populated with random gaussian distributed values
  * **Graph** - Graph holding the Operations
  * **NNModel** - Genralized Class containing the Weights (and `Graph`) for `RNN` and `LSTM` 
  * **PreviousOutputs** - Standardized Interface for parameter injection in forward-pass of `NNModel`s.

* Network Classes:
  * **Net** - Simple Neural Network
  * **RNN** - Recurrent Neural Network. Extends `NNModel`.
  * **LSTM** - Long Short Term Memory Network. Extends `NNModel`.

These classes can be directly imported from this `npm` module, e.g.:
```typescript
import { Graph, Net } from 'recurrent-js';
```

For JavaScript usage `require` classes from this `npm` module as follows:
```javascript
const Graph = require('recurrent-js').Graph;
const Net = require('recurrent-js').Net;
```

#### Further Info for Production Usage

The transpiled Javascript-target is `ES6`, with a `CommonJS` module format.

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
