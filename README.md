# RECURRENT-js

**RECURRENT-js** is an object-oriented Javascript library (baked with Typescript). It is a port of _Andrej Karpathy's_ RecurrentJS library that implements the following:

* Deep **Recurrent Neural Networks** (RNN) 
* **Long Short-Term Memory networks** (LSTM) 
* In fact, the library is more general because it has functionality to construct arbitrary **expression graphs** over which the library can perform **automatic differentiation** similar to what you may find in Theano for Python, or in Torch etc. Currently, the code uses this very general functionality to implement RNN/LSTM, but one can build arbitrary Neural Networks and do automatic backprop.

For further Information see the [recurrentjs](https://github.com/karpathy/recurrentjs) repository.

# For Production Use

## How to install as dependency

Download available `@npm`: [recurrent-js](https://www.npmjs.com/package/recurrent-js)

Install via command line:
```
npm install --save recurrent-js
```

## How To use the Library in Production

Currently exposed Classes:

* *R* - Collection of Utility functions
* *Mat* - Sophisticated Matrix Structure
* *RandMat* - `Mat` with populated with random gaussian distributed values
* *Graph* - Graph with Operations
* *Net*
* *LSTM*

These classes can be directly imported from this `npm` module, e.g.:
```typescript
import { Garph, Net } from 'recurrent-js';
```

## Further Info

The transpiled Javascript-target is `ES6`. The supported module format is `CommonJS`.

# For Contributors

1. `Clone` this project to a working directory.
2. `npm install` to setup the development dependencies.
3. To compile the codebase:

```
tsc -p .
```

This project relies on Visual Studio Codes built-in Typescript linting facilities. It primarily follows the [Google TypeScript Style-Guide](https://github.com/google/ts-style) through the provided *tslint-google.json* configuration file.

# License

As of License-File: *MIT*
