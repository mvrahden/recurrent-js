# RECURRENT-ts

**RECURRENT-ts** is an object-oriented Typescript port of the _Andrej Karpathy's_ RecurrentJS library that implements the following:

* Deep **Recurrent Neural Networks** (RNN) 
* **Long Short-Term Memory networks** (LSTM) 
* In fact, the library is more general because it has functionality to construct arbitrary **expression graphs** over which the library can perform **automatic differentiation** similar to what you may find in Theano for Python, or in Torch etc. Currently, the code uses this very general functionality to implement RNN/LSTM, but one can build arbitrary Neural Networks and do automatic backprop.

For further Information see the [recurrent-js](https://github.com/karpathy/recurrentjs) repository.

# Use as `npm`-Project Dependency

## To install as dependency:

Download available `@npm`: [recurrent-ts](https://www.npmjs.com/package/recurrent-ts)

Install via command line:
```
npm install --save recurrent-ts
```

## To use the Library in Production:

Currently exposed Classes:

* *R* - Collection of Utility functions
* *Mat* - Sophisticated Matrix Structure
* *RandMat* - `Mat` with populated with random gaussian distributed values
* *Graph* - Graph with Operations
* *Net*
* *LSTM*

These classes can be directly imported from this `npm` module, e.g.:
```typescript
import { Garph, Net } from 'recurrent-ts';
```

# Contribute

1. `Clone` this project to a working directory.
2. `npm install` to setup the development dependencies.
3. To compile the codebase:

```
tsc -p .
```

This project relies on Visual Studio Codes built-in Typescript linting facilities. Let's follow primarily the [Google TypeScript Style-Guide](https://github.com/google/ts-style) through the included *tslint-google.json* configuration file.

# License

As of License-File: *MIT*
