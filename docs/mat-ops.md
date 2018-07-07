# Class: `MatOps`

The class `MatOps` holds all the necessary matrix operations for assambling a forward-pass and their respective derivatives (`get...Backprop`) to perform backpropagation.
All methods are static (stateindependent) methods.

## Class Structure

* Each Matrix Operation:
  * executes the actual Matrix Operation
  * throws an Error Message if dimensions are not aligned
  * returns a new `Mat`-instance with resulting values
* Each `get...Backprop`-Function:
  * returns a derivative Function, that keeps the **references** of the given inputs
* Available Matrix Operations:
  * `[static] rowPluck(m: Mat, rowIndex: number): Mat`
  * `[static] getRowPluckBackprop(m: Mat, rowIndex: number, out: Mat): Mat`
  * `[static] gauss(m: Mat, std: Mat): Mat`
  * `[static] tanh(m: Mat): Mat`
  * `[static] sig(m: Mat): Mat`
  * `[static] relu(m: Mat): Mat`
  * `[static] add(mat1: Mat, mat2: Mat): Mat`
  * `[static] mul(mat1: Mat, mat2: Mat): Mat`
  * `[static] dot(mat1: Mat, mat2: Mat): Mat`
  * `[static] eltmul(mat1: Mat, mat2: Mat): Mat`
* Available `get...Backprop`-Functions:
  * `[static] getTanhBackprop(m: Mat, out: Mat): Function`
  * `[static] getSigmoidBackprop(m: Mat, out: Mat): Function`
  * `[static] getReluBackprop(m: Mat, out: Mat): Function`
  * `[static] getAddBackprop(mat1: Mat, mat2: Mat, out: Mat): Function`
  * `[static] getMulBackprop(mat1: Mat, mat2: Mat, out: Mat): Function`
  * `[static] getDotBackprop(mat1: Mat, mat2: Mat, out: Mat): Function`
  * `[static] getEltmulBackprop(mat1: Mat, mat2: Mat, out: Mat): Function`

## Usage

Import the `MatOps`-class and use the provided **static** methods.
Each method comes with its respective description.

### Matrix Operation

All Matrix Operations are **non-destructive**, meaning: they all return a new `Mat`-instance with the resulting values.

```typescript
import { MatOps } from 'recurrent-js';

/* 
 * const input = new Mat(1, 4);
 * input.setFrom([0, 1, 2, 3]);
 */

const mat = MatOps.tanh(input);
```

### Backprop/Derivative-Functions

```typescript
import { MatOps } from 'recurrent-js';

/* 
 * const input = new Mat(1, 4);
 * input.setFrom([0, 1, 2, 3]);
 * 
 * const mat = MatOps.tanh(input);
 */

/*
 * Inject:
 * - input: the input that lead to an output
 * - out:   the corresponding output
 */
const tanhBackprop = MatOps.getTanhBackprop(input, mat);

/* ready to perform backprop */
tanhBackprop();
```
