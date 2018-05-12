# Class: `Mat`

The Mat class is important for the neural networks in a way that it holds the matrix values, their associated values of derivative operations and it holds the matrix operations.
The following sections further describe the `Mat` and its usage.

## Class Structure
* Constructor: `Mat(rows: number, cols: number)`
* Available Properties:
  * `w`: 1d array, holding the actual values of the matrix
  * `dw`: 1d array, holding the derivatives of the values of a matrix after calling derivative operations
* Administrative Methods:
  * `get(row: number, col: number): number`
  * `set(row: number, col: number, v: number): void`
  * `setFrom(arr: Array<number> | Float64Array): void`
  <!-- * `setRow(m: Mat, rowIndex): void` -->
  * `setColumn(m: Mat, colIndex: number): void`
  * `[static] toJSON(m: Mat | any): {rows, cols, w}`
  * `[static] fromJSON(m: {rows, cols, w}): Mat`
* Matrix Operations:
  * Execute the actual Matrix Operation
  * Throw an Error Message if dimensions are not aligned
  * Available Matrix Operations are:
    * `[static] rowPluck(m: Mat, rowIndex: number): Mat`
    * `[static] gauss(m: Mat, std: Mat): Mat`
    * `[static] tanh(m: Mat): Mat`
    * `[static] sig(m: Mat): Mat`
    * `[static] relu(m: Mat): Mat`
    * `[static] add(mat1: Mat, mat2: Mat): Mat`
    * `[static] mul(mat1: Mat, mat2: Mat): Mat`
    * `[static] dot(mat1: Mat, mat2: Mat): Mat`
    * `[static] eltmul(mat1: Mat, mat2: Mat): Mat`
* Other Methods:
  * `update(alpha: number): void`

## Usage

### Object creation

Create a matrix with defined rows and columns.

```typescript
const matrix = new Mat(4, 1);
```

### Set Values from an Array<number> | Float64Array

When copying the values from an Array to the matrix object, the array needs to be of length `rows * cols` and have the values ordered according to the sequence of the rows. (e.g. `[ row1 row2 ... ]`)

```typescript
matrix.setFrom([1, 4, 3, 2]);
```

### Matrix Operation Call e.g. `sig(m: Mat): Mat`

A `Mat` object executing a sigmoid operation on its respective elements.

```typescript
const mat = new Mat(4, 1);

mat.setFrom([0.1, 0.3, 0.9, 0.4]);

const result = Mat.sig(mat);
```

### Call of the `update(aplha: number): void` Method

To train a neural network, the weights need to manipulated in a way.
The `update()`-Method is a way to discount the values (kept in `w`) by the values of their respective derivatives (kept in `dw`).
The underlying mechanism subtracts the discounted derivative value from the real value like: `w[i] = w[i] - (dq[i] * alpha)`

```typescript
mat.update(0.01);
```