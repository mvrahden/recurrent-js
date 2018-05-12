# Class: `Graph`

The Graph class is important for the neural networks in a way that it keeps track of the relations between the matrices used in these networks.
The following sections further describe the `Graph` class and its usage.

## Class Structure
* Constructor: `Graph(needsBackpropagation: boolean)`
* Provided Matrix Operations:
  * Delegate the actual Matrix Operation call
  * Adds the Derivative of that Matrix Operation to Backpropagation Stack
  * Returns a new Matrix object containing the results.
  * Available Matrix Operations are:
    * `rowPluck(m: Mat, rowIndex: number): Mat`
    * `tanh(m: Mat): Mat`
    * `sig(m: Mat): Mat`
    * `relu(m: Mat): Mat`
    * `add(mat1: Mat, mat2: Mat): Mat`
    * `mul(mat1: Mat, mat2: Mat): Mat`
    * `dot(mat1: Mat, mat2: Mat): Mat`
    * `eltmul(mat1: Mat, mat2: Mat): Mat`
* `backward(): void`: Calls the Backpropagation Stack in reverse (LIFO) order of Matrix Operation Derivatives.

## Usage

### Object creation

Create a graph that does (or does not) memorize the sequence of Matrix Operations.

```typescript
const graph = new Graph(true); // if no backpropagation needed: false
```

### Matrix Operation Call e.g. `sig(m: Mat): Mat`

Create a graph and inject a `Mat` object to call a sigmoid operation on its respective elements. A graph object with backpropagation memorizes the derivatives of the matrix operations in the order they have been called.

```typescript
const graph = new Graph(true); // if no backpropagation needed: false
const mat = new Mat(4, 1);

/* fill Matrix with values */
mat.setFrom([0.1, 0.3, 0.9, 0.4]);

/* 
 - call sigmoid operation on matrix,
 - register the matrix operation to the backpropagation stack and
 - receive a new matrix object with results
*/
const result = graph.sig(mat);
```

### Call of the `backward(): void` Method

After the execution of a sequence of matrix operations via a graph-object, the graph is then able to execute the backpropagation process in reverse (LIFO) order.

```typescript
graph.backward();
```