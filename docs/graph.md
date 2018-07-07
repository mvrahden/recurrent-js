# Class: `Graph`

The Graph class is important for the neural networks in a way that it keeps track of the relations between the matrices used in these networks.
The following sections further describe the `Graph` class and its usage.

## Class Structure
* Constructor: `Graph()`
* Provided Matrix Operations:
  * Delegate the actual Matrix Operation call
  * Keeps protocol of the sequence of matrix operations
  * Each operation returns a new `Mat`-Object, containing the specific results.
  * Available Matrix Operations are:
    * `rowPluck(m: Mat, rowIndex: number): Mat`
    * `gauss(m: Mat, std: Mat): Mat`
    * `tanh(m: Mat): Mat`
    * `sig(m: Mat): Mat`
    * `relu(m: Mat): Mat`
    * `add(mat1: Mat, mat2: Mat): Mat`
    * `mul(mat1: Mat, mat2: Mat): Mat`
    * `dot(mat1: Mat, mat2: Mat): Mat`
    * `eltmul(mat1: Mat, mat2: Mat): Mat`
* `memorizeOperationSequence(isMemorizing: boolean): void`: Switch, whether the graph should keep a protocol of the operation sequence for backpropagation
* `isMemorizingSequence(): boolean`: Get current memorization state
* `forgetCurrentSequence(): void`: Clear the graph memory
* `backward(): void`: Calls the Backpropagation Stack in reverse (LIFO) order of Matrix Operation Derivatives.

## Usage

### Object creation

Create a graph that does (or does not) memorize the sequence of Matrix Operations.

```typescript
const graph = new Graph();

/* OPTIONAL: Set Backprop-state to `true` */
graph.memorizeOperationSequence(true);
```

### Matrix Operation Call e.g. `sig(m: Mat): Mat`

Create a graph and inject a `Mat` object to call a sigmoid operation on its respective elements. A graph object - with backpropagation enabled - memorizes the derivatives of the matrix operations in the order they have been called.

```typescript
const graph = new Graph();
graph.memorizeOperationSequence(true); /* OPTIONAL: Set Backprop-state to `true` */

const mat = new Mat(4, 1);
mat.setFrom([0.1, 0.3, 0.9, 0.4]); /* fill Matrix with values */

/* 
 * - calls sigmoid operation on matrix,
 * - registers the matrix operation to the backpropagation stack (if activated) and
 * - returns a new matrix object with respective results
 */
const result = graph.sig(mat);
```

### Call of the `backward(): void` Method

After the execution of a sequence of matrix operations via a `Graph`-object, this graph is then able to execute the backpropagation process in reverse (LIFO) order.

**Prerequisite:** The `Graph`-object needs to memorize the sequence of matrix operations.
* Call `graph.memorizeOperationSequence(true);` for that
* Check the memorization state of graph with `graph.isMemorizingSequence();`

```typescript
graph.backward();
```
