import { Mat } from '.';
import { MatOps } from './utils/mat-ops';

export class Graph {
  private needsBackpropagation: boolean;

  private readonly backpropagationStack: Array<Function>;

  /**
   * Initializes a Graph to memorize Matrix Operation Sequences.
   */
  constructor() {
    this.needsBackpropagation = false;

    this.backpropagationStack = new Array<Function>();
  }

  /**
   * Switch whether to memorize the operation sequence for Backpropagation (true) or ignore it (false).
   * @param {boolean} isMemorizing true or false [defaults to false]
   */
  public memorizeOperationSequence(isMemorizing: boolean = false) {
    this.needsBackpropagation = isMemorizing;
  }

  /**
   * Gives back the state of either memorizing or not a sequence of operations
   */
  public isMemorizingSequence(): boolean {
    return this.needsBackpropagation;
  }

  /**
   * Clears the memorized sequence of operations
   */
  public forgetCurrentSequence(): void {
    this.backpropagationStack.length = 0; // reset array
  }

  /**
   * Executes the memorized sequence of derivative operations in LIFO order
   */
  public backward(): void {
    for (let i = this.backpropagationStack.length - 1; i >= 0; i--) {
      this.backpropagationStack[i]();
    }
  }

  /**
   * Non-destructively pluck a row of m with rowIndex
   * @param m 
   * @param rowIndex 
   */
  public rowPluck(m: Mat, rowIndex: number): Mat {
    const out = MatOps.rowPluck(m, rowIndex);
    this.addRowPluckToBackpropagationStack(m, rowIndex, out);
    return out;
  }

  private addRowPluckToBackpropagationStack(m: Mat, rowIndex: number, out: Mat) {
    if (this.needsBackpropagation) {
      const backward = MatOps.getRowPluckBackprop(m, rowIndex, out);
      this.backpropagationStack.push(backward);
    }
  }

  /**
   * Non-destructively pluck a row of m with rowIndex
   * @param m 
   * @param rowIndex 
   */
  public gauss(m: Mat, std: Mat): Mat {
    const out = MatOps.gauss(m, std);
    return out;
  }

  /**
   * Non-destructive elementwise tanh
   * @param m 
   */
  public tanh(m: Mat): Mat {
    const out = MatOps.tanh(m);
    this.addTanhToBackpropagationStack(m, out);
    return out;
  }

  private addTanhToBackpropagationStack(m: Mat, out: Mat) {
    if (this.needsBackpropagation) {
      const backward = MatOps.getTanhBackprop(m, out);
      this.backpropagationStack.push(backward);
    }
  }

  /**
   * Non-destructive elementwise sigmoid
   * @param m 
   */
  public sig(m: Mat): Mat {
    const out = MatOps.sig(m);
    this.addSigmoidToBackpropagationStack(m, out);
    return out;
  }

  private addSigmoidToBackpropagationStack(m: Mat, out: Mat) {
    if (this.needsBackpropagation) {
      const backward = MatOps.getSigmoidBackprop(m, out);
      this.backpropagationStack.push(backward);
    }
  }

  /**
   * Non-destructive elementwise ReLU (rectified linear unit)
   * @param m 
   */
  public relu(m: Mat): Mat {
    const out = MatOps.relu(m);
    this.addReluToBackpropagationStack(m, out);
    return out;
  }

  private addReluToBackpropagationStack(m: Mat, out: Mat) {
    if (this.needsBackpropagation) {
      const backward = MatOps.getReluBackprop(m, out);
      this.backpropagationStack.push(backward);
    }
  }

  /**
   * Non-destructive matrix multiplication
   * @param m1 
   * @param m2 
   */
  public mul(m1: Mat, m2: Mat): Mat {
    const out = MatOps.mul(m1, m2);
    this.addMultiplyToBackpropagationStack(m1, m2, out);
    return out;
  }

  private addMultiplyToBackpropagationStack(m1: Mat, m2: Mat, out: Mat) {
    if (this.needsBackpropagation) {
      const backward = MatOps.getMulBackprop(m1, m2, out);
      this.backpropagationStack.push(backward);
    }
  }

  /**
   * Non-destructive elementwise addition
   * @param m1 
   * @param m2 
   */
  public add(m1: Mat, m2: Mat): Mat {
    const out = MatOps.add(m1, m2);
    this.addAdditionToBackpropagationStack(m1, m2, out);
    return out;
  }

  private addAdditionToBackpropagationStack(m1: Mat, m2: Mat, out: Mat) {
    if (this.needsBackpropagation) {
      const backward = MatOps.getAddBackprop(m1, m2, out);
      this.backpropagationStack.push(backward);
    }
  }

  /**
   * Non-destructive Dot product.
   * @param m1 
   * @param m2 
   */
  public dot(m1: Mat, m2: Mat): Mat {
    const out = MatOps.dot(m1, m2);
    this.addDotToBackpropagationStack(m1, m2, out);
    return out;
  }

  private addDotToBackpropagationStack(m1: Mat, m2: Mat, out: Mat) {
    if (this.needsBackpropagation) {
      const backward = MatOps.getDotBackprop(m1, m2, out);
      this.backpropagationStack.push(backward);
    }
  }

  /**
   * Non-destructively elementwise multiplication
   * @param m1 
   * @param m2 
   */
  public eltmul(m1: Mat, m2: Mat): Mat {
    const out = MatOps.eltmul(m1, m2);
    this.addEltmulToBackpropagationStack(m1, m2, out);
    return out;
  }

  private addEltmulToBackpropagationStack(m1: Mat, m2: Mat, out: Mat) {
    if (this.needsBackpropagation) {
      const backward = MatOps.getEltmulBackprop(m1, m2, out);
      this.backpropagationStack.push(backward);
    }
  }
}
