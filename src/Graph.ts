import { Mat } from './Mat';

export class Graph {
  private needsBackprop: boolean;

  public readonly backprop: Array<Function>;

  constructor(needsBackprop: boolean = true) {
    this.needsBackprop = needsBackprop;

    // this will store a list of functions that perform backprop,
    // in their forward pass order. So in backprop we will go
    // backwards and evoke each one
    this.backprop = new Array<Function>();
  }

  /**
   * Backpropagation
   */
  public backward(): void {
    for (let i = this.backprop.length - 1; i >= 0; i--) {
      this.backprop[i]();
    }
  }

  /**
   * Non-destructively pluck a row of m with rowIndex
   * @param m 
   * @param ix 
   */
  public rowPluck(m: Mat, ix: number): Mat {
    const out = Mat.rowPluck(m, ix);
    this.addRowPluckToBackprop(m, ix, out);
    return out;
  }

  private addRowPluckToBackprop(m: Mat, ix: number, out: Mat) {
    if (this.needsBackprop) {
      const backward = () => {
        for (let i = 0; i < m.cols; i++) {
          m.dw[m.cols * ix + i] += out.dw[i];
        }
      };
      this.backprop.push(backward);
    }
  }

  /**
   * Non-destructive elementwise tanh
   * @param m 
   */
  public tanh(m: Mat): Mat {
    const out = Mat.tanh(m);
    this.addTanhToBackprop(m, out);
    return out;
  }

  private addTanhToBackprop(m: Mat, out: Mat) {
    if (this.needsBackprop) {
      const backward = () => {
        for (let i = 0; i < m.w.length; i++) {
          // grad for z = tanh(x) is (1 - z^2)
          const mwi = out.w[i];
          m.dw[i] += (1.0 - mwi * mwi) * out.dw[i];
        }
      };
      this.backprop.push(backward);
    }
  }

  /**
   * Non-destructive elementwise sigmoid
   * @param m 
   */
  public sigmoid(m: Mat): Mat {
    const out = Mat.sig(m);
    this.addSigmoidToBackprop(m, out);
    return out;
  }

  private addSigmoidToBackprop(m: Mat, out: Mat) {
    if (this.needsBackprop) {
      const backward = () => {
        for (let i = 0; i < m.w.length; i++) {
          // grad for z = tanh(x) is (1 - z^2)
          const mwi = out.w[i];
          m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
        }
      };
      this.backprop.push(backward);
    }
  }

  /**
   * Non-destructive elementwise ReLU (rectified linear unit)
   * @param m 
   */
  public relu(m: Mat): Mat {
    const out = Mat.relu(m);
    this.addReluToBackprop(m, out);
    return out;
  }

  private addReluToBackprop(m: Mat, out: Mat) {
    if (this.needsBackprop) {
      const backward = () => {
        for (let i = 0; i < m.w.length; i++) {
          m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0;
        }
      };
      this.backprop.push(backward);
    }
  }
  /**
   * Non-destructive matrix multiplication
   * @param m1 
   * @param m2 
   */
  public mul(m1: Mat, m2: Mat): Mat {
    const out = Mat.mul(m1, m2);
    this.addMultiplyToBackprop(m1, m2, out);
    return out;
  }

  private addMultiplyToBackprop(m1: Mat, m2: Mat, out: Mat) {
    if (this.needsBackprop) {
      const backward = () => {
        for (let i = 0; i < m1.rows; i++) {
          for (let j = 0; j < m2.cols; j++) {
            for (let k = 0; k < m1.cols; k++) {
              const b = out.dw[m2.cols * i + j];
              m1.dw[m1.cols * i + k] += m2.w[m2.cols * k + j] * b;
              m2.dw[m2.cols * k + j] += m1.w[m1.cols * i + k] * b;
            }
          }
        }
      };
      this.backprop.push(backward);
    }
  }

  /**
   * Non-destructive elementwise addition
   * @param m1 
   * @param m2 
   */
  public add(m1: Mat, m2: Mat): Mat {
    const out = Mat.add(m1, m2);
    this.addAdditionToBackprop(m1, m2, out);
    return out;
  }

  private addAdditionToBackprop(m1: Mat, m2: Mat, out: Mat) {
    if (this.needsBackprop) {
      const backward = () => {
        for (let i = 0; i < m1.w.length; i++) {
          m1.dw[i] += out.dw[i];
          m2.dw[i] += out.dw[i];
        }
      };
      this.backprop.push(backward);
    }
  }

  /**
   * Non-destructive Dot product.
   * @param m1 
   * @param m2 
   */
  public dot(m1: Mat, m2: Mat): Mat {
    const out = Mat.dot(m1, m2);
    this.addDotToBackprop(m1, m2, out);
    return out;
  }

  private addDotToBackprop(m1: Mat, m2: Mat, out: Mat) {
    if (this.needsBackprop) {
      const backward = () => {
        for (let i = 0; i < m1.w.length; i++) {
          m1.dw[i] += m2.w[i] * out.dw[0];
          m2.dw[i] += m1.w[i] * out.dw[0];
        }
      };
      this.backprop.push(backward);
    }
  }

  /**
   * Non-destructively elementwise multiplication
   * @param m1 
   * @param m2 
   */
  public eltmul(m1: Mat, m2: Mat): Mat {
    const out = Mat.eltmul(m1, m2);
    this.addEltmulToBackprop(m1, m2, out);
    return out;
  }

  private addEltmulToBackprop(m1: Mat, m2: Mat, out: Mat) {
    if (this.needsBackprop) {
      const backward = () => {
        for (let i = 0; i < m1.w.length; i++) {
          m1.dw[i] += m2.w[i] * out.dw[i];
          m2.dw[i] += m1.w[i] * out.dw[i];
        }
      };
      this.backprop.push(backward);
    }
  }
}
