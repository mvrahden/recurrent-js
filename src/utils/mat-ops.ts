import { Mat, Utils } from '..';
import { Assertable } from './assertable';

export class MatOps extends Assertable {

  /**
   * Non-destructively pluck a row of m with rowIndex
   * @param m 
   * @param rowIndex index of row
   * @returns a column Vector [cols, 1]
   */
  public static rowPluck(m: Mat, rowIndex: number): Mat {
    Mat.assert(rowIndex >= 0 && rowIndex < m.rows, '[class:MatOps] rowPluck: dimensions misaligned');
    const out = new Mat(m.cols, 1);
    for (let i = 0; i < m.cols; i++) {
      out.w[i] = m.w[m.cols * rowIndex + i];
    }
    return out;
  }

  public static getRowPluckBackprop(m: Mat, rowIndex: number, out: Mat): Function {
    return () => {
      for (let i = 0; i < m.cols; i++) {
        m.dw[m.cols * rowIndex + i] += out.dw[i];
      }
    };
  }

  /**
   * Non-destructive elementwise gaussian-distributed noise-addition.
   * @param {Mat} m 
   * @param {number} std Matrix with STD values
   * @returns {Mat} Matrix with results
   */
  public static gauss(m: Mat, std: Mat): Mat {
    Mat.assert(m.w.length === std.w.length, '[class:MatOps] gauss: dimensions misaligned');
    const out = new Mat(m.rows, m.cols);
    for (let i = 0; i < m.w.length; i++) {
      out.w[i] = Utils.randn(m.w[i], std.w[i]);
    }
    return out;
  }

  /**
   * Non-destructive elementwise tanh.
   * @param {Mat} m
   * @returns {Mat} Matrix with results
   */
  public static tanh(m: Mat): Mat {
    const out = new Mat(m.rows, m.cols);
    for (let i = 0; i < m.w.length; i++) {
      out.w[i] = Math.tanh(m.w[i]);
    }
    return out;
  }

  public static getTanhBackprop(m: Mat, out: Mat): Function {
    return () => {
      for (let i = 0; i < m.w.length; i++) {
        // grad for z = tanh(x) is (1 - z^2)
        const mwi = out.w[i];
        m.dw[i] += (1.0 - mwi * mwi) * out.dw[i];
      }
    };
  }

  /**
   * Non-destructive elementwise sigmoid.
   * @param m 
   * @returns Mat with results
   */
  public static sig(m: Mat): Mat {
    const out = new Mat(m.rows, m.cols);
    for (let i = 0; i < m.w.length; i++) {
      out.w[i] = MatOps.sigmoid(m.w[i]);
    }
    return out;
  }
  
  private static sigmoid(x: number): number {
    // helper function for computing sigmoid
    return 1.0 / (1 + Math.exp(-x));
  }

  public static getSigmoidBackprop(m: Mat, out: Mat): Function {
    return () => {
      for (let i = 0; i < m.w.length; i++) {
        // grad for z = tanh(x) is (1 - z^2)
        const mwi = out.w[i];
        m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
      }
    };
  }

  /**
   * Non-destructive elementwise ReLu.
   * @returns Mat with results
   */
  public static relu(m: Mat): Mat {
    const out = new Mat(m.rows, m.cols);
    for (let i = 0; i < m.w.length; i++) {
      out.w[i] = Math.max(0, m.w[i]); // relu
    }
    return out;
  }

  public static getReluBackprop(m: Mat, out: Mat): Function {
    return () => {
      for (let i = 0; i < m.w.length; i++) {
        m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0;
      }
    };
  }

  /**
   * Non-destructive elementwise add.
   * @param {Mat} m1 
   * @param {Mat} m2 
   */
  public static add(m1: Mat, m2: Mat): Mat {
    Mat.assert(m1.w.length === m2.w.length && m1.rows === m2.rows, '[class:MatOps] add: dimensions misaligned');
    const out = new Mat(m1.rows, m1.cols);
    for (let i = 0; i < m1.w.length; i++) {
      out.w[i] = m1.w[i] + m2.w[i];
    }
    return out;
  }

  public static getAddBackprop(m1: Mat, m2: Mat, out: Mat): Function {
    return () => {
      for (let i = 0; i < m1.w.length; i++) {
        m1.dw[i] += out.dw[i];
        m2.dw[i] += out.dw[i];
      }
    };
  }

  /**
   * Non-destructive Matrix multiplication.
   * @param m1 
   * @param m2 
   * @returns Mat with results
   */
  public static mul(m1: Mat, m2: Mat): Mat {
    Mat.assert(m1.cols === m2.rows, '[class:MatOps] mul: dimensions misaligned');
    const out = new Mat(m1.rows, m2.cols);
    for (let row = 0; row < m1.rows; row++) { // loop over rows of m1
      for (let col = 0; col < m2.cols; col++) { // loop over cols of m2
        let dot = 0.0;
        for (let k = 0; k < m1.cols; k++) { // dot product loop
          dot += m1.w[m1.cols * row + k] * m2.w[m2.cols * k + col];
        }
        out.w[m2.cols * row + col] = dot;
      }
    }
    return out;
  }

  public static getMulBackprop(m1: Mat, m2: Mat, out: Mat): Function {
    return () => {
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
  }

  /**
   * Non-destructive dot Product.
   * @param m1 
   * @param m2 
   * @return {Mat} Matrix of dimension 1x1
   */
  public static dot(m1: Mat, m2: Mat): Mat {
    Mat.assert(m1.w.length === m2.w.length && m1.rows === m2.rows, '[class:MatOps] dot: dimensions misaligned');
    const out = new Mat(1, 1);
    let dot = 0.0;
    for (let i = 0; i < m1.w.length; i++) {
      dot += m1.w[i] * m2.w[i];
    }
    out.w[0] = dot;
    return out;
  }

  public static getDotBackprop(m1: Mat, m2: Mat, out: Mat): Function {
    return () => {
      for (let i = 0; i < m1.w.length; i++) {
        m1.dw[i] += m2.w[i] * out.dw[0];
        m2.dw[i] += m1.w[i] * out.dw[0];
      }
    };
  }

  /**
   * Non-destructive elementwise Matrix multiplication.
   * @param m1 
   * @param m2 
   * @return {Mat} Matrix with results
   */
  public static eltmul(m1: Mat, m2: Mat): Mat {
    Mat.assert(m1.w.length === m2.w.length && m1.rows === m2.rows, '[class:MatOps] eltmul: dimensions misaligned');
    const out = new Mat(m1.rows, m1.cols);
    for (let i = 0; i < m1.w.length; i++) {
      out.w[i] = m1.w[i] * m2.w[i];
    }
    return out;
  }

  public static getEltmulBackprop(m1: Mat, m2: Mat, out: Mat): Function {
    return () => {
      for (let i = 0; i < m1.w.length; i++) {
        m1.dw[i] += m2.w[i] * out.dw[i];
        m2.dw[i] += m1.w[i] * out.dw[i];
      }
    };
  }
}
