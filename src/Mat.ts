import { Assertable } from './utils/Assertable';

import { Utils } from './Utils';

export class Mat extends Assertable {
  
  public readonly rows: number;
  public readonly cols: number;
  private readonly length: number;  // length of 1d-representaion of Mat

  public readonly w: Array<number> | Float64Array;
  public readonly dw: Array<number> | Float64Array;

  /**
   * 
   * @param rows rows of Matrix
   * @param cols columns of Matrix
   */
  constructor(rows: number, cols: number) {
    super();
    this.rows = rows;
    this.cols = cols;
    this.length = rows * cols;
    this.w = Utils.zeros(this.length);
    this.dw = Utils.zeros(this.length);
  }

  /**
   * Accesses the value of given row and column.
   * @param row 
   * @param col
   * @returns the value of given row and column
   */
  get(row: number, col: number): number {
    const ix = this.getIndexBy(row, col);
    Mat.assert(ix >= 0 && ix < this.w.length, 'index out of bounds.');
    return this.w[ix];
  }

  /**
   * Mutates the value of given row and column.
   * @param row 
   * @param col 
   * @param v 
   */
  set(row: number, col: number, v: number): void {
    const ix = this.getIndexBy(row, col);
    Mat.assert(ix >= 0 && ix < this.w.length, 'index out of bounds.');
    this.w[ix] = v;
  }

  /**
   * Gets Index by Row-major order
   * @param row 
   * @param col 
   */
  protected getIndexBy(row: number, col: number) {
    return (row * this.cols) + col;
  }

  /**
   * Sets values according to the given Array.
   * @param arr 
   */
  public setFrom(arr: Array<number> | Float64Array): void {
    for (let i = 0; i < arr.length; i++) {
      this.w[i] = arr[i];
    }
  }

  public setColumn(m: Mat, i: number): void {
    for (let q = 0; q < m.w.length; q++) {
      this.w[(this.cols * q) + i] = m.w[q];
    }
  }

  /**
   * Discounts all values as follows: w[i] = w[i] - (alpha * dw[i])
   * @param alpha discount factor
   */
  public update(alpha: number): void {
    for (let i = 0; i < this.length; i++) {
      if (this.dw[i] !== 0) {
        this.w[i] += - alpha * this.dw[i];
        this.dw[i] = 0;
      }
    }
  }

  public static toJSON(m: Mat | any): {} {
    const json = {};
    json['rows'] = m.rows || m.n;
    json['cols'] = m.cols || m.d;
    json['w'] = m.w;
    return json;
  }

  public static fromJSON(json: {rows, n?, cols, d?, w}): Mat {
    const rows = json.rows || json.n;
    const cols = json.cols || json.d;
    const mat = new Mat(rows, cols);
    for (let i = 0; i < mat.length; i++) {
      mat.w[i] = json.w[i]; // copy over weights
    }
    return mat;
  }

  /**
   * Non-destructive elementwise add.
   * @param {Mat} m1 
   * @param {Mat} m2 
   */
  public static add(m1: Mat, m2: Mat): Mat {
    Mat.assert(m1.w.length === m2.w.length && m1.rows === m2.rows, 'matadd dimensions misaligned');
    const out = new Mat(m1.rows, m1.cols);
    for (let i = 0; i < m1.w.length; i++) {
      out.w[i] = m1.w[i] + m2.w[i];
    }
    return out;
  }

  /**
   * Non-destructive elementwise sigmoid.
   * @param m 
   * @returns Mat with results
   */
  public static sig(m: Mat): Mat {
    const out = new Mat(m.rows, m.cols);
    for (let i = 0; i < m.w.length; i++) {
      out.w[i] = Mat.sigmoid(m.w[i]);
    }
    return out;
  }
  
  private static sigmoid(x: number): number {
    // helper function for computing sigmoid
    return 1.0 / (1 + Math.exp(-x));
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

  /**
   * Non-destructive Matrix multiplication.
   * @param m1 
   * @param m2 
   * @returns Mat with results
   */
  public static mul(m1: Mat, m2: Mat): Mat {
    Mat.assert(m1.cols === m2.rows, 'matmul dimensions misaligned');
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

  /**
   * Non-destructive dot Product.
   * @param m1 
   * @param m2 
   * @return {Mat} Matrix of dimension 1x1
   */
  public static dot(m1: Mat, m2: Mat): Mat {
    Mat.assert(m1.w.length === m2.w.length, 'matdot dimensions misaligned');
    const out = new Mat(1, 1);
    let dot = 0.0;
    for (let i = 0; i < m1.w.length; i++) {
      dot += m1.w[i] * m2.w[i];
    }
    out.w[0] = dot;
    return out;
  }

  /**
   * Non-destructive elementwise Matrix multiplication.
   * @param m1 
   * @param m2 
   * @return {Mat} Matrix with results
   */
  public static eltmul(m1: Mat, m2: Mat): Mat {
    Mat.assert(m1.w.length === m2.w.length && m1.rows === m2.rows, 'mateltmul dimensions misaligned');
    const out = new Mat(m1.rows, m1.cols);
    for (let i = 0; i < m1.w.length; i++) {
      out.w[i] = m1.w[i] * m2.w[i];
    }
    return out;
  }

  /**
   * Non-destructively pluck a row of m with rowIndex
   * @param m 
   * @param rowIndex index of row
   * @returns a column Vector [cols, 1]
   */
  static rowPluck(m: Mat, rowIndex: number): Mat {
    Mat.assert(rowIndex >= 0 && rowIndex < m.rows, 'mateltmul dimensions misaligned');
    const out = new Mat(m.cols, 1);
    for (let i = 0; i < m.cols; i++) {
      out.w[i] = m.w[m.cols * rowIndex + i];
    }
    return out;
  }

}
