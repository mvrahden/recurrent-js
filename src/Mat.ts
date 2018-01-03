import { Assertable } from './utils/Assertable';

import { R } from './R';

export class Mat extends Assertable {
  public n: number;
  public d: number;

  public readonly w: Array<number>;
  public readonly dw: Array<number>;

  /**
   * 
   * @param n length of Matrix
   * @param d depth of Matrix
   */
  constructor(n: number, d: number) {
    super();
    this.n = n;
    this.d = d;
    this.w = R.zeros(n * d);
    this.dw = R.zeros(n * d);
  }

  /**
   * Accesses the value of given row and column.
   * @param row 
   * @param col
   * @returns the value of given row and column
   */
  get(row: number, col: number): number {
    const ix = this.getIndexBy(row, col);
    Mat.assert(ix >= 0 && ix < this.w.length);
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
    Mat.assert(ix >= 0 && ix < this.w.length);
    this.w[ix] = v;
  }

  /**
   * Get Index by Row-major order
   * @param row 
   * @param col 
   */
  private getIndexBy(row: number, col: number) {
    return (row * this.d) + col;
  }

  public setFrom(arr: Array<number>): void {
    for (let i = 0; i < arr.length; i++) {
      this.w[i] = arr[i];
    }
  }

  public setColumn(m: Mat, i: number): void {
    for (let q = 0; q < m.w.length; q++) {
      this.w[(this.d * q) + i] = m.w[q];
    }
  }

  /**
   * updates all values
   * @param alpha discount rate
   */
  public update(alpha: number): void {
    for (let i = 0, n = this.n * this.d; i < n; i++) {
      if (this.dw[i] !== 0) {
        this.w[i] += - alpha * this.dw[i];
        this.dw[i] = 0;
      }
    }
  }

  static toJSON(m: Mat): {} {
    const json = {};
    json['n'] = m.n;
    json['d'] = m.d;
    json['w'] = m.w;
    return json;
  }

  static fromJSON(json: {n, d, w}): Mat {
    const mat = new Mat(json.n, json.d);
    for (let i = 0, n = mat.n * mat.d; i < n; i++) {
      mat.w[i] = json.w[i]; // copy over weights
    }
    return mat;
  }
}
