import { Utils } from '.';
import { Assertable } from './utils/assertable';

export class Mat extends Assertable {
  
  public readonly rows: number;
  public readonly cols: number;
  private readonly _length: number;  // length of 1d-representation of Mat

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
    this._length = rows * cols;
    this.w = Utils.zeros(this._length);
    this.dw = Utils.zeros(this._length);
  }

  /**
   * Accesses the value of given row and column.
   * @param row 
   * @param col
   * @returns the value of given row and column
   */
  public get(row: number, col: number): number {
    const ix = this.getIndexBy(row, col);
    Mat.assert(ix >= 0 && ix < this.w.length, '[class:Mat] get: index out of bounds.');
    return this.w[ix];
  }

  /**
   * Mutates the value of given row and column.
   * @param row 
   * @param col 
   * @param v 
   */
  public set(row: number, col: number, v: number): void {
    const ix = this.getIndexBy(row, col);
    Mat.assert(ix >= 0 && ix < this.w.length, '[class:Mat] set: index out of bounds.');
    this.w[ix] = v;
  }

  /**
   * Gets Index by Row-major order
   * @param row 
   * @param col 
   */
  protected getIndexBy(row: number, col: number): number {
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

  /**
   * Overrides the values from the column of the matrix
   * @param m 
   * @param colIndex 
   */
  public setColumn(m: Mat, colIndex: number): void {
    Mat.assert(m.w.length === this.rows, '[class:Mat] setColumn: dimensions misaligned.');
    for (let i = 0; i < m.w.length; i++) {
      this.w[(this.cols * i) + colIndex] = m.w[i];
    }
  }

  /**
   * Checks equality of matrices.
   * The check includes the value equality and a dimensionality check.
   * Derivatives are not considered.
   * @param {Mat} m Matrix to be compared with
   * @returns {boolean} true if equal and false otherwise
   */
  public equals(m: Mat): boolean {
    if(this.rows !== m.rows || this.cols !== m.cols) {
      return false;
    }
    for(let i = 0; i < this._length; i++) {
      if(this.w[i] !== m.w[i]) {
        return false;
      }
    }
    return true;
  }

  public static toJSON(m: Mat | any): {rows, cols, w} {
    const json = {rows: 0, cols: 0, w: []};
    json.rows = m.rows || m.n;
    json.cols = m.cols || m.d;
    json.w = m.w;
    return json;
  }

  public static fromJSON(json: {rows, n?, cols, d?, w}): Mat {
    const rows = json.rows || json.n;
    const cols = json.cols || json.d;
    const mat = new Mat(rows, cols);
    for (let i = 0; i < mat._length; i++) {
      mat.w[i] = json.w[i];
    }
    return mat;
  }

  /**
   * Discounts all values as follows: w[i] = w[i] - (alpha * dw[i])
   * @param alpha discount factor
   */
  public update(alpha: number): void {
    for (let i = 0; i < this._length; i++) {
      if (this.dw[i] !== 0) {
        this.w[i] = this.w[i] - alpha * this.dw[i];
        this.dw[i] = 0;
      }
    }
  }

}
