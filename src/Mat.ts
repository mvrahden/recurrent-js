import { Assertable } from './utils/Assertable';

import { R } from './R';

export class Mat extends Assertable {
  n: number;
  d: number;
  // TODO: should be declared private
  dw: Array<number>;
  w: Array<number>;

  constructor(n: number, d: number) {
    super();
    this.n = n;
    this.d = d;
    this.w = R.zeros(n * d);
    this.dw = R.zeros(n * d);
  }

  get(row: number, col: number): number {
    // slow but careful accessor function
    // we want row-major order
    const ix = (this.d * row) + col;
    Mat.assert(ix >= 0 && ix < this.w.length);
    return this.w[ix];
  }

  set(row: number, col: number, v: number): void {
    // slow but careful accessor function
    const ix = (this.d * row) + col;
    Mat.assert(ix >= 0 && ix < this.w.length);
    this.w[ix] = v;
  }

  setFrom(arr: Array<number>): void {
    for (let i = 0, n = arr.length; i < n; i++) {
      this.w[i] = arr[i];
    }
  }

  setColumn(m: Mat, i: number): void {
    for (let q = 0, n = m.w.length; q < n; q++) {
      this.w[(this.d * q) + i] = m.w[q];
    }
  }

  static update(m: Mat, alpha: number): void {
    // updates in place
    for (let i = 0, n = m.n * m.d; i < n; i++) {
      if (m.dw[i] !== 0) {
        m.w[i] += - alpha * m.dw[i];
        m.dw[i] = 0;
      }
    }
  }

  toJSON(): {} {
    const json = {};
    json['n'] = this.n;
    json['d'] = this.d;
    json['w'] = this.w;
    return json;
  }

  fromJSON(json): void {
    this.n = json.n;
    this.d = json.d;
    this.w = R.zeros(this.n * this.d);
    this.dw = R.zeros(this.n * this.d);
    for (let i = 0, n = this.n * this.d; i < n; i++) {
      this.w[i] = json.w[i]; // copy over weights
    }
  }
}
