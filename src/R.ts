import { RandMat } from './RandMat';
import { Graph } from "./Graph";
import { Net } from './Net';
import { Mat } from './Mat';

export class R {

  // Random numbers utils
  public static randf(a: number, b: number):number { return Math.random() * (b - a) + a; }
  public static randi(a: number, b: number):number { return Math.floor(R.randf(a, b)); }
  public static randn(mu: number, std: number):number { return mu + R.gaussRandom() * std; }

  // TODO: Static could lead to unwanted behavior in async processes
  private static returnV = false;
  private static vVal = 0.0;

  private static gaussRandom ():number {
    if (R.returnV) {
      R.returnV = false;
      return R.vVal;
    }
    const u = 2 * Math.random() - 1;
    const v = 2 * Math.random() - 1;
    const r = u * u + v * v;
    if (r === 0 || r > 1) { return R.gaussRandom(); }
    const c = Math.sqrt(-2 * Math.log(r) / r);
    R.vVal = v * c; // cache this
    R.returnV = true;
    return u * c;
  }

  // Mat utils
  public static fillRandn(m: Mat, mu: number, std: number) { for (let i = 0; i < m.w.length; i++) { m.w[i] = R.randn(mu, std); } }
  public static fillRand(m: Mat, lo: number, hi: number) { for (let i = 0; i < m.w.length; i++) { m.w[i] = R.randf(lo, hi); } }
  public static gradFillConst(m: Mat, c: number) { for (let i = 0; i < m.dw.length; i++) { m.dw[i] = c; } }


  // Array utils
  /**
   * Populates an array with a constant value
   * @param arr Array to be filled
   * @param c value to be set
   */
  public static setConst(arr:Array<number>, c:number) {
    for (let i = 0; i < arr.length; i++) {
      arr[i] = c;
    }
  }

  /**
   * returns array of zeros of length n and uses typed arrays if available
   * @param n length of Array
   */
  public static zeros(n): any {
    if (typeof (n) === 'undefined' || isNaN(n)) { return []; }
    if (typeof ArrayBuffer === 'undefined') {
      const arr = new Array(n);
      R.setConst(arr, 0);
      return arr;
    } else {
      return new Float64Array(n);
    }
  }

  /**
   * Argmax of Array `w`
   * @param w Array of Numbers
   * @returns Index of Argmax Operation
   */
  public static maxi(w:Array<number>): number {
    let maxv = w[0];
    let maxix = 0;
    for (let i = 1; i < w.length; i++) {
      const v = w[i];
      if (v > maxv) {
        maxix = i;
        maxv = v;
      }
    }
    return maxix;
  }
  // // return Mat but filled with random numbers from gaussian
  // static Graph (needsBackprop:boolean):Graph {
  //   return new Graph(needsBackprop);
  // }

  // // return Mat but filled with random numbers from gaussian
  // static Mat (n, d):Mat {
  //   return new Mat(n, d);
  // }

  // // return Mat but filled with random numbers from gaussian
  // static RandMat (n:number, d:number, mu:number, std:number):Mat {
  //   return new RandMat(n, d, mu, std);
  // }

  // static updateNet (net:Net, alpha:number):void { Net.update(net, alpha); }

}
