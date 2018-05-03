import { Mat } from '.';
import { Assertable } from './utils/assertable';

export class Utils extends Assertable {

  // Random numbers utils
  public static randf(min: number, max: number): number {
    return Math.random() * (max - min) + min;
  }
  public static randi(min: number, max: number): number {
    return Math.floor(Utils.randf(min, max));
  }
  public static randn(mu: number, std: number): number {
    return mu + Utils.gaussRandom() * std;
  }

  // TODO: Static could lead to unwanted behavior in async processes
  private static returnV = false;
  private static vVal = 0.0;

  private static gaussRandom(): number {
    if (Utils.returnV) {
      Utils.returnV = false;
      return Utils.vVal;
    }
    const u = 2 * Math.random() - 1;
    const v = 2 * Math.random() - 1;
    const r = u * u + v * v;
    if (r === 0 || r > 1) { return Utils.gaussRandom(); }
    const c = Math.sqrt(-2 * Math.log(r) / r);
    Utils.vVal = v * c; // cache this
    Utils.returnV = true;
    return u * c;
  }

  // Mat utils
  public static fillRandn(arr: Array<number> | Float64Array, mu: number, std: number): void {
    for (let i = 0; i < arr.length; i++) { arr[i] = Utils.randn(mu, std); }
  }
  public static fillRand(arr: Array<number> | Float64Array, lo: number, hi: number): void {
    for (let i = 0; i < arr.length; i++) { arr[i] = Utils.randf(lo, hi); }
  }
  public static fillConst(arr: Array<number> | Float64Array, c: number): void {
    for (let i = 0; i < arr.length; i++) { arr[i] = c; }
  }


  // Array utils
  /**
   * returns array of zeros of length n and uses typed arrays if available
   * @param n length of Array
   */
  public static zeros(n): Array<number> | Float64Array {
    if (typeof (n) === 'undefined' || isNaN(n)) { return []; }
    if (typeof ArrayBuffer === 'undefined') {
      const arr = new Array<number>(n);
      Utils.fillConst(arr, 0);
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
  public static maxi(w: Array<number> | Float64Array): number {
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

  /**
   * Returns an index of the weighted sample of Array `p`
   * @param p Array to be sampled
   */
  public static sampleWeighted(p: Array<number>): number {
    const r = Math.random();
    let c = 0.0;
    for (let i = 0; i < p.length; i++) {
      c += p[i];
      if (c >= r) { return i; }
    }

    Utils.assert(false, 'weighted sampling went wrong');
  }

}
