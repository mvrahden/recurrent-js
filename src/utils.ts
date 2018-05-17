import { Mat } from '.';

export class Utils {

  /**
   * Returns a random floating point number between `min` and `max`
   * @param {number} min lower bound
   * @param {number} max upper bound
   */
  public static randf(min: number, max: number): number {
    return Math.random() * (max - min) + min;
  }

  /**
   * Returns a random integer number between `min` and `max`
   * @param {number} min lower bound
   * @param {number} max upper bound
   */
  public static randi(min: number, max: number): number {
    return Math.floor(Utils.randf(min, max));
  }

  /**
   * Returns a random sample number from a normal distributed set
   * @param {number} mu mean
   * @param {number} std standard deviation
   * @returns {number} random value
   */
  public static randn(mu: number, std: number): number {
    return mu + Utils.gaussRandom() * std;
  }

  // TODO: Static could lead to unwanted behavior in parallel processes
  private static returnV = false;
  private static vVal = 0.0;

  private static gaussRandom(): number {
    if (Utils.returnV) {
      Utils.returnV = false;
      return Utils.vVal;
    }
    const u = 2 * Math.random() - 1;
    const v = 2 * Math.random() - 1;
    const rsq = u * u + v * v;
    if (rsq === 0 || rsq >= 1) { return Utils.gaussRandom(); }
    const c = Math.sqrt(-2 * Math.log(rsq) / rsq);
    Utils.vVal = v * c; // cache this
    Utils.returnV = true;
    return u * c;
  }

  /**
   * Fills the given array with normal distributed random values.
   * @param arr Array to be filled
   * @param mu mean
   * @param std standard deviation
   * @returns {void} void
   */
  public static fillRandn(arr: Array<number> | Float64Array, mu: number, std: number): void {
    for (let i = 0; i < arr.length; i++) { arr[i] = Utils.randn(mu, std); }
  }

  /**
   * Fills the given array with pseudo-random values between `min` and `max`.
   * @param arr Array to be filled
   * @param min lower bound
   * @param max upper bound
   * @returns {void} void
   */
  public static fillRand(arr: Array<number> | Float64Array, min: number, max: number): void {
    for (let i = 0; i < arr.length; i++) { arr[i] = Utils.randf(min, max); }
  }

  /**
   * Fills the pointed array with constant values.
   * @param {Array<number> | Float64Array} arr Array to be filled
   * @param {number} c value
   * @returns {void} void
   */
  public static fillConst(arr: Array<number> | Float64Array, c: number): void {
    for (let i = 0; i < arr.length; i++) { arr[i] = c; }
  }

  /**
   * returns array populated with ones of length n and uses typed arrays if available
   * @param {number} n length of Array
   * @returns {Array<number> | Float64Array} Array
   */
  public static ones(n: number): Array<number> | Float64Array {
    return Utils.fillArray(n, 1);
  }

  /**
   * returns array of zeros of length n and uses typed arrays if available
   * @param {number} n length of Array
   * @returns {Array<number> | Float64Array} Array
   */
  public static zeros(n: number): Array<number> | Float64Array {
    return Utils.fillArray(n, 0);
  }

  private static fillArray(n: number, val: number): Array<number> | Float64Array {
    if (typeof n === 'undefined' || isNaN(n)) { return []; }
    if (typeof ArrayBuffer === 'undefined') {
      const arr = new Array<number>(n);
      Utils.fillConst(arr, val);
      return arr;
    } else {
      const arr = new Float64Array(n);
      Utils.fillConst(arr, val);
      return arr;
    }
  }

  /**
   * Argmax of Array `arr`
   * @param {Array<number> | Float64Array} arr Array of Numbers
   * @returns {number} Index of Argmax Operation
   */
  public static argmax(arr: Array<number> | Float64Array): number {
    let maxValue = arr[0];
    let maxIndex = 0;
    for (let i = 1; i < arr.length; i++) {
      const v = arr[i];
      if (v > maxValue) {
        maxIndex = i;
        maxValue = v;
      }
    }
    return maxIndex;
  }

  /**
   * Returns an index of the weighted sample of Array `arr`
   * @param {Array<number> | Float64Array} arr Array to be sampled
   * @returns {number} 
   */
  public static sampleWeighted(arr: Array<number> | Float64Array): number {
    const r = Math.random();
    let c = 0.0;
    for (let i = 0; i < arr.length; i++) {
      c += arr[i];
      if (c >= r) { return i; }
    }
    
    return 0;
  }

}
