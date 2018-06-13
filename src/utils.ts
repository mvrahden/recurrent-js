import { Mat } from '.';

export class Utils {
  /**
   * Returns a random floating point number of a uniform distribution between `min` and `max`
   * @param {number} min lower bound
   * @param {number} max upper bound
   * @returns {number} random float value
   */
  public static randf(min: number, max: number): number {
    return Math.random() * (max - min) + min;
  }

  /**
   * Returns a random integer number of a uniform distribution between [`min`, `max`)
   * @param {number} min lower bound
   * @param {number} max upper bound
   * @returns {number} random integer value
   */
  public static randi(min: number, max: number): number {
    return Math.floor(Utils.randf(min, max));
  }

  /**
   * Returns a sample of a normal distribution
   * @param {number} mu mean
   * @param {number} std standard deviation
   * @returns {number} random value
   */
  public static randn(mu: number, std: number): number {
    return mu + Utils.gaussRandom() * std;
  }

  /**
   * Returns a random sample number from a normal distributed set
   * @param {number} min lower bound
   * @param {number} max upper bound
   * @param {number} skew factor of skewness; < 1 shifts to the right; > 1 shifts to the left
   * @returns {number} random value
   */
  public static skewedRandn(mu: number, std: number, skew: number): number {
    let sample = Utils.box_muller();
    sample = Math.pow(sample, skew);
    sample = (sample - 0.5) * 10;
    return mu + sample * std;
  }

  /**
   * Gaussian-distributed sample from a normal distributed set.
   */
  private static gaussRandom(): number {
    return (Utils.box_muller() - 0.5) * 10;
  }

  /**
   * Box-Muller Transform, to transform uniform random values into standard gaussian distributed random values.
   * @returns random value between of interval (0,1)
   */
  private static box_muller(): number {
    // Based on:
    // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    // and
    // https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
    let z0 = 0, u1 = 0, u2 = 0;
    do {
      u1 = u2 = 0;
      // Convert interval from [0,1) to (0,1)
      do { u1 = Math.random(); } while (u1 === 0);
      do { u2 = Math.random(); } while (u2 === 0);
      z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      z0 = z0 / 10.0 + 0.5;
    } while (z0 > 1 || z0 < 0); // resample c
    return z0;
  }

  /**
   * Calculates the sum of a given set
   * @param arr randomly populated array of numbers
   */
  public static sum(arr: Array<number> | Float64Array): number {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      sum += arr[i];
    }
    return sum;
  }

  /**
   * Calculates the mean of a given set
   * @param arr set of values
   */
  public static mean(arr: Array<number> | Float64Array): number {
    // mean of [3, 5, 4, 4, 1, 1, 2, 3] is 2.875
    let count = arr.length;
    let sum = Utils.sum(arr);
    return sum / count;
  }

  /**
   * Calculates the median of a given set
   * @param arr set of values
   */
  public static median(arr: Array<number>): number {
    // median of [3, 5, 4, 4, 1, 1, 2, 3] = 3
    let median = 0;
    let count = arr.length;
    arr.sort();
    if (count % 2 === 0) { // is even
      // average of two middle numbers
      median = (arr[count / 2 - 1] + arr[count / 2]) / 2;
    } else { // is odd
      // middle number only
      median = arr[(count - 1) / 2];
    }
    return median;
  }

  /**
   * Calculates the standard deviation of a given set
   * @param arr set of values
   * @param precision the floating point precision for grouping results, e.g. 1e3 [defaults to 1e6]
   */
  public static mode(arr: Array<number>, precision?: number): Array<number> {
    // as result can be bimodal or multimodal,
    // the returned result is provided as an array
    // mode of [3, 5, 4, 4, 1, 1, 2, 3] = [1, 3, 4]
    precision = precision ? precision : 1e6;
    let modes = [],
      count = [],
      number,
      maxCount = 0;
    // populate array with number counts
    for (let i = 0; i < arr.length; i++) {
      number = Math.round(arr[i] * precision) / precision;
      count[number] = (count[number] || 0) + 1; // initialize or increment for number
      if (count[number] > maxCount) {
        maxCount = count[number]; // memorize count value of max index
      }
    }
    // memorize numbers equal with maxCount
    for (let i in count) {
      if (count.hasOwnProperty(i)) {
        if (count[i] === maxCount) {
          modes.push(Number(i));
        }
      }
    }
    return modes;
  }

  /**
   * Calculates the population variance (uncorrected), the sample variance (unbiased) or biased variance of a given set
   * @param arr set of values
   * @param normalization defaults to sample variance ('unbiased')
   */
  public static var(arr: Array<number>, normalization?: 'uncorrected' | 'biased' | 'unbiased'): number {
    normalization = normalization ? normalization : 'unbiased';
    let count = arr.length;

    // calculate the variance
    let mean = Utils.mean(arr);
    let sum = 0;
    let diff = 0;
    for (let i = 0; i < arr.length; i++) {
      diff = arr[i] - mean;
      sum += diff * diff;
    }

    switch (normalization) {
      case 'uncorrected':
        return sum / count;

      case 'biased':
        return sum / (count + 1);

      case 'unbiased':
        return (count == 1) ? 0 : sum / (count - 1);
    }
  }

  /**
   * Calculates the standard deviation of a given set
   * @param arr set of values
   * @param normalization defaults to sample variance ('unbiased')
   */
  public static std = (arr: Array<number>, normalization?: 'uncorrected' | 'biased' | 'unbiased'): number => {
    return Math.sqrt(Utils.var(arr, normalization));
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
   * Fills the given array with uniformly distributed random values between `min` and `max`.
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
