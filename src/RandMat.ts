import { Mat } from './Mat';
import { Utils } from './Utils';

export class RandMat extends Mat {

  /**
   * 
   * @param n length of Matrix
   * @param d depth of Matrix
   * @param mu Population mean for initialization
   * @param std Standard deviation for initialization
   */
  constructor(n: number, d: number, mu: number, std: number) {
    super(n, d);
    Utils.fillRandn(this.w, mu, std);
  }
}
