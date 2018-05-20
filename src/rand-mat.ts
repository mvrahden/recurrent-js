import { Mat, Utils } from '.';

export class RandMat extends Mat {

  /**
   * 
   * @param rows length of Matrix
   * @param cols depth of Matrix
   * @param mu Population mean for initialization
   * @param std Standard deviation for initialization
   */
  constructor(rows: number, cols: number, mu: number, std: number) {
    super(rows, cols);
    Utils.fillRandn(this.w, mu, std);
  }
}
