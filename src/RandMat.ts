import { Mat } from "./Mat";
import { R } from "./R";

/**
 * `Mat` prefilled with gaussian distributed random numbers.
 */
export class RandMat extends Mat {
  
  constructor (n:number, d:number, mu:number, std:number) {
    super(n, d);
    R.fillRandn(this, mu, std);
  }
}
