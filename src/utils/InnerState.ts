import { Mat } from '../Mat';

/**
 * State of inner activations
 */
export interface InnerState {
  hiddenUnits: Array<Mat>;
  output: Mat;
  cells?: Array<Mat>;
}
