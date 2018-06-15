export interface ANN {
  forward(input: Array<number> | Float64Array): Array<number> | Float64Array;
  backward(expectedOutput: Array<number> | Float64Array, alpha?: number): number;
  setTrainability(isTrainable: boolean): void;
}
