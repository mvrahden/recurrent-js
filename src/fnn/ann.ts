export interface ANN {
  forward(input: Array<number> | Float64Array): Array<number> | Float64Array;
  backward(expectedOutput: Array<number> | Float64Array, alpha?: number): void;
  setTrainability(isTrainable: boolean): void;
  getSquaredLossFor(input: Array<number> | Float64Array, expectedOutput: Array<number> | Float64Array): number;
}
