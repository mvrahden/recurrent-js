export interface NetOpts {
  inputSize: number;
  hiddenUnits: Array<number>;
  outputSize: number;
  needsBackpropagation?: boolean;
  mu?: number;
  std?: number;
}
