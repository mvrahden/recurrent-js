export interface NetOpts {
  inputSize: number;
  hiddenUnits: Array<number>;
  outputSize: number;
  needsBackprop?: boolean;
  mu?: number;
  std?: number;
}
