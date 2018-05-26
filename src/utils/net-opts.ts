export interface NetOpts {
  architecture: {
    inputSize: number,
    hiddenUnits: Array<number>,
    outputSize: number
  };
  other?: {
    mu?: number;
    std?: number;
  };
}
