export interface NetOpts {
  architecture: {
    inputSize: number,
    hiddenUnits: Array<number>,
    outputSize: number
  };
  training?: {
    alpha?: number,
    loss?: Array<number> | number
  };
  other?: {
    mu?: number;
    std?: number;
  };
}
