export interface NetOpts {
  architecture: {
    inputSize: number,
    hiddenUnits: Array<number>,
    outputSize: number
  };
  training?: {
    alpha?: number,
    lossClamp?: number,
    loss?: number
  };
  other?: {
    mu?: number;
    std?: number;
  };
}
