import { Mat, Graph, DNN, Utils } from '.';

export class BNN extends DNN {

  private readonly hiddenStd: Array<Mat>;

  /**
   * Generates a Neural Net instance from a pretrained Neural Net JSON.
   * @param {{ hidden: { Wh, bh }, decoder: { Wh, b }}} opt Specs of the Neural Net.
   */
  constructor(opt: { hidden: { Wh, bh }, decoder: { Wh, b } });
  /**
   * Generates a Neural Net with given specs.
   * @param {{inputSize: number, hiddenSize: Array<number>, outputSize: number, mu: number = 0, std: number = 0.01}} opt Specs of the Neural Net.
   */
  constructor(opt: { inputSize: number, hiddenUnits: Array<number>, outputSize: number, needsBackprop?: boolean, mu?: number, std?: number });
  constructor(opt: any) {
    super(opt);
    this.hiddenStd = new Array<Mat>(this.hiddenUnits.length);

    for(let i = 0; i < this.hiddenUnits.length; i++) {
      this.hiddenStd[i] = new Mat(this.hiddenUnits[i], 1);
    }
    this.initializeStaticStd();
  }

  /**
   * Assign a STD per hidden Unit per Layer
   */
  private initializeStaticStd() {
    for (let i = 0; i < this.hiddenUnits.length; i++) {
      Utils.fillRand(this.hiddenStd[i].w, 0, 0.3);
    }
  }

  /**
   * Forward pass for a single tick of Neural Network
   * @param state 1D column vector with observations
   * @param graph optional: inject Graph to append Operations
   * @returns Output of type `Mat`
   */
  public forward(state: Mat, graph: Graph): Mat {
    const activations = this.computeHiddenActivations(state, graph);

    // Add random normal distributed noise to activations
    for(let i = 0; i < this.hiddenUnits.length; i++) {
      activations[i] = Mat.gauss(activations[i], this.hiddenStd[i]);
    }

    const output = this.computeOutput(activations, graph);
    return output;
  }
}
