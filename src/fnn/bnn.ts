import { Mat, Utils, NetOpts } from './..';
import { DNN } from './dnn';

export class BNN extends DNN {

  private hiddenStd: Array<Mat>;

  /**
   * Generates a Neural Net instance from a pre-trained Neural Net JSON.
   * @param {{ hidden: { Wh, bh }, decoder: { Wh, b }}} opt Specs of the Neural Net.
   */
  constructor(opt: { hidden: { Wh, bh }, decoder: { Wh, b } });
  /**
   * Generates a Neural Net with given specs.
   * @param {NetOpts} opt Specs of the Neural Net.  [defaults to: needsBackprop = false, mu = 0, std = 0.01]
   */
  constructor(opt: NetOpts);
  constructor(opt: any) {
    super(opt);
  }

  protected initializeModelAsFreshInstance(opt: NetOpts) {
    super.initializeModelAsFreshInstance(opt);
    this.initializeHiddenLayerStds(opt);
  }

  /**
   * Assign a STD per hidden Unit per Layer
   */
  private initializeHiddenLayerStds(opt: NetOpts) {
    this.hiddenStd = new Array<Mat>(this.architecture.hiddenUnits.length);

    for (let i = 0; i < this.architecture.hiddenUnits.length; i++) {
      this.hiddenStd[i] = new Mat(this.architecture.hiddenUnits[i], 1);
    }

    for (let i = 0; i < this.architecture.hiddenUnits.length; i++) {
      Utils.fillRandn(this.hiddenStd[i].w, 0, 0.001);
    }
  }

  /**
   * Compute forward pass of Neural Network
   * @param state 1D column vector with observations
   * @param graph optional: inject Graph to append Operations
   * @returns Output of type `Mat`
   */
  public specificForwardpass(state: Mat): Mat[] {
    const activations = this.computeHiddenActivations(state);

    // Add random normal distributed noise to activations
    for (let i = 0; i < this.architecture.hiddenUnits.length; i++) {
      activations[i] = this.graph.gauss(activations[i], this.hiddenStd[i]);
    }
    return activations;
  }
}
