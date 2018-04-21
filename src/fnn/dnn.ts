import { Mat, Graph, FNNModel, NetOpts } from './..';

export class DNN extends FNNModel {

  /**
   * Generates a Neural Net instance from a pretrained Neural Net JSON.
   * @param {{ hidden: { Wh, bh }, decoder: { Wh, b } }} opt Specs of the Neural Net.
   */
  constructor(opt: { hidden: { Wh, bh }, decoder: { Wh, b } });
  /**
   * Generates a Neural Net with given specs.
   * @param {NetOpts} opt Specs of the Neural Net. [defaults to: needsBackprop = true, mu = 0, std = 0.01]
   */
  constructor(opt: NetOpts);
  constructor(opt: any) {
    super(opt);
  }

  /**
   * Forward pass for a single tick of Neural Network
   * @param state 1D column vector with observations
   * @param graph optional: inject Graph to append Operations
   * @returns Output of type `Mat`
   */
  public forward(state: Mat, graph: Graph): Mat {
    const activations = this.computeHiddenActivations(state, graph);

    const output = this.computeOutput(activations, graph);
    return output;
  }

  protected computeHiddenActivations(state: Mat, graph: Graph): Mat[] {
    const hiddenActivations = new Array<Mat>();
    for (let d = 0; d < this.hiddenUnits.length; d++) {
      const inputVector = d === 0 ? state : hiddenActivations[d - 1];
      const weightedInput = graph.mul(this.model.hidden.Wh[d], inputVector);
      const biasedWeightedInput = graph.add(weightedInput, this.model.hidden.bh[d]);
      const activation = graph.tanh(biasedWeightedInput);
      hiddenActivations.push(activation);
    }
    return hiddenActivations;
  }

  public static toJSON(dnn: DNN): { hidden: { Wh, bh }, decoder: { Wh, b } } {
    const json = { hidden: { Wh: [], bh: [] }, decoder: { Wh: null, b: null } };
    for (let i = 0; i < dnn.model.hidden.Wh.length; i++) {
      json.hidden.Wh[i] = Mat.toJSON(dnn.model.hidden.Wh[i]);
      json.hidden.bh[i] = Mat.toJSON(dnn.model.hidden.bh[i]);
    }
    json.decoder.Wh = Mat.toJSON(dnn.model.decoder.Wh);
    json.decoder.b = Mat.toJSON(dnn.model.decoder.b);
    return json;
  }

  public static fromJSON(json: { hidden: { Wh, bh }, decoder: { Wh, b } }): DNN {
    return new DNN(json);
  }
}
