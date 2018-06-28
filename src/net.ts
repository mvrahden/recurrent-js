import { Mat, RandMat, Graph, NetOpts } from '.';

export class Net {
  public W1: Mat;
  public b1: Mat;
  public W2: Mat;
  public b2: Mat;

  /**
   * Generates a Neural Net instance from a pre-trained Neural Net JSON.
   * @param {{W1, b1, W2, b2}} opt Specs of the Neural Net.
   */
  constructor(opt: { W1, b1, W2, b2 });
  /**
   * Generates a Neural Net with given specs.
   * @param {NetOpts} opt Specs of the Neural Net.
   */
  constructor(opt: NetOpts);
  constructor(opt: any) {
    if (this.isFromJSON(opt)) {
      this.initializeFromJSONObject(opt);
    } else if (this.isFreshInstanceCall(opt)) {
      this.initializeAsFreshInstance(opt);
    } else {
      this.initializeAsFreshInstance({ architecture: { inputSize: 1, hiddenUnits: [1], outputSize: 1 } });
    }
  }

  private isFromJSON(opt: any) {
    return Net.has(opt, ['W1', 'b1', 'W2', 'b2']);
  }

  private isFreshInstanceCall(opt: NetOpts) {
    return Net.has(opt, ['architecture']) && Net.has(opt.architecture, ['inputSize', 'hiddenUnits', 'outputSize']);
  }

  private initializeFromJSONObject(opt: { W1, b1, W2, b2 }) {
    this.W1 = Mat.fromJSON(opt['W1']);
    this.b1 = Mat.fromJSON(opt['b1']);
    this.W2 = Mat.fromJSON(opt['W2']);
    this.b2 = Mat.fromJSON(opt['b2']);
  }

  private initializeAsFreshInstance(opt: NetOpts) {
    let mu = 0;
    let std = 0.01;
    if(Net.has(opt, ['other'])) {
      mu = opt.other['mu'] ? opt.other['mu'] : mu;
      std = opt.other['std'] ? opt.other['std'] : std;
    }
    const firstLayer = 0; // only consider the first layer => shallowness
    this.W1 = new RandMat(opt.architecture['hiddenUnits'][firstLayer], opt.architecture['inputSize'], mu, std);
    this.b1 = new Mat(opt.architecture['hiddenUnits'][firstLayer], 1);
    this.W2 = new RandMat(opt.architecture['outputSize'], opt.architecture['hiddenUnits'][firstLayer], mu, std);
    this.b2 = new Mat(opt.architecture['outputSize'], 1);
  }

  /**
   * Updates all weights
   * @param alpha discount factor for weight updates
   */
  public update(alpha: number): void {
    this.W1.update(alpha);
    this.b1.update(alpha);
    this.W2.update(alpha);
    this.b2.update(alpha);
  }

  public static toJSON(net: Net): {} {
    const json = {};
    json['W1'] = Mat.toJSON(net.W1);
    json['b1'] = Mat.toJSON(net.b1);
    json['W2'] = Mat.toJSON(net.W2);
    json['b2'] = Mat.toJSON(net.b2);
    return json;
  }

  /**
   * Compute forward pass of Neural Network
   * @param state 1D column vector with observations
   * @param graph optional: inject Graph to append Operations
   * @returns output of type `Mat`
   */
  public forward(state: Mat, graph: Graph): Mat {
    const weightedInput = graph.mul(this.W1, state);

    const a1mat = graph.add(weightedInput, this.b1);

    const h1mat = graph.tanh(a1mat);

    const a2Mat = this.computeOutput(h1mat, graph);
    return a2Mat;
  }

  private computeOutput(hiddenUnits: Mat, graph: Graph) {
    const weightedActivation = graph.mul(this.W2, hiddenUnits);
    // a2 = Output Vector of Weight2 (W2) and hyperbolic Activation (h1)
    const a2Mat = graph.add(weightedActivation, this.b2);
    return a2Mat;
  }

  public static fromJSON(json: { W1, b1, W2, b2 }): Net {
    return new Net(json);
  }

  private static has(obj: any, keys: Array<string>) {
    for (const key of keys) {
      if (Object.hasOwnProperty.call(obj, key)) { continue; }
      return false;
    }
    return true;
  }
}
