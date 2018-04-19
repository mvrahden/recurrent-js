import { Mat } from './Mat';
import { RandMat } from './RandMat';
import { Graph } from './Graph';

export class Net {
  public W1: Mat | null = null;
  public b1: Mat | null = null;
  public W2: Mat | null = null;
  public b2: Mat | null = null;

  /**
   * Generates a default initialized Neural Net.
   * Default Setup:
   * @param {{inputSize: number = 1, hiddenUnits: number = 1, outputSize: number = 1}} opt Specs of the Neural Net.
   */
  constructor();
  /**
   * Generates a Neural Net instance from a pretrained Neural Net JSON.
   * @param {{W1, b1, W2, b2}} opt Specs of the Neural Net.
   */
  constructor(opt: { W1, b1, W2, b2 });
  /**
   * Generates a Neural Net with given specs.
   * @param {{inputSize: number, hiddenSize: number, outputSize: number, mu: number = 0, std: number = 0.01}} opt Specs of the Neural Net.
   */
  constructor(opt: { inputSize: number, hiddenUnits: number, outputSize: number, mu?: number, std?: number });
  constructor(opt?: any) {
    if (this.isFromJSON(opt)) {
      this.initializeFromJSONObject(opt);
    } else if (this.isFreshInstanceCall(opt)) {
      this.initializeAsFreshInstance(opt);
    } else {
      this.initializeAsFreshInstance({inputSize: 1, hiddenUnits: 1, outputSize: 1 });
    }
  }

  private isFromJSON(opt: any) {
    return Net.has(opt, ['W1', 'b1', 'W2', 'b2']);
  }

  private isFreshInstanceCall(opt: { inputSize: number; hiddenUnits: number; outputSize: number; mu?: number; std?: number; }) {
    return Net.has(opt, ['inputSize', 'hiddenUnits', 'outputSize']);
  }

  private initializeFromJSONObject(opt: { W1, b1, W2, b2 }) {
    this.W1 = Mat.fromJSON(opt['W1']);
    this.b1 = Mat.fromJSON(opt['b1']);
    this.W2 = Mat.fromJSON(opt['W2']);
    this.b2 = Mat.fromJSON(opt['b2']);
  }

  private initializeAsFreshInstance(opt: { inputSize: number; hiddenUnits: number; outputSize: number; mu?: number; std?: number; }) {
    const mu = opt['mu'] ? opt['mu'] : 0;
    const std = opt['std'] ? opt['std'] : 0.01;
    this.W1 = new RandMat(opt['hiddenUnits'], opt['inputSize'], mu, std);
    this.b1 = new Mat(opt['hiddenUnits'], 1);
    this.W2 = new RandMat(opt['outputSize'], opt['hiddenUnits'], mu, std);
    this.b2 = new Mat(opt['outputSize'], 1);
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
   * Forward pass for a single tick of Neural Network
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
