import { Mat } from './Mat';
import { RandMat } from './RandMat';
import { Graph } from './Graph';

export class Net {
  public readonly W1: Mat | null = null;
  public readonly b1: Mat | null = null;
  public readonly W2: Mat | null = null;
  public readonly b2: Mat | null = null;

  /**
   * Generates a `null`-initialized Neural Net.
   */
  constructor();
  /**
   * Generates a Neural Net instance from a pretrained Neural Net JSON.
   * @param {{W1, b1, W2, b2}} opt Specs of the Neural Net.
   */
  constructor(opt: any);
  /**
   * Generates a Neural Net with given specs.
   * @param {{inputSize: number, hiddenSize: number, outputSize: number, mu: number = 0, std: number = 0.01}} opt Specs of the Neural Net.
   */
  constructor(opt?: {inputSize: number, hiddenSize: number, outputSize: number, mu?: number, std?: number}) {
    if (Net.has(opt, ['W1', 'b1', 'W2', 'b2'])) {
      this.W1 = Mat.fromJSON(opt['W1']);
      this.b1 = Mat.fromJSON(opt['b1']);
      this.W2 = Mat.fromJSON(opt['W2']);
      this.b2 = Mat.fromJSON(opt['b2']);
    } else if (Net.has(opt, ['inputSize', 'hiddenSize', 'outputSize'])) {
      const mu = opt['mu'] ? opt['mu'] : 0;
      const std = opt['std'] ? opt['std'] : 0.01;
      this.W1 = new RandMat(opt['hiddenSize'], opt['inputSize'], mu, std);
      this.b1 = new Mat(opt['hiddenSize'], 1);
      this.W2 = new RandMat(opt['outputSize'], opt['hiddenSize'], mu, std);
      this.b2 = new Mat(opt['outputSize'], 1);
    }
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
   * Forward propagation for a single tick of RNN
   * @param observations 1D column vector with observations
   * @param graph optional: inject Graph to append Operations
   * @returns Structure containing hidden representation ['h'] of type `Mat[]` and output ['output'] of type `Mat`
   */
  public forward(observations: Mat, graph: Graph): Mat {
    // TODO: Is this a Deep Net? Hyperbolic activation h1 is a Vector of size hiddenUnits (= each Neuron is one Layer???)
    const weightedStates = graph.mul(this.W1, observations);
    // a1 = Activation Input Vector of Weights 1 (W1) and stateVector
    const a1mat = graph.add(weightedStates, this.b1);
    // h1 = Hyperbolic activation
    const h1mat = graph.tanh(a1mat);

    const weightedActivations = graph.mul(this.W2, h1mat);
    // a2 = Action Vector of Weight2 (W2) and hyperbolic Activation (h1)
    const a2Mat = graph.add(weightedActivations, this.b2);
    // TODO: Hyperbolic activation of a2Mat
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
