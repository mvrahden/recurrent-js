import { Mat } from './Mat';
import { RandMat } from './RandMat';

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
    const j = {};
    for (const property in net) {
      if (net.hasOwnProperty(property)) {
        j[property] = net[property].toJSON();
      }
    }
    return j;
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
