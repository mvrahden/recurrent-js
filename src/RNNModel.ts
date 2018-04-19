import { Graph } from './Graph';
import { Mat } from './Mat';
import { InnerState } from './utils/InnerState';
import { RandMat } from './RandMat';
import { Assertable } from './utils/Assertable';

export abstract class RNNModel extends Assertable {

  protected inputSize: number;
  protected hiddenUnits: Array<number>;
  protected outputSize: number;

  public readonly model: { hidden: any, decoder: { Wh: Mat, b: Mat } };
  protected readonly graph: Graph;

  /**
   * Generates a Neural Net instance from a pretrained Neural Net JSON.
   * @param {{ hidden: any, decoder: { Wh: Mat, b: Mat } }} opt Specs of the Neural Net.
   */
  constructor(opt: { hidden: any, decoder: { Wh: Mat, b: Mat } });
  /**
   * Generates a Neural Net with given specs.
   * @param {{inputSize: number, hiddenSize: Array<number>, outputSize: number, needsBackprop: boolean = true, mu: number = 0, std: number = 0.01}} opt Specs of the Neural Net.
   */
  constructor(opt: { inputSize: number, hiddenUnits: Array<number>, outputSize: number, needsBackprop?: boolean, mu?: number, std?: number });
  constructor(opt: any) {
    super();
    const needsBackprop = opt && opt.needsBackprop ? opt.needsBackprop : true;

    this.model = this.initializeNetworkModel();
    this.graph = new Graph(needsBackprop);

    if (this.isFromJSON(opt)) {
      this.initializeModelFromJSONObject(opt);
    } else if (this.isFreshInstanceCall(opt)) {
      this.initializeModelAsFreshInstance(opt);
    } else {
      RNNModel.assert(false, 'Improper input for DNN.');
    }
  }

  protected abstract initializeNetworkModel(): { hidden: any; decoder: { Wh: Mat; b: Mat; }; };

  protected abstract isFromJSON(opt: any);

  protected initializeModelFromJSONObject(opt: { hidden: any, decoder: { Wh: Mat, b: Mat } }): void {
    this.initializeHiddenLayerFromJSON(opt);
    this.model.decoder.Wh = Mat.fromJSON(opt['decoder']['Wh']);
    this.model.decoder.b = Mat.fromJSON(opt['decoder']['b']);
  }
  
  protected abstract initializeHiddenLayerFromJSON(opt: { hidden: any, decoder: { Wh: Mat, b: Mat } }): void;

  private isFreshInstanceCall(opt: { inputSize: number; hiddenUnits: Array<number>; outputSize: number; mu?: number; std?: number; }) {
    return RNNModel.has(opt, ['inputSize', 'hiddenUnits', 'outputSize']);
  }

  private initializeModelAsFreshInstance(opt: { inputSize: number; hiddenUnits: Array<number>; outputSize: number; mu?: number; std?: number; }) {
    this.inputSize = opt.inputSize;
    this.hiddenUnits = opt.hiddenUnits;
    this.outputSize = opt.outputSize;

    const mu = opt['mu'] ? opt['mu'] : 0;
    const std = opt['std'] ? opt['std'] : 0.01;

    this.initializeHiddenLayer(mu, std);

    this.initializeDecoder(mu, std);
  }

  protected abstract initializeHiddenLayer(mu: number, std: number): void;

  protected initializeDecoder(mu: number, std: number): void {
    this.model.decoder.Wh = new RandMat(this.outputSize, (this.hiddenUnits.length - 1), mu, std);
    this.model.decoder.b = new Mat(this.outputSize, 1);
  }

  public abstract forward(state: Mat, previousInnerState: InnerState, graph?: Graph): InnerState;

  protected computeOutput(hiddenUnitActivations: Mat[], graph: Graph): Mat {
    const weightedInputs = graph.mul(this.model.decoder.Wh, hiddenUnitActivations[hiddenUnitActivations.length - 1]);
    return graph.add(weightedInputs, this.model.decoder.b);
  }

  protected static has(obj: any, keys: Array<string>) {
    RNNModel.assert(obj, 'Improper input for DNN.');
    for (const key of keys) {
      if (Object.hasOwnProperty.call(obj, key)) { continue; }
      return false;
    }
    return true;
  }
}
