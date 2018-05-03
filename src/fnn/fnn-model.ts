import { Graph, Mat, RandMat, NetOpts } from './..';
import { Assertable } from './../utils/assertable';

export abstract class FNNModel extends Assertable {
  protected inputSize: number;
  protected hiddenUnits: Array<number>;
  protected outputSize: number;

  public readonly model: {hidden: {Wh: Mat[], bh: Mat[]}, decoder: {Wh: Mat, b: Mat}};
  protected readonly graph: Graph;
  
  /**
   * Generates a Neural Net instance from a pretrained Neural Net JSON.
   * @param {{ hidden: { Wh, bh }, decoder: { Wh: Mat, b: Mat } }} opt Specs of the Neural Net.
   */
  constructor(opt: { hidden: { Wh, bh }, decoder: { Wh: Mat, b: Mat } });
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
      FNNModel.assert(false, 'Improper input for DNN.');
    }
  }

  protected initializeModelFromJSONObject(opt: { hidden: { Wh, bh }, decoder: { Wh, b } }) {
    this.initializeHiddenLayerFromJSON(opt);
    this.model.decoder.Wh = Mat.fromJSON(opt['decoder']['Wh']);
    this.model.decoder.b = Mat.fromJSON(opt['decoder']['b']);
  }

  protected initializeHiddenLayerFromJSON(opt: { hidden: { Wh: any; bh: any; }; decoder: { Wh: any; b: any; }; }): void {
    FNNModel.assert(!Array.isArray(opt['hidden']['Wh']), 'Wrong JSON Format to recreat Hidden Layer.');
    for (let i = 0; i < opt.hidden.Wh.length; i++) {
      this.model.hidden.Wh[i] = Mat.fromJSON(opt.hidden.Wh[i]);
      this.model.hidden.bh[i] = Mat.fromJSON(opt.hidden.bh[i]);
    }
  }

  /**
   * Updates all weights depending on their specific gradients
   * @param alpha discount factor for weight updates
   */
  public update(alpha: number): void {
    this.updateHiddenUnits(alpha);
    this.updateDecoder(alpha);
  }

  private updateHiddenUnits(alpha: number) {
    for (let i = 0; i < this.hiddenUnits.length; i++) {
      this.model.hidden.Wh[i].update(alpha);
      this.model.hidden.bh[i].update(alpha);
    }
  }

  private updateDecoder(alpha: number) {
    this.model.decoder.Wh.update(alpha);
    this.model.decoder.b.update(alpha);
  }

  /**
   * Forward pass for a single tick of Neural Network
   * @param state 1D column vector with observations
   * @param graph optional: inject Graph to append Operations
   * @returns Output of type `Mat`
   */
  public abstract forward(state: Mat, graph?: Graph): Mat;

  private initializeModelAsFreshInstance(opt: { inputSize: number; hiddenUnits: Array<number>; outputSize: number; mu?: number; std?: number; }) {
    this.inputSize = opt.inputSize;
    this.hiddenUnits = opt.hiddenUnits;
    this.outputSize = opt.outputSize;
    
    const mu = opt['mu'] ? opt['mu'] : 0;
    const std = opt['std'] ? opt['std'] : 0.01;

    this.initializeHiddenLayer(mu, std);

    this.initializeDecoder(mu, std);
  }

  private initializeDecoder(mu: number, std: number) {
    this.model.decoder.Wh = new RandMat(this.outputSize, this.hiddenUnits[this.hiddenUnits.length - 1], mu, std);
    this.model.decoder.b = new Mat(this.outputSize, 1);
  }

  private initializeHiddenLayer(mu: number, std: number) {
    let hiddenSize;
    for (let d = 0; d < this.hiddenUnits.length; d++) {
      const previousSize = d === 0 ? this.inputSize : this.hiddenUnits[d - 1];
      hiddenSize = this.hiddenUnits[d];
      this.model.hidden.Wh[d] = new RandMat(hiddenSize, previousSize, mu, std);
      this.model.hidden.bh[d] = new Mat(hiddenSize, 1);
    }
  }

  private isFromJSON(opt: any) {
    return FNNModel.has(opt, ['hidden', 'decoder'])
      && FNNModel.has(opt.hidden, ['Wh', 'bh'])
      && FNNModel.has(opt.decoder, ['Wh', 'b']);
  }

  private isFreshInstanceCall(opt: { inputSize: number; hiddenUnits: Array<number>; outputSize: number; mu?: number; std?: number; }) {
    return FNNModel.has(opt, ['inputSize', 'hiddenUnits', 'outputSize']);
  }

  private initializeNetworkModel(): { hidden: { Wh: Mat[]; bh: Mat[]; }; decoder: { Wh: Mat; b: Mat; }; } {
    return {
      hidden: {
        Wh: new Array<Mat>(this.hiddenUnits.length),
        bh: new Array<Mat>(this.hiddenUnits.length)
      },
      decoder: {
        Wh: null,
        b: null
      } };
  }

  protected computeOutput(hiddenUnitActivations: Mat[], graph: Graph): Mat {
    const weightedInputs = graph.mul(this.model.decoder.Wh, hiddenUnitActivations[hiddenUnitActivations.length - 1]);
    return graph.add(weightedInputs, this.model.decoder.b);
  }

  private static has(obj: any, keys: Array<string>) {
    FNNModel.assert(obj, 'Improper input for DNN.');
    for (const key of keys) {
      if (Object.hasOwnProperty.call(obj, key)) { continue; }
      return false;
    }
    return true;
  }
}
