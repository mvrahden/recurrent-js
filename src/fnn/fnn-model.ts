import { Graph, Mat, RandMat, NetOpts, Utils } from './..';
import { Assertable } from './../utils/assertable';

export abstract class FNNModel extends Assertable {

  protected architecture: { inputSize: number, hiddenUnits: Array<number>, outputSize: number };
  protected training: { alpha: number, loss: Mat };

  public model: { hidden: { Wh: Mat[], bh: Mat[] }, decoder: { Wh: Mat, b: Mat } } = { hidden: { Wh: [], bh: [] }, decoder: { Wh: null, b: null } };

  protected graph: Graph;

  /**
   * Generates a Neural Net instance from a pre-trained Neural Net JSON.
   * @param {{ hidden: { Wh, bh }, decoder: { Wh: Mat, b: Mat } }} opt Specs of the Neural Net.
   */
  constructor(opt: { hidden: { Wh, bh }, decoder: { Wh: Mat, b: Mat } });
  /**
   * Generates a Neural Net with given specs.
   * @param {NetOpts} opt Specs of the Neural Net. [defaults to: needsBackprop = false, mu = 0, std = 0.01]
   */
  constructor(opt: NetOpts);
  constructor(opt: any) {
    super();

    const needsBackpropagation = opt && opt.needsBackpropagation ? opt.needsBackpropagation : false;
    this.graph = new Graph();
    this.graph.memorizeOperationSequence(true);

    if (FNNModel.isFromJSON(opt)) {
      this.initializeModelFromJSONObject(opt);
    } else if (FNNModel.isFreshInstanceCall(opt)) {
      this.initializeModelAsFreshInstance(opt);
    } else {
      FNNModel.assert(false, 'Improper input for DNN.');
    }
  }

  private static isFromJSON(opt: any) {
    return FNNModel.has(opt, ['hidden', 'decoder'])
      && FNNModel.has(opt.hidden, ['Wh', 'bh'])
      && FNNModel.has(opt.decoder, ['Wh', 'b']);
  }

  protected initializeModelFromJSONObject(opt: { hidden: { Wh, bh }, decoder: { Wh, b } }) {
    this.initializeHiddenLayerFromJSON(opt);
    this.model.decoder.Wh = Mat.fromJSON(opt['decoder']['Wh']);
    this.model.decoder.b = Mat.fromJSON(opt['decoder']['b']);
  }

  protected initializeHiddenLayerFromJSON(opt: { hidden: { Wh: Mat[]; bh: Mat[]; }; decoder: { Wh: Mat; b: Mat; }; }): void {
    FNNModel.assert(!Array.isArray(opt['hidden']['Wh']), 'Wrong JSON Format to recreate Hidden Layer.');
    for (let i = 0; i < opt.hidden.Wh.length; i++) {
      this.model.hidden.Wh[i] = Mat.fromJSON(opt.hidden.Wh[i]);
      this.model.hidden.bh[i] = Mat.fromJSON(opt.hidden.bh[i]);
    }
  }

  private static isFreshInstanceCall(opt: any) {
    return FNNModel.has(opt, ['architecture']) && FNNModel.has(opt.architecture, ['inputSize', 'hiddenUnits', 'outputSize']);
  }

  private initializeModelAsFreshInstance(opt: NetOpts) {
    this.architecture = this.determineArchitecture(opt);
    this.training = this.determineTraining(opt);

    const mu = opt['mu'] ? opt['mu'] : 0;
    const std = opt['std'] ? opt['std'] : 0.01;

    this.model = this.initializeFreshNetworkModel();

    this.initializeHiddenLayer(mu, std);

    this.initializeDecoder(mu, std);
  }

  protected determineArchitecture(opt: NetOpts): { inputSize: number, hiddenUnits: Array<number>, outputSize: number } {
    const out = { inputSize: null, hiddenUnits: null, outputSize: null };
    out.inputSize = typeof opt.architecture.inputSize === 'number' ? opt.architecture.inputSize : 1;
    out.hiddenUnits = Array.isArray(opt.architecture.hiddenUnits) ? opt.architecture.hiddenUnits : [1];
    out.outputSize = typeof opt.architecture.outputSize === 'number' ? opt.architecture.outputSize : 1;
    return out;
  }

  protected determineTraining(opt: NetOpts): { alpha: number, loss: Mat } {
    const out = { alpha: null, loss: null };
    if(!opt.training) {
      // patch `opt`
      opt.training = { alpha: null, loss: null };
    }

    out.alpha = typeof opt.training.alpha === 'number' ? opt.training.alpha : 0.01;
    out.loss = new Mat(1, this.architecture.outputSize);

    if(Array.isArray(opt.training.loss) && opt.training.loss.length === this.architecture.outputSize) {
      out.loss.setFrom(opt.training.loss);
    } else if (typeof opt.training.loss === 'number') {
      Utils.fillConst(out.loss.w, opt.training.loss);
    } else {
      Utils.fillConst(out.loss.w, 1e-6);
    }

    return out;
  }

  protected initializeFreshNetworkModel(): { hidden: { Wh: Mat[]; bh: Mat[]; }; decoder: { Wh: Mat; b: Mat; }; } {
    return {
      hidden: {
        Wh: new Array<Mat>(this.architecture.hiddenUnits.length),
        bh: new Array<Mat>(this.architecture.hiddenUnits.length)
      },
      decoder: {
        Wh: null,
        b: null
      }
    };
  }

  protected initializeHiddenLayer(mu: number, std: number): void {
    let hiddenSize;
    for (let i = 0; i < this.architecture.hiddenUnits.length; i++) {
      const previousSize = i === 0 ? this.architecture.inputSize : this.architecture.hiddenUnits[i - 1];
      hiddenSize = this.architecture.hiddenUnits[i];
      this.model.hidden.Wh[i] = new RandMat(hiddenSize, previousSize, mu, std);
      this.model.hidden.bh[i] = new Mat(hiddenSize, 1);
    }
  }

  protected initializeDecoder(mu: number, std: number): void {
    this.model.decoder.Wh = new RandMat(this.architecture.outputSize, this.architecture.hiddenUnits[this.architecture.hiddenUnits.length - 1], mu, std);
    this.model.decoder.b = new Mat(this.architecture.outputSize, 1);
  }

  /**
   * Updates all weights depending on their specific gradients
   * @param alpha discount factor for weight updates
   * @returns {void}
   */
  public update(alpha: number): void {
    this.updateHiddenUnits(alpha);
    this.updateDecoder(alpha);
  }

  private updateHiddenUnits(alpha: number): void {
    for (let i = 0; i < this.architecture.hiddenUnits.length; i++) {
      this.model.hidden.Wh[i].update(alpha);
      this.model.hidden.bh[i].update(alpha);
    }
  }

  private updateDecoder(alpha: number): void {
    this.model.decoder.Wh.update(alpha);
    this.model.decoder.b.update(alpha);
  }

  /**
   * Compute forward pass of Neural Network
   * @param input 1D column vector with observations
   * @param graph optional: inject Graph to append Operations
   * @returns Output of type `Mat`
   */
  public forward(input: Array<number> | Float64Array): Array<number> | Float64Array {
    const mat = this.transformArrayToMat(input);
    const activations = this.specificForwardpass(mat);
    const outputMat = this.computeOutput(activations);
    const output = this.transformMatToArray(outputMat);
    return output;
  }

  private transformArrayToMat(input: Array<number> | Float64Array): Mat {
    const mat = new Mat(this.architecture.inputSize, 1);
    mat.setFrom(input);
    return mat;
  }

  private transformMatToArray(input: Mat): Array<number> | Float64Array {
    const arr = input.w.slice(0);
    return arr;
  }

  protected abstract specificForwardpass(state: Mat): Mat[];

  protected computeOutput(hiddenUnitActivations: Mat[]): Mat {
    const weightedInputs = this.graph.mul(this.model.decoder.Wh, hiddenUnitActivations[hiddenUnitActivations.length - 1]);
    return this.graph.add(weightedInputs, this.model.decoder.b);
  }

  private static has(obj: any, keys: Array<string>): boolean {
    FNNModel.assert(obj, 'Improper input for DNN.');
    for (const key of keys) {
      if (Object.hasOwnProperty.call(obj, key)) { continue; }
      return false;
    }
    return true;
  }
}
