import { Graph, Mat, RandMat, NetOpts } from './..';
import { ANN } from './ann';
import { Assertable } from './../utils/assertable';

export abstract class FNNModel extends Assertable implements ANN {

  protected architecture: { inputSize: number, hiddenUnits: Array<number>, outputSize: number };
  protected training: { alpha: number, lossClamp: number, loss: number };

  public model: { hidden: { Wh: Array<Mat>, bh: Array<Mat> }, decoder: { Wh: Mat, b: Mat } } = { hidden: { Wh: [], bh: [] }, decoder: { Wh: null, b: null } };

  protected graph: Graph;
  protected previousOutput: Mat;
  protected previousInput: Array<number> | Float64Array;

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
    this.initializeNeuralNetworkFromGivenOptions(opt);
  }

  private initializeNeuralNetworkFromGivenOptions(opt: any): void {
    this.graph = new Graph();
    if (FNNModel.isFromJSON(opt)) {
      this.initializeModelFromJSONObject(opt);
    }
    else if (FNNModel.isFreshInstanceCall(opt)) {
      this.initializeModelAsFreshInstance(opt);
    }
    else {
      FNNModel.assert(false, 'Improper input for DNN.');
    }
  }

  protected static isFromJSON(opt: any): boolean {
    return FNNModel.has(opt, ['hidden', 'decoder'])
      && FNNModel.has(opt.hidden, ['Wh', 'bh'])
      && FNNModel.has(opt.decoder, ['Wh', 'b']);
  }

  protected initializeModelFromJSONObject(opt: { hidden: { Wh, bh }, decoder: { Wh, b } }): void {
    this.initializeHiddenLayerFromJSON(opt);
    this.model.decoder.Wh = Mat.fromJSON(opt['decoder']['Wh']);
    this.model.decoder.b = Mat.fromJSON(opt['decoder']['b']);
  }

  protected initializeHiddenLayerFromJSON(opt: { hidden: { Wh: Array<Mat>; bh: Array<Mat>; }; decoder: { Wh: Mat; b: Mat; }; }): void {
    FNNModel.assert(!Array.isArray(opt['hidden']['Wh']), 'Wrong JSON Format to recreate Hidden Layer.');
    for (let i = 0; i < opt.hidden.Wh.length; i++) {
      this.model.hidden.Wh[i] = Mat.fromJSON(opt.hidden.Wh[i]);
      this.model.hidden.bh[i] = Mat.fromJSON(opt.hidden.bh[i]);
    }
  }

  protected static isFreshInstanceCall(opt: any): boolean {
    return FNNModel.has(opt, ['architecture']) && FNNModel.has(opt.architecture, ['inputSize', 'hiddenUnits', 'outputSize']);
  }

  protected initializeModelAsFreshInstance(opt: NetOpts): void {
    this.architecture = this.determineArchitectureProperties(opt);
    this.training = this.determineTrainingProperties(opt);

    const mu = opt['mu'] ? opt['mu'] : 0;
    const std = opt['std'] ? opt['std'] : 0.1;

    this.model = this.initializeFreshNetworkModel();

    this.initializeHiddenLayer(mu, std);

    this.initializeDecoder(mu, std);
  }

  protected determineArchitectureProperties(opt: NetOpts): { inputSize: number, hiddenUnits: Array<number>, outputSize: number } {
    const out = { inputSize: null, hiddenUnits: null, outputSize: null };
    out.inputSize = typeof opt.architecture.inputSize === 'number' ? opt.architecture.inputSize : 1;
    out.hiddenUnits = Array.isArray(opt.architecture.hiddenUnits) ? opt.architecture.hiddenUnits : [1];
    out.outputSize = typeof opt.architecture.outputSize === 'number' ? opt.architecture.outputSize : 1;
    return out;
  }

  protected determineTrainingProperties(opt: NetOpts): { alpha: number, lossClamp: number, loss: number } {
    const out = { alpha: null, lossClamp: null, loss: null };
    if (!opt.training) {
      // patch `opt`
      opt.training = out;
    }

    out.alpha = typeof opt.training.alpha === 'number' ? opt.training.alpha : 0.01;
    out.lossClamp = typeof opt.training.lossClamp === 'number' ? opt.training.lossClamp : 1;
    out.loss = typeof opt.training.loss === 'number' ? opt.training.loss : 1e-6;

    return out;
  }

  protected initializeFreshNetworkModel(): { hidden: { Wh: Array<Mat>; bh: Array<Mat>; }; decoder: { Wh: Mat; b: Mat; }; } {
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
      const previousSize = this.getPrecedingLayerSize(i);
      hiddenSize = this.architecture.hiddenUnits[i];
      this.model.hidden.Wh[i] = new RandMat(hiddenSize, previousSize, mu, std);
      this.model.hidden.bh[i] = new Mat(hiddenSize, 1);
    }
  }

  /**
   * According to the given hiddenLayer Index, get the size of preceding layer.
   * @param i current hidden layer index
   */
  private getPrecedingLayerSize(i: number) {
    return i === 0 ? this.architecture.inputSize : this.architecture.hiddenUnits[i - 1];
  }

  protected initializeDecoder(mu: number, std: number): void {
    this.model.decoder.Wh = new RandMat(this.architecture.outputSize, this.architecture.hiddenUnits[this.architecture.hiddenUnits.length - 1], mu, std);
    this.model.decoder.b = new Mat(this.architecture.outputSize, 1);
  }

  /**
   * Sets the neural network into a trainable state.
   * Also cleans the memory of forward pass operations, meaning that the last forward pass cannot be used for backpropagation.
   * @param isTrainable 
   */
  public setTrainability(isTrainable: boolean): void {
    this.graph.forgetCurrentSequence();
    this.graph.memorizeOperationSequence(isTrainable);
  }

  /**
   * 
   * @param expectedOutput Corresponding target for previous Input of forward-pass
   * @param alpha update factor
   * @returns squared summed loss
   */
  public backward(expectedOutput: Array<number> | Float64Array, alpha?: number): number {
    FNNModel.assert(this.graph['needsBackpropagation'], '['+ this.constructor.name +'] Trainability is not enabled.');
    FNNModel.assert(typeof this.previousOutput !== 'undefined', '['+ this.constructor.name +'] Please execute `forward()` before calling `backward()`');
    this.propagateLossIntoDecoderLayer(expectedOutput);
    this.backwardGraph();
    this.updateWeights(alpha);
    const lossSum = this.calculateLossSumByForwardPass(expectedOutput);
    this.cleanUp();
    return lossSum * lossSum;
  }

  private cleanUp(): void {
    this.resetGraph();
    this.previousOutput = undefined;
    this.previousInput = undefined;
  }

  private backwardGraph(): void {
    this.graph.backward();
  }

  private resetGraph(): void {
    this.graph.forgetCurrentSequence();
  }

  private propagateLossIntoDecoderLayer(expected: Array<number> | Float64Array): void {
    let loss;
    for (let i = 0; i < this.architecture.outputSize; i++) {
      loss = this.previousOutput.w[i] - expected[i];
      if (Math.abs(loss) <= this.training.loss) {
        continue;
      } else {
        loss = this.clipLoss(loss);
        this.previousOutput.dw[i] = loss;
      }
    }
  }

  private calculateLossSumByForwardPass(expected: Array<number> | Float64Array): number {
    let loss, lossSum = 0;
    const out = this.forward(this.previousInput);
    for (let i = 0; i < this.architecture.outputSize; i++) {
      loss = out[i] - expected[i];
      lossSum += loss;
    }
    return lossSum;
  }

  private clipLoss(loss: number): number {
    if (loss > this.training.lossClamp) { return this.training.lossClamp; }
    else if (loss < -this.training.lossClamp) { return -this.training.lossClamp; }
    return loss;
  }

  protected updateWeights(alpha?: number): void {
    alpha = alpha ? alpha : this.training.alpha;
    this.updateHiddenLayer(alpha);
    this.updateDecoderLayer(alpha);
  }

  private updateHiddenLayer(alpha: number): void {
    for (let i = 0; i < this.architecture.hiddenUnits.length; i++) {
      this.model.hidden.Wh[i].update(alpha);
      this.model.hidden.bh[i].update(alpha);
    }
  }

  private updateDecoderLayer(alpha: number): void {
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
    this.previousInput = input;
    this.previousOutput = outputMat;
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

  protected abstract specificForwardpass(state: Mat): Array<Mat>;

  protected computeOutput(hiddenUnitActivations: Array<Mat>): Mat {
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
