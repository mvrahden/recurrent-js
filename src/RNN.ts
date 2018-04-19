import { RandMat } from './RandMat';
import { Mat } from './Mat';
import { Graph } from './Graph';
import { RNNModel } from './RNNModel';
import { InnerState } from './utils/InnerState';

export class RNN extends RNNModel {
  /**
   * Generates a Neural Net instance from a pretrained Neural Net JSON.
   * @param {{ hidden: { Wh, Wx, bh }, decoder: { Wh, b } }} opt Specs of the Neural Net.
   */
  constructor(opt: { hidden: { Wh, Wx, bh }, decoder: { Wh, b } });
  /**
   * Generates a Neural Net with given specs.
   * @param {{inputSize: number, hiddenSize: Array<number>, outputSize: number, needsBackprop?: boolean, mu: number = 0, std: number = 0.01}} opt Specs of the Neural Net.
   */
  constructor(opt: { inputSize: number, hiddenUnits: Array<number>, outputSize: number, needsBackprop?: boolean, mu?: number, std?: number });
  constructor(opt: any) {
    super(opt);
  }

  protected isFromJSON(opt: any) {
    return RNNModel.has(opt, ['hidden', 'decoder'])
      && RNNModel.has(opt.hidden, ['Wh', 'Wx', 'bh'])
      && RNNModel.has(opt.decoder, ['Wh', 'b']);
  }

  protected initializeHiddenLayerFromJSON(opt: { hidden: { Wh: Mat[], Wx: Mat[], bh: Mat[] }, decoder: { Wh: Mat, b: Mat } }): void {
    RNNModel.assert(!Array.isArray(opt['hidden']['Wh']), 'Wrong JSON Format to recreat Hidden Layer.');
    RNNModel.assert(!Array.isArray(opt['hidden']['Wx']), 'Wrong JSON Format to recreat Hidden Layer.');
    RNNModel.assert(!Array.isArray(opt['hidden']['bh']), 'Wrong JSON Format to recreat Hidden Layer.');
    for (let i = 0; i < opt.hidden.Wh.length; i++) {
      this.model.hidden.Wx[i] = Mat.fromJSON(opt.hidden.Wx[i]);
      this.model.hidden.Wh[i] = Mat.fromJSON(opt.hidden.Wh[i]);
      this.model.hidden.bh[i] = Mat.fromJSON(opt.hidden.bh[i]);
    }
  }

  protected initializeNetworkModel(): { hidden: any; decoder: { Wh: Mat; b: Mat; }; } {
    return {
      hidden: {
        Wx: new Array<Mat>(this.hiddenUnits.length),
        Wh: new Array<Mat>(this.hiddenUnits.length),
        bh: new Array<Mat>(this.hiddenUnits.length)
      },
      decoder: {
        Wh: null,
        b: null
      }
    };
  }

  protected initializeHiddenLayer(): void {
    let hiddenSize;
    for (let d = 0; d < this.hiddenUnits.length; d++) {
      const previousSize = d === 0 ? this.inputSize : this.hiddenUnits[d - 1];
      hiddenSize = this.hiddenUnits[d];
      this.model.hidden.Wx[d] = new RandMat(hiddenSize, previousSize, 0, 0.08);
      this.model.hidden.Wh[d] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.model.hidden.bh[d] = new Mat(hiddenSize, 1);
    }
  }

  /**
   * Forward pass for a single tick of Neural Network
   * @param state 1D column vector with observations
   * @param previousInnerState Structure containing hidden representation ['h'] of type `Mat[]` from previous iteration
   * @param graph optional: inject Graph to append Operations
   * @returns Structure containing hidden representation ['h'] of type `Mat[]` and output ['output'] of type `Mat`
   */
  forward(state: Mat, previousInnerState: InnerState, graph?: Graph): InnerState {
    graph = graph ? graph : this.graph;

    const previousHiddenUnits = this.getPreviousHiddenUnits(previousInnerState);

    const hiddenActivations = this.computeHiddenActivations(state, previousHiddenUnits, graph);

    const output = this.computeOutput(hiddenActivations, graph);

    // return hidden representation and output
    return { 'hiddenUnits': hiddenActivations, 'output': output };
  }

  private getPreviousHiddenUnits(previousInnerState: InnerState) {
    let previousHiddenUnits;
    if (typeof previousInnerState.hiddenUnits === 'undefined') {
      previousHiddenUnits = new Array<Mat>();
      for (let d = 0; d < this.hiddenUnits.length; d++) {
        previousHiddenUnits.push(new Mat(this.hiddenUnits[d], 1));
      }
    }
    else {
      previousHiddenUnits = previousInnerState.hiddenUnits;
    }
    return previousHiddenUnits;
  }

  private computeHiddenActivations(state: Mat, previousHiddenUnits: Mat[], graph: Graph): Mat[] {
    const hiddenActivations = new Array<Mat>();
    for (let d = 0; d < this.hiddenUnits.length; d++) {
      const inputVector = d === 0 ? state : hiddenActivations[d - 1];
      const hiddenPrev = previousHiddenUnits[d];
      const h0 = graph.mul(this.model.hidden.Wx[d], inputVector);
      const h1 = graph.mul(this.model.hidden.Wh[d], hiddenPrev);
      const activation = graph.relu(graph.add(graph.add(h0, h1), this.model.hidden.bh[d]));
      hiddenActivations.push(activation);
    }
    return hiddenActivations;
  }
}
