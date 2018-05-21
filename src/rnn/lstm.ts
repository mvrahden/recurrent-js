import { RandMat, Mat, Graph, InnerState, NetOpts } from './..';
import { RNNModel } from './rnn-model';

export class LSTM extends RNNModel {
  /**
   * Generates a Neural Net instance from a pretrained Neural Net JSON.
   * @param {{ hidden: { input: { Wh, Wx, bh }, forget: { Wh, Wx, bh }, output: { Wh, Wx, bh }, cell: { Wh, Wx, bh } }, decoder: { Wh, b } }} opt Specs of the Neural Net.
   */
  constructor(opt: { hidden: { input: { Wh, Wx, bh }, forget: { Wh, Wx, bh }, output: { Wh, Wx, bh }, cell: { Wh, Wx, bh } }, decoder: { Wh, b } });
  /**
   * Generates a Neural Net with given specs.
   * @param {NetOpts} opt Specs of the Neural Net. [defaults to: needsBackprop = true, mu = 0, std = 0.01]
   */
  constructor(opt: NetOpts);
  constructor(opt: any) {
    super(opt);
  }

  protected initializeNetworkModel(): { hidden: any; decoder: { Wh: Mat; b: Mat; }; } {
    return {
      hidden: {
        input: {
          Wx: new Array<Mat>(this.hiddenUnits.length),
          Wh: new Array<Mat>(this.hiddenUnits.length),
          bh: new Array<Mat>(this.hiddenUnits.length)
        },
        forget: {
          Wx: new Array<Mat>(this.hiddenUnits.length),
          Wh: new Array<Mat>(this.hiddenUnits.length),
          bh: new Array<Mat>(this.hiddenUnits.length)
        },
        output: {
          Wx: new Array<Mat>(this.hiddenUnits.length),
          Wh: new Array<Mat>(this.hiddenUnits.length),
          bh: new Array<Mat>(this.hiddenUnits.length)
        },
        cell: {
          Wx: new Array<Mat>(this.hiddenUnits.length),
          Wh: new Array<Mat>(this.hiddenUnits.length),
          bh: new Array<Mat>(this.hiddenUnits.length)
        },
      },
      decoder: {
        Wh: null,
        b: null
      }
    };
  }

  protected isFromJSON(opt: any) {
    return RNNModel.has(opt, ['hidden', 'decoder'])
      && RNNModel.has(opt.hidden, ['input', 'forget', 'output', 'cell'])
      && RNNModel.has(opt.input, ['Wh', 'Wx', 'bh'])
      && RNNModel.has(opt.forget, ['Wh', 'Wx', 'bh'])
      && RNNModel.has(opt.output, ['Wh', 'Wx', 'bh'])
      && RNNModel.has(opt.cell, ['Wh', 'Wx', 'bh'])
      && RNNModel.has(opt.decoder, ['Wh', 'b']);
  }

  protected initializeHiddenLayerFromJSON(opt: { hidden: any; decoder: { Wh: Mat; b: Mat; }; }): void {
    RNNModel.assert(opt.hidden.forget && opt.hidden.forget && opt.hidden.output && opt.hidden.cell, 'Wrong JSON Format to recreat Hidden Layer.');
    this.isValid(opt.hidden.input);
    this.isValid(opt.hidden.forget);
    this.isValid(opt.hidden.output);
    this.isValid(opt.hidden.cell);

    for (let i = 0; i < opt.hidden.Wh.length; i++) {
      this.model.hidden.input.Wx = Mat.fromJSON(opt.hidden.input.Wx);
      this.model.hidden.input.Wh = Mat.fromJSON(opt.hidden.input.Wh);
      this.model.hidden.input.bh = Mat.fromJSON(opt.hidden.input.bh);
      this.model.hidden.forget.Wx = Mat.fromJSON(opt.hidden.Wx);
      this.model.hidden.forget.Wh = Mat.fromJSON(opt.hidden.Wh);
      this.model.hidden.forget.bh = Mat.fromJSON(opt.hidden.bh);
      this.model.hidden.output.Wx = Mat.fromJSON(opt.hidden.Wx);
      this.model.hidden.output.Wh = Mat.fromJSON(opt.hidden.Wh);
      this.model.hidden.output.bh = Mat.fromJSON(opt.hidden.bh);
      // cell write params
      this.model.hidden.cell.Wx = Mat.fromJSON(opt.hidden.Wx);
      this.model.hidden.cell.Wh = Mat.fromJSON(opt.hidden.Wh);
      this.model.hidden.cell.bh = Mat.fromJSON(opt.hidden.hb);
    }
  }

  private isValid(component: any): any {
    RNNModel.assert(component && !Array.isArray(component['Wx']), 'Wrong JSON Format to recreat Hidden Layer.');
    RNNModel.assert(component && !Array.isArray(component['Wh']), 'Wrong JSON Format to recreat Hidden Layer.');
    RNNModel.assert(component && !Array.isArray(component['bh']), 'Wrong JSON Format to recreat Hidden Layer.');
  }

  protected initializeHiddenLayer() {
    for (let i = 0; i < this.hiddenUnits.length; i++) {
      // loop over hidden depths
      const prevSize = i === 0 ? this.inputSize : this.hiddenUnits[i - 1];
      const hiddenSize = this.hiddenUnits[i];
      // gate parameters
      this.model.hidden.input.Wx[i] = new RandMat(hiddenSize, prevSize, 0, 0.08);
      this.model.hidden.input.Wh[i] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.model.hidden.input.bh[i] = new Mat(hiddenSize, 1);
      this.model.hidden.forget.Wx[i] = new RandMat(hiddenSize, prevSize, 0, 0.08);
      this.model.hidden.forget.Wh[i] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.model.hidden.forget.bh[i] = new Mat(hiddenSize, 1);
      this.model.hidden.output.Wx[i] = new RandMat(hiddenSize, prevSize, 0, 0.08);
      this.model.hidden.output.Wh[i] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.model.hidden.output.bh[i] = new Mat(hiddenSize, 1);
      // cell write params
      this.model.hidden.cell.Wx[i] = new RandMat(hiddenSize, prevSize, 0, 0.08);
      this.model.hidden.cell.Wh[i] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.model.hidden.cell.bh[i] = new Mat(hiddenSize, 1);
    }
  }

  /**
   * Forward pass for a single tick of Neural Network
   * @param input 1D column vector with observations
   * @param previousActivationState Structure containing hidden representation ['h'] and cell memory ['c'] of type `Mat[]` from previous iteration
   * @param graph Optional: inject Graph to append Operations
   * @returns Structure containing hidden representation ['h'] and cell memory ['c'] of type `Mat[]` and output ['output'] of type `Mat`
   */
  public forward(input: Mat, previousActivationState?: InnerState, graph?: Graph): InnerState {
    previousActivationState = previousActivationState ? previousActivationState : null;
    graph = graph ? graph : this.graph;

    const previousHiddenActivations = { cells: null, units: null };
    previousHiddenActivations.cells = this.getPreviousCellActivationsFrom(previousActivationState);
    previousHiddenActivations.units = this.getPreviousHiddenUnitActivationsFrom(previousActivationState);

    const hiddenActivations: { units, cells } = this.computeHiddenActivations(input, previousHiddenActivations, graph);

    const output = this.computeOutput(hiddenActivations.units, graph);

    // return cell memory, hidden representation and output
    return { 'hiddenActivationState': hiddenActivations.units, 'cells': hiddenActivations.cells, 'output': output };
  }

  private computeHiddenActivations(state: Mat, previousHiddenActivations: { units: Mat[], cells: Mat[] }, graph: Graph) {
    const hiddenActivations = { units: [], cells: [] };
    for (let i = 0; i < this.hiddenUnits.length; i++) {
      const inputVector = (i === 0) ? state : hiddenActivations.units[i - 1]; // first iteration fill Observations
      const previousUnitActivations = previousHiddenActivations.units[i];
      const previousCellActivations = previousHiddenActivations.cells[i];
      // input gate
      const weightedStatelessInputPortion1 = graph.mul(this.model.hidden.input.Wx[i], inputVector);
      const weightedStatefulInputPortion1 = graph.mul(this.model.hidden.input.Wh[i], previousUnitActivations);
      const summedUpInput1 = graph.add(graph.add(weightedStatelessInputPortion1, weightedStatefulInputPortion1), this.model.hidden.input.bh[i]);
      const inputGateActivation = graph.sig(summedUpInput1);
      // forget gate
      const weightedStatelessInputPortion2 = graph.mul(this.model.hidden.forget.Wx[i], inputVector);
      const weightedStatefulInputPortion2 = graph.mul(this.model.hidden.forget.Wh[i], previousUnitActivations);
      const summedUpInput2 = graph.add(graph.add(weightedStatelessInputPortion2, weightedStatefulInputPortion2), this.model.hidden.forget.bh[i]);
      const forgetGateActivation = graph.sig(summedUpInput2);
      // output gate
      const weightedStatelessInputPortion3 = graph.mul(this.model.hidden.output.Wx[i], inputVector);
      const weightedStatefulInputPortion3 = graph.mul(this.model.hidden.output.Wh[i], previousUnitActivations);
      const summedUpInput3 = graph.add(graph.add(weightedStatelessInputPortion3, weightedStatefulInputPortion3), this.model.hidden.output.bh[i]);
      const outputGateActivation = graph.sig(summedUpInput3);
      // write operation on cells
      const weightedStatelessInputPortion4 = graph.mul(this.model.hidden.cell.Wx[i], inputVector);
      const weightedStatefulInputPortion4 = graph.mul(this.model.hidden.cell.Wh[i], previousUnitActivations);
      const summedUpInput4 = graph.add(graph.add(weightedStatelessInputPortion4, weightedStatefulInputPortion4), this.model.hidden.cell.bh[i]);
      const cellWriteActivation = graph.tanh(summedUpInput4);
      // compute new cell activation
      const retainCell = graph.eltmul(forgetGateActivation, previousCellActivations); // what do we keep from cell
      const writeCell = graph.eltmul(inputGateActivation, cellWriteActivation); // what do we write to cell
      const cellActivations = graph.add(retainCell, writeCell); // new cell contents
      // compute hidden state as gated, saturated cell activations
      const activations = graph.eltmul(outputGateActivation, graph.tanh(cellActivations));
      hiddenActivations.cells.push(cellActivations);
      hiddenActivations.units.push(activations);
    }
    return hiddenActivations;
  }

  private getPreviousCellActivationsFrom(previousActivationState: InnerState): Mat[] {
    let previousCellsActivations;
    if (this.givenPreviousActivationState(previousActivationState)) {
      previousCellsActivations = previousActivationState.cells;
    }
    else {
      previousCellsActivations = new Array<Mat>();
      for (let i = 0; i < this.hiddenUnits.length; i++) {
        previousCellsActivations.push(new Mat(this.hiddenUnits[i], 1));
      }
    }
    return previousCellsActivations;
  }

  private getPreviousHiddenUnitActivationsFrom(previousActivationState: InnerState): Mat[] {
    let previousHiddenActivations;
    if (this.givenPreviousActivationState(previousActivationState)) {
      previousHiddenActivations = previousActivationState.hiddenActivationState;
    }
    else {
      previousHiddenActivations = new Array<Mat>();
      for (let i = 0; i < this.hiddenUnits.length; i++) {
        previousHiddenActivations.push(new Mat(this.hiddenUnits[i], 1));
      }
    }
    return previousHiddenActivations;
  }

  private givenPreviousActivationState(previousInnerState: InnerState) {
    return previousInnerState && typeof previousInnerState.hiddenActivationState !== 'undefined';
  }

  protected updateHiddenUnits(alpha: number): void {
    for (let i = 0; i < this.hiddenUnits.length; i++) {
      this.model.hidden.input.Wx[i].update(alpha);
      this.model.hidden.input.Wh[i].update(alpha);
      this.model.hidden.input.bh[i].update(alpha);

      this.model.hidden.output.Wx[i].update(alpha);
      this.model.hidden.output.Wh[i].update(alpha);
      this.model.hidden.output.bh[i].update(alpha);

      this.model.hidden.forget.Wx[i].update(alpha);
      this.model.hidden.forget.Wh[i].update(alpha);
      this.model.hidden.forget.bh[i].update(alpha);

      this.model.hidden.cell.Wx[i].update(alpha);
      this.model.hidden.cell.Wh[i].update(alpha);
      this.model.hidden.cell.bh[i].update(alpha);
    }
  }

  protected updateDecoder(alpha: number): void {
    this.model.decoder.Wh.update(alpha);
    this.model.decoder.b.update(alpha);
  }
}
