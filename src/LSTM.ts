import { RandMat, Mat, Graph, RNNModel, InnerState, NetOpts } from '.';

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
   * @param state 1D column vector with observations
   * @param previousInnerState Structure containing hidden representation ['h'] and cell memory ['c'] of type `Mat[]` from previous iteration
   * @param graph Optional: inject Graph to append Operations
   * @returns Structure containing hidden representation ['h'] and cell memory ['c'] of type `Mat[]` and output ['output'] of type `Mat`
   */
  public forward(state: Mat, previousInnerState: InnerState, graph?: Graph): InnerState {
    graph = graph ? graph : this.graph;
    
    const previousCells = this.getPreviousCellActivations(previousInnerState);
    const previousHiddenUnits = this.getPreviousHiddenUnitActivations(previousInnerState);

    const cells = new Array<Mat>();
    const hiddenUnits = new Array<Mat>();
    this.computeHiddenLayer(state, previousHiddenUnits, previousCells, hiddenUnits, cells, graph);

    const output = this.computeOutput(hiddenUnits, graph);

    // return cell memory, hidden representation and output
    return { 'hiddenUnits': hiddenUnits, 'cells': cells, 'output': output };
  }

  private computeHiddenLayer(state: Mat, previousHiddenUnits: Mat[], previousCells: Mat[], hiddenUnits: Mat[], cells: Mat[], graph: Graph) {
    for (let i = 0; i < this.hiddenUnits.length; i++) {
      const inputVector = (i === 0) ? state : hiddenUnits[i - 1]; // first iteration fill Observations
      const hiddenPrev = previousHiddenUnits[i];
      const previousCell = previousCells[i];
      // input gate
      const h0 = graph.mul(this.model.hidden.input.Wx[i], inputVector);
      const h1 = graph.mul(this.model.hidden.input.Wh[i], hiddenPrev);
      const inputGate = graph.sigmoid(graph.add(graph.add(h0, h1), this.model.hidden.input.bh[i]));
      // forget gate
      const h2 = graph.mul(this.model.hidden.forget.Wx[i], inputVector);
      const h3 = graph.mul(this.model.hidden.forget.Wh[i], hiddenPrev);
      const forgetGate = graph.sigmoid(graph.add(graph.add(h2, h3), this.model.hidden.forget.bh[i]));
      // output gate
      const h4 = graph.mul(this.model.hidden.output.Wx[i], inputVector);
      const h5 = graph.mul(this.model.hidden.output.Wh[i], hiddenPrev);
      const outputGate = graph.sigmoid(graph.add(graph.add(h4, h5), this.model.hidden.output.bh[i]));
      // write operation on cells
      const h6 = graph.mul(this.model.hidden.cell.Wx[i], inputVector);
      const h7 = graph.mul(this.model.hidden.cell.Wh[i], hiddenPrev);
      const cellWrite = graph.tanh(graph.add(graph.add(h6, h7), this.model.hidden.cell.bh[i]));
      // compute new cell activation
      const retainCell = graph.eltmul(forgetGate, previousCell); // what do we keep from cell
      const writeCell = graph.eltmul(inputGate, cellWrite); // what do we write to cell
      const cellActivations = graph.add(retainCell, writeCell); // new cell contents
      // compute hidden state as gated, saturated cell activations
      const activations = graph.eltmul(outputGate, graph.tanh(cellActivations));
      cells.push(cellActivations);
      hiddenUnits.push(activations);
    }
  }

  private getPreviousHiddenUnitActivations(previousInnerState: InnerState): Mat[] {
    let previousHiddenUnits;
    if (this.givenPreviousInnerState(previousInnerState)) {
      previousHiddenUnits = previousInnerState.hiddenUnits;
    }
    else {
      previousHiddenUnits = new Array<Mat>();
      // populate
      for (let d = 0; d < this.hiddenUnits.length; d++) {
        previousHiddenUnits.push(new Mat(this.hiddenUnits[d], 1));
      }
    }
    return previousHiddenUnits;
  }

  private getPreviousCellActivations(previousInnerState: InnerState): Mat[] {
    let previousCells;
    if (this.givenPreviousInnerState(previousInnerState)) {
      previousCells = previousInnerState.cells;
    }
    else {
      previousCells = new Array<Mat>();
      // populate
      for (let d = 0; d < this.hiddenUnits.length; d++) {
        previousCells.push(new Mat(this.hiddenUnits[d], 1));
      }
    }
    return previousCells;
  }

  private givenPreviousInnerState(previousInnerState: InnerState) {
    return previousInnerState || typeof previousInnerState.hiddenUnits !== 'undefined';
  }
}
