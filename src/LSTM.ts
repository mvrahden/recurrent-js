import { Mat } from './Mat';
import { RandMat } from './RandMat';
import { Graph } from './Graph';
import { NNModel } from './NNModel';
import { PreviousOutput } from './utils/PreviousOutput';

export class LSTM extends NNModel {
  
  inputSize: number;
  hiddenSizes: Array<number>;
  outputSize: number;

  constructor(inputSize: number, hiddenSizes: Array<number>, outputSize: number, needsBackProp: boolean = true) {
    super(needsBackProp);
    
    this.inputSize = inputSize;
    this.hiddenSizes = hiddenSizes;
    this.outputSize = outputSize;

    for (let i = 0; i < hiddenSizes.length; i++) {
      // loop over hidden depths
      const prevSize = i === 0 ? inputSize : hiddenSizes[i - 1];
      const hiddenSize = hiddenSizes[i];
      // gate parameters
      this.model.inputWx[i] = new RandMat(hiddenSize, prevSize, 0, 0.08);
      this.model.inputWh[i] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.model.inputb[i] = new Mat(hiddenSize, 1);
      this.model.forgetWx[i] = new RandMat(hiddenSize, prevSize, 0, 0.08);
      this.model.forgetWh[i] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.model.forgetb[i] = new Mat(hiddenSize, 1);
      this.model.outputWx[i] = new RandMat(hiddenSize, prevSize, 0, 0.08);
      this.model.outputWh[i] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.model.outputb[i] = new Mat(hiddenSize, 1);
      // cell write params
      this.model.cellWx[i] = new RandMat(hiddenSize, prevSize, 0, 0.08);
      this.model.cellWh[i] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.model.cellb[i] = new Mat(hiddenSize, 1);
    }
    // decoder params
    this.model.decoderWh = new RandMat(outputSize, (hiddenSizes.length - 1), 0, 0.08);
    this.model.decoderb = new Mat(outputSize, 1);
  }

  /**
   * Forward Propagation for a single tick of LSTM
   * @param observations 1D column vector with observations
   * @param previousOutput Structure containing hidden representation ['h'] and cell memory ['c'] of type `Mat[]` from previous iteration
   * @param graph Optional: inject Graph to append Operations
   * @returns Structure containing hidden representation ['h'] and cell memory ['c'] of type `Mat[]` and output ['output'] of type `Mat`
   */
  public forward(observations: Mat, previousOutput: PreviousOutput, graph: Graph = this.graph): PreviousOutput {

    let hiddenPrevs: Array<Mat>, cellPrevs: Array<Mat>;

    if (previousOutput == null || typeof previousOutput.h === 'undefined') {
      hiddenPrevs = new Array<Mat>();
      cellPrevs = new Array<Mat>();
      // populate
      for (let d = 0; d < this.hiddenSizes.length; d++) {
        hiddenPrevs.push(new Mat(this.hiddenSizes[d], 1));
        cellPrevs.push(new Mat(this.hiddenSizes[d], 1));
      }
    } else {
      hiddenPrevs = previousOutput.h;
      cellPrevs = previousOutput.c;
    }

    const hidden: Array<Mat> = [];
    const cell: Array<Mat> = [];
    for (let d = 0; d < this.hiddenSizes.length; d++) {

      const inputVector = (d === 0) ? observations : hidden[d - 1]; // first iteration fill Observations
      const hiddenPrev = hiddenPrevs[d];
      const cellPrev = cellPrevs[d];

      // input gate
      const h0 = graph.mul(this.model.inputWx[d], inputVector);
      const h1 = graph.mul(this.model.inputWh[d], hiddenPrev);
      const inputGate = graph.sigmoid(graph.add(graph.add(h0, h1), this.model.inputb[d]));

      // forget gate
      const h2 = graph.mul(this.model.forgetWx[d], inputVector);
      const h3 = graph.mul(this.model.forgetWh[d], hiddenPrev);
      const forgetGate = graph.sigmoid(graph.add(graph.add(h2, h3), this.model.forgetb[d]));

      // output gate
      const h4 = graph.mul(this.model.outputWx[d], inputVector);
      const h5 = graph.mul(this.model.outputWh[d], hiddenPrev);
      const outputGate = graph.sigmoid(graph.add(graph.add(h4, h5), this.model.outputb[d]));

      // write operation on cells
      const h6 = graph.mul(this.model.cellWx[d], inputVector);
      const h7 = graph.mul(this.model.cellWh[d], hiddenPrev);
      const cellWrite = graph.tanh(graph.add(graph.add(h6, h7), this.model.cellb[d]));

      // compute new cell activation
      const retainCell = graph.eltmul(forgetGate, cellPrev); // what do we keep from cell
      const writeCell = graph.eltmul(inputGate, cellWrite); // what do we write to cell
      const cellD = graph.add(retainCell, writeCell); // new cell contents

      // compute hidden state as gated, saturated cell activations
      const hiddenD = graph.eltmul(outputGate, graph.tanh(cellD));

      hidden.push(hiddenD);
      cell.push(cellD);
    }

    // one decoder to outputs at end
    const output = graph.add(graph.mul(this.model.decoderWh, hidden[hidden.length - 1]), this.model.decoderb);

    // return cell memory, hidden representation and output
    return { 'h': hidden, 'c': cell, 'o': output };
  }
}
