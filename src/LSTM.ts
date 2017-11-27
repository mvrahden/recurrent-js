import { Mat } from "./Mat";
import { RandMat } from "./RandMat";
import { Graph } from "./Graph";

export class LSTM {
  inputSize: number;
  hiddenSizes: Array<number>;
  outputSize: number;

  inputb: Array<Mat>;
  inputWh: Array<RandMat>;
  inputWx: Array<RandMat>;
  forgetb: Array<Mat>;
  forgetWh: Array<RandMat>;
  forgetWx: Array<RandMat>;
  outputb: Array<Mat>;
  outputWh: Array<RandMat>;
  outputWx: Array<RandMat>;
  cellb: Array<RandMat>;
  cellWh: Array<RandMat>;
  cellWx: Array<RandMat>;
  
  decoderb: Mat;
  decoderWh: RandMat;

  constructor(inputSize:number, hiddenSizes:Array<number>, outputSize:number) {
    this.inputSize = inputSize;
    this.hiddenSizes = hiddenSizes;
    this.outputSize = outputSize;

    for(let d = 0; d<hiddenSizes.length; d++) {
      // loop over hidden depths
      const prevSize = d === 0 ? inputSize : hiddenSizes[d - 1];
      const hiddenSize = hiddenSizes[d];
      // gates parameters
      this.inputWx[d] = new RandMat(hiddenSize, prevSize, 0, 0.08);
      this.inputWh[d] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.inputb[d] = new Mat(hiddenSize, 1);
      this.forgetWx[d] = new RandMat(hiddenSize, prevSize, 0, 0.08);
      this.forgetWh[d] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.forgetb[d] = new Mat(hiddenSize, 1);
      this.outputWx[d] = new RandMat(hiddenSize, prevSize, 0, 0.08);
      this.outputWh[d] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.outputb[d] = new Mat(hiddenSize, 1);
      // cell write params
      this.cellWx[d] = new RandMat(hiddenSize, prevSize, 0, 0.08);
      this.cellWh[d] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.cellb[d] = new Mat(hiddenSize, 1);
    }
    // decoder params
    this.decoderWh = new RandMat(outputSize, (hiddenSizes.length - 1), 0, 0.08);
    this.decoderb = new Mat(outputSize, 1);
  }

  /**
   * Forward Propagation for a single tick of LSTM
   * @param graph Graph to append Operations (ops) to
   * @param observations 1D column vector with observations
   * @param prev Structure containing hidden representation ['h'] and cell memory ['c'] of type `Mat[]` from previous iteration
   * @returns Structure containing hidden representation ['h'] and cell memory ['c'] of type `Mat[]` and output ['output'] of type `Mat`
   */
  public forwardLSTM (graph:Graph, observations:Mat, prev:any):{'c':Mat[], 'h':Mat[], 'o':Mat} {

    let hiddenPrevs:Array<Mat>, cellPrevs:Array<Mat>;
    
    if (prev == null || typeof prev.h === 'undefined') {
      hiddenPrevs = new Array<Mat>();
      cellPrevs = new Array<Mat>();
      // populate
      for (let d = 0; d < this.hiddenSizes.length; d++) {
        hiddenPrevs.push(new Mat(this.hiddenSizes[d], 1));
        cellPrevs.push(new Mat(this.hiddenSizes[d], 1));
      }
    } else {
      hiddenPrevs = prev.h;
      cellPrevs = prev.c;
    }

    const hidden:Array<Mat> = [];
    const cell:Array<Mat> = [];
    for (let d = 0; d < this.hiddenSizes.length; d++) {

      const inputVector = (d === 0) ? observations : hidden[d - 1]; // first iteration fill Observations
      const hiddenPrev = hiddenPrevs[d];
      const cellPrev = cellPrevs[d];

      // input gate
      const h0 = graph.mul(this.inputWx[d], inputVector);
      const h1 = graph.mul(this.inputWh[d], hiddenPrev);
      const inputGate = graph.sigmoid(graph.add(graph.add(h0, h1), this.inputb[d]));

      // forget gate
      const h2 = graph.mul(this.forgetWx[d], inputVector);
      const h3 = graph.mul(this.forgetWh[d], hiddenPrev);
      const forgetGate = graph.sigmoid(graph.add(graph.add(h2, h3), this.forgetb[d]));

      // output gate
      const h4 = graph.mul(this.outputWx[d], inputVector);
      const h5 = graph.mul(this.outputWh[d], hiddenPrev);
      const outputGate = graph.sigmoid(graph.add(graph.add(h4, h5), this.outputb[d]));

      // write operation on cells
      const h6 = graph.mul(this.cellWx[d], inputVector);
      const h7 = graph.mul(this.cellWh[d], hiddenPrev);
      const cellWrite = graph.tanh(graph.add(graph.add(h6, h7), this.cellb[d]));

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
    const output = graph.add(graph.mul(this.decoderWh, hidden[hidden.length - 1]), this.decoderb);

    // return cell memory, hidden representation and output
    return { 'h': hidden, 'c': cell, 'o': output };
  }
}
