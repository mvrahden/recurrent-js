import { RandMat } from './RandMat';
import { Mat } from './Mat';
import { Graph } from './Graph';
import { NNModel } from './NNModel';
import { PreviousOutput } from './utils/PreviousOutput';

export class RNN extends NNModel {
  hiddenSizes: Array<number>;

  constructor(inputSize: number, hiddenSizes: Array<number>, outputSize: number, needsBackProp: boolean = true) {
    super(needsBackProp);

    this.hiddenSizes = hiddenSizes;

    let hiddenSize;
    for (let d = 0; d < this.hiddenSizes.length; d++) {
      const previousSize = d === 0 ? inputSize : this.hiddenSizes[d - 1];
      hiddenSize = this.hiddenSizes[d];
      this.model.hiddenWx[d] = new RandMat(hiddenSize, previousSize, 0, 0.08);
      this.model.hiddenWh[d] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.model.hiddenbh[d] = new Mat(hiddenSize, 1);
    }

    // decoder params
    this.model.decoderWh = new RandMat(outputSize, hiddenSize, 0, 0.08);
    this.model.decoderbd = new Mat(outputSize, 1);
  }

  /**
   * Forward propagation for a single tick of RNN
   * @param observations 1D column vector with observations
   * @param previousOutput Structure containing hidden representation ['h'] of type `Mat[]` from previous iteration
   * @param graph optional: inject Graph to append Operations
   * @returns Structure containing hidden representation ['h'] of type `Mat[]` and output ['output'] of type `Mat`
   */
  forward(observations: Mat, previousOutput: PreviousOutput, graph: Graph = this.graph): PreviousOutput {

    let hiddenPrevs;
    if (typeof previousOutput.h === 'undefined') {
      hiddenPrevs = [];
      for (let d = 0; d < this.hiddenSizes.length; d++) {
        hiddenPrevs.push(new Mat(this.hiddenSizes[d], 1));
      }
    } else {
      hiddenPrevs = previousOutput.h;
    }

    const hidden = new Array<Mat>();
    for (let d = 0; d < this.hiddenSizes.length; d++) {

      const inputVector = d === 0 ? observations : hidden[d - 1];
      const hiddenPrev = hiddenPrevs[d];

      const h0 = graph.mul(this.model.hiddenWx[d], inputVector);
      const h1 = graph.mul(this.model.hiddenWh[d], hiddenPrev);
      const hiddenD = graph.relu(graph.add(graph.add(h0, h1), this.model.hiddenbh[d]));

      hidden.push(hiddenD);
    }

    // one decoder to outputs at end
    const output = graph.add(graph.mul(this.model.decoderWh, hidden[hidden.length - 1]), this.model.decoderbd);

    // return cell memory, hidden representation and output
    return { 'h': hidden, 'o': output, 'c': null };
  }
}
