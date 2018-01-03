import { Graph } from './Graph';
import { Mat } from './Mat';
import { PreviousOutput } from './utils/PreviousOutput';

export abstract class NNModel {

  public readonly model: any;
  protected readonly graph: Graph;

  constructor(needsBackProp: boolean = true) {
    this.model = {};
    this.graph = new Graph(needsBackProp);
  }

  abstract forward(observations: Mat, previousOutput: PreviousOutput, graph: Graph): PreviousOutput;

}
