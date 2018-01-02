import { Graph } from './Graph';

export class NNModel {

  public readonly model;
  protected readonly graph: Graph;

  constructor(needsBackProp: boolean = true) {
    this.model = {};
    this.graph = new Graph(needsBackProp);
  }

}
