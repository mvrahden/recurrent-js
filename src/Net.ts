import { Mat } from './Mat';

export class Net {
  public W1: Mat | null = null;
  public b1: Mat | null = null;
  public W2: Mat | null = null;
  public b2: Mat | null = null;

  constructor() {}

  /**
   * Updates all 
   * @param net 
   * @param alpha 
   */
  public static update(net: Net, alpha: number): void {
    for (const property in net) {
      if (net.hasOwnProperty(property)) {
        Mat.update(net[property], alpha);
      }
    }
  }

  public static toJSON(net: Net): {} {
    const j = {};
    for (const property in net) {
      if (net.hasOwnProperty(property)) {
        j[property] = net[property].toJSON();
      }
    }
    return j;
  }

  public static fromJSON(j: { W1: Mat, b1: Mat, W2: Mat, b2: Mat }): Net {
    const net = new Net();
    for (const property in j) {
      if (j.hasOwnProperty(property)) {
        net[property].fromJSON(j[property]);
      }
    }
    return net;
  }
}
