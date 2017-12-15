import { Mat } from './Mat';

export class Net {
  public W1: Mat | null = null;
  public b1: Mat | null = null;
  public W2: Mat | null = null;
  public b2: Mat | null = null;

  static update(net: Net, alpha: number): void {
    for (const property in net) {
      if (net.hasOwnProperty(property)) {
        Mat.update(net[property], alpha);
      }
    }
  }

  static toJSON(net: Net): {} {
    const j = {};
    for (const property in net) {
      if (net.hasOwnProperty(property)) {
        j[property] = net[property].toJSON();
      }
    }
    return j;
  }

  static fromJSON(j): Net {
    const net = new Net();
    for (const property in j) {
      if (j.hasOwnProperty(property)) {
        net[property] = new Mat(1, 1); // not proud of this
        net[property].fromJSON(j[property]);
      }
    }
    return net;
  }
}
