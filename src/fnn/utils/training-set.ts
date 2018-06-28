import { Sample } from "./sample";

export class TrainingSet {

  private samples: Array<Sample>;

  setSamples(samples: Array<Sample>): void {
    this.samples = samples;
  }

  length(): number {
    return this.samples.length;
  }

  getInputForSample(i: number): Array<number> {
    return this.samples[i].input;
  }

  getExpectedOutputForSample(i: number): Array<number> {
    return this.samples[i].output;
  }
}
