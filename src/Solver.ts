import { Mat } from './Mat';
import { RNNModel } from './RNNModel';

export class Solver {
  protected readonly decayRate: number;
  protected readonly smoothEps: number;
  protected readonly stepCache: {};

  protected stepTotalNumber: number;
  protected stepNumberOfClippings: number;

  constructor(decayRate: number = 0.999, smoothEps: number = 1e-8) {
    this.decayRate = decayRate;
    this.smoothEps = smoothEps;
    this.stepCache = {};
  }

  private reset(): void {
    this.stepNumberOfClippings = 0;
    this.stepTotalNumber = 0;
  }

  /**
   * Performs a RMSprop parameter update of Model.
   * @param stepSize 
   * @param l2Regularization 
   * @param clippingValue Gradient clipping
   * @returns an Object containing the Clipping Ratio
   */
  public step(model: RNNModel, stepSize: number, l2Regularization: number, clippingValue: number): { 'ratioClipped': number } {
    this.reset();
    const solverStats = { ratioClipped: 0 };

    for (const key in model) {
      if (model.hasOwnProperty(key)) {
        this.iterateModelLayer(model, key, clippingValue, l2Regularization, stepSize);
      }
    }

    solverStats.ratioClipped = this.stepNumberOfClippings * 1.0 / this.stepTotalNumber;
    return solverStats;
  }

  private iterateModelLayer(model: RNNModel, key: any, clipval: number, regc: number, stepSize: number): void {
    const currentModelLayer = model[key];
    if (!(this.stepCache.hasOwnProperty(key))) {
      this.stepCache[key] = new Mat(currentModelLayer.n, currentModelLayer.d);
    }

    const currentStepCache = this.stepCache[key];
    for (let i = 0; i < currentModelLayer.w.length; i++) {
      let mdwi = this.RMSprop(currentModelLayer, i, currentStepCache);
      mdwi = this.gradientClipping(mdwi, clipval);
      this.update(currentModelLayer, i, stepSize, mdwi, currentStepCache, regc);
      this.resetGradients(currentModelLayer, i);
    }
  }

  /**
   * rmsprop with adaptive learning rates
   * RMSprop decay the past accumulated gradient,
   * so only a portion of past gradients are considered.
   * Now, instead of considering all of the past gradients,
   * RMSprop behaves like moving average.
   * (https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/)
   */
  private RMSprop(modelLayer: Mat, i: number, stepCache: Mat): number {
    const mdwi = modelLayer.dw[i];
    stepCache.w[i] = stepCache.w[i] * this.decayRate + (1.0 - this.decayRate) * mdwi * mdwi;
    return mdwi;
  }

  /**
   * 
   * @param mdwi 
   * @param clipval 
   */
  private gradientClipping(mdwi: number, clipval: number): number {
    if (mdwi > clipval) {
      mdwi = clipval;
      this.stepNumberOfClippings++;
    }
    else if (mdwi < -clipval) {
      mdwi = -clipval;
      this.stepNumberOfClippings++;
    }
    this.stepTotalNumber++;
    return mdwi;
  }

  /**
   * updates and regularizes
   * @param m 
   * @param i 
   * @param stepSize 
   * @param mdwi 
   * @param s 
   * @param regc 
   */
  private update(m: Mat, i: number, stepSize: number, mdwi: number, stepCache: Mat, regc: number): void {
    m.w[i] += -stepSize * mdwi / Math.sqrt(stepCache.w[i] + this.smoothEps) - regc * m.w[i];
  }

  /**
   * resets the gradients for the next iteration
   * @param currentModelLayer 
   * @param i 
   */
  private resetGradients(currentModelLayer: any, i: number) {
    currentModelLayer.dw[i] = 0;
  }
}
