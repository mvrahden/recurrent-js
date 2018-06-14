import { Utils, DNN, BNN } from '..';
import { ANN } from './ann';


describe('Examples with Neural Networks:', () => {
  /**
   * All testparameters are configured to surpass a test-series of 10.000 Iterations without failing it.
   */

  let sut: ANN;
  const config = {
    architecture: { inputSize: 2, hiddenUnits: [2, 3], outputSize: 3 },
    training: { loss: 1e-11 }
  };

  describe('Stateless Network Architectures:', () => {
    // Training data
    const trainingData = {
      samples: [
        { input: [0, 1], output: [0, 1, 0] },
        { input: [1, 0], output: [1, 0, 1] }
      ],
      getInputForSample: (i: number): Array<number> => {
        return trainingData.samples[i].input;
      },
      getExpectedOutputForSample: (i: number): Array<number> => {
        return trainingData.samples[i].output;
      }
    };

    describe('Deep Feedforward Neural Network (DNN):', () => {

      it('given fresh instance >> perform iterative training routine >> should output exact expected results after 12.000 iterations', () => {
        // for(let j = 0; j < 10000; j++) {
        const trainingIterations = 12000;

        sut = new DNN(config);

        // Output container
        const actualOutputsForSample = [];

        // Get Output of the untrained network
        for (let i = 0; i < trainingData.samples.length; i++) {
          actualOutputsForSample[i] = sut.forward(trainingData.getInputForSample(i));
        }

        // Expect untrained output not to be matching the expected results (even with a low precision)
        expectOutputOfUntrainedNetworkToNotBeCloseToExpectedOutputs(actualOutputsForSample);

        // Prepare for Training
        sut.setTrainability(true);

        // keep track of the squared prediction loss
        const losses = [];
        const performTrainingRoutineForSample = (i: number): number => {
          sut.forward(trainingData.getInputForSample(i));
          const actualLoss = sut.backward(trainingData.getExpectedOutputForSample(i));
          return actualLoss;
        };

        // Start Training
        for (let i = 0; i < trainingIterations; i++) {
          const ix = getRandomIndex();
          // perform training routine for sample and gather squared loss
          losses[i] = performTrainingRoutineForSample(ix);
        }

        // Get Output of the trained network
        for (let i = 0; i < trainingData.samples.length; i++) {
          actualOutputsForSample[i] = sut.forward(trainingData.getInputForSample(i));
        }

        // Expect trained output to be close to the expected results (with a high precision of 1e-11 resp. 10)
        expectOutputOfTrainedNetworkToBeCloseToExpectedOutputs(actualOutputsForSample, 10);
        // Expect squared error to be near 0
        expect(losses[trainingIterations - 1]).toBeCloseTo(0);
      // }
      });

      it('given fresh instance >> perform iterative training routine with varying alpha >> should output exact expected results after 3.000 iterations', () => {
        // for (let j = 0; j < 10000; j++) {
        const trainingIterations = 3000;
        const alphaMin = 0.001;
        const alphaMax = 0.9;

        sut = new DNN(config);

        // Output container
        const actualOutputsForSample = [];

        // Get Output of the untrained network
        for (let i = 0; i < trainingData.samples.length; i++) {
          actualOutputsForSample[i] = sut.forward(trainingData.getInputForSample(i));
        }

        // Expect untrained output not to be matching the expected results (even with a low precision)
        expectOutputOfUntrainedNetworkToNotBeCloseToExpectedOutputs(actualOutputsForSample);

        // Prepare for Training
        sut.setTrainability(true);

        // keep track of the squared prediction loss
        const losses = [];
        const performTrainingRoutineForSample = (i: number, alpha: number): number => {
          sut.forward(trainingData.getInputForSample(i));
          const actualLoss = sut.backward(trainingData.getExpectedOutputForSample(i), alpha);
          return actualLoss;
        };

        // Start Training
        for (let i = 0; i < trainingIterations; i++) {
          const ix = getRandomIndex();
          // perform training routine for sample and gather squared loss
          const alpha = getRandomAlpha(alphaMin, alphaMax);
          losses[i] = performTrainingRoutineForSample(ix, alpha);
        }

        // Get Output of the trained network
        for (let i = 0; i < trainingData.samples.length; i++) {
          actualOutputsForSample[i] = sut.forward(trainingData.getInputForSample(i));
        }

        // Expect trained output to be close to the expected results (with a high precision of 1e-11 resp. 10)
        expectOutputOfTrainedNetworkToBeCloseToExpectedOutputs(actualOutputsForSample, 10);
        // Expect squared error to be near 0
        expect(losses[trainingIterations - 1]).toBeCloseTo(0);
      // }
      });
    });

    describe('Deep Bayesian Feedforward Neural Network (BNN):', () => {

      it('given fresh instance >> perform iterative training routine >> should output a small squared loss after 10.000 iterations (squared error < 1)', () => {
        // for (let j = 0; j < 10000; j++) {
        const trainingIterations = 10000;

        sut = new BNN(config);
        // Output container
        const actualOutputsForSample = [];

        // Get Output of the untrained network
        for (let i = 0; i < trainingData.samples.length; i++) {
          actualOutputsForSample[i] = sut.forward(trainingData.getInputForSample(i));
        }

        // Expect untrained output not to be matching the expected results (even with a low precision)
        expectOutputOfUntrainedNetworkToNotBeCloseToExpectedOutputs(actualOutputsForSample);

        // Prepare for Training
        sut.setTrainability(true);

        // keep track of the squared prediction loss
        const losses = [];
        const performTrainingRoutineForSample = (i: number): number => {
          sut.forward(trainingData.getInputForSample(i));
          const actualLoss = sut.backward(trainingData.getExpectedOutputForSample(i));
          return actualLoss;
        };

        // Start Training
        for (let i = 0; i < trainingIterations; i++) {
          const ix = getRandomIndex();
          // perform training routine for sample and gather squared loss
          losses[i] = performTrainingRoutineForSample(ix);
        }

        // Get Output of the trained network
        for (let i = 0; i < trainingData.samples.length; i++) {
          actualOutputsForSample[i] = sut.forward(trainingData.getInputForSample(i));
        }

        // Expect trained output to be close to the expected results (with a high precision of 1e-11 resp. 10)
        // expectOutputOfTrainedNetworkToBeCloseToExpectedOutputs(actualOutputsForSample, 3);
        // Expect squared error to be near 0
        expect(losses[trainingIterations - 1]).toBeCloseTo(0, 0);
      // }
      });

      fit('given fresh instance >> perform iterative training routine with varying alpha >> should output a good approximation of expected results after 6.000 iterations (squared error < 1)', () => {
        for (let j = 0; j < 1000; j++) {
        const trainingIterations = 6000;
        const alphaMin = 0.001;
        const alphaMax = 0.1;

        sut = new BNN(config);

        // Output container
        const actualOutputsForSample = [];

        // Get Output of the untrained network
        for (let i = 0; i < trainingData.samples.length; i++) {
          actualOutputsForSample[i] = sut.forward(trainingData.getInputForSample(i));
        }

        // Expect untrained output not to be matching the expected results (even with a low precision)
        expectOutputOfUntrainedNetworkToNotBeCloseToExpectedOutputs(actualOutputsForSample);

        // Prepare for Training
        sut.setTrainability(true);

        // keep track of the squared prediction loss
        const losses = [];
        const performTrainingRoutineForSample = (i: number, alpha: number): number => {
          sut.forward(trainingData.getInputForSample(i));
          const actualLoss = sut.backward(trainingData.getExpectedOutputForSample(i), alpha);
          return actualLoss;
        };

        // Start Training
        for (let i = 0; i < trainingIterations; i++) {
          const ix = getRandomIndex();
          // perform training routine for sample and gather squared loss
          const alpha = getRandomAlpha(alphaMin, alphaMax);
          losses[i] = performTrainingRoutineForSample(ix, alpha);
        }

        // Get Output of the trained network
        for (let i = 0; i < trainingData.samples.length; i++) {
          actualOutputsForSample[i] = sut.forward(trainingData.getInputForSample(i));
        }

        // Expect trained output to be close to the expected results (with a high precision of 1e-11 resp. 10)
        // expectOutputOfTrainedNetworkToBeCloseToExpectedOutputs(actualOutputsForSample, 1);
        // Expect squared error to be near 0
        expect(losses[trainingIterations - 1]).toBeLessThan(0.5, actualOutputsForSample);
      }
      });

      const patchFillRandn = () => {
        spyOn(Utils, 'fillRandn').and.callFake(fillConstOnes);
        spyOn(Utils, 'randf').and.callFake(() => { return 0.009; });
      };

      const fillConstOnes = (arr) => {
        Utils.fillConst(arr, 1);
      };
    });

    const expectOutputOfUntrainedNetworkToNotBeCloseToExpectedOutputs = (actualOutputs: Array<number>) => {
      for (let trainingSample = 0; trainingSample < trainingData.samples.length; trainingSample++) {
        const actual = actualOutputs[trainingSample];
        const expected = trainingData.getExpectedOutputForSample(trainingSample);
        for (let dataPoint = 0; dataPoint < expected.length; dataPoint++) {
          expect(actual[dataPoint]).not.toBeCloseTo(expected[dataPoint], '[untrained] Actual output for Sample ' + trainingSample + ' at position ' + dataPoint + ' was close to ' + expected[dataPoint]);
        }
      }
    };

    const expectOutputOfTrainedNetworkToBeCloseToExpectedOutputs = (actualOutputs: Array<number>, precision?: number): void => {
      for (let trainingSample = 0; trainingSample < trainingData.samples.length; trainingSample++) {
        const actual = actualOutputs[trainingSample];
        const expected = trainingData.getExpectedOutputForSample(trainingSample);
        for (let dataPoint = 0; dataPoint < expected.length; dataPoint++) {
          const errorMessage = '[trained] Actual output for Sample ' + trainingSample + ' at position ' + dataPoint + ' was not close to ' + expected[dataPoint];
          if (precision) { expect(actual[dataPoint]).toBeCloseTo(expected[dataPoint], precision, errorMessage); }
          else { expect(actual[dataPoint]).toBeCloseTo(expected[dataPoint], errorMessage); }
        }
      }
    };

    const getRandomIndex = (): number => {
      return Utils.randi(0, trainingData.samples.length);
    };

    const getRandomAlpha = (min: number, max: number): number => {
      return Utils.randf(min, max);
    };
  });
});
