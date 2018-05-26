import { Utils, DNN } from '..';
import { ANN } from './ann';


describe('Reproducible examples with patched Neural Networks:', () => {

  let sut: ANN;
  const config = {
    architecture: { inputSize: 2, hiddenUnits: [3, 4], outputSize: 3 },
    training: { loss: [1e-11, 1e-11, 1e-11] }
  };

  describe('Deep Feedforward Neural Network (DNN):', () => {

    beforeEach(() => {
      patchFillRandn();
    });

    it('given fresh (patched) instance >> forward([0,1]), backward([0, 1, 0]), forward([1,0]), backward([0, 1, 0]) >> should output expected results after 20.000 iterations', () => {
      sut = new DNN(config);
      // Output container
      let output1 = null;
      let output2 = null;
      
      // Training data
      const input1 = [0, 1];
      const expectedResultsForInput1 = [0, 1, 0];
      const input2 = [1, 0];
      const expectedResultsForInput2 = [1, 0, 1];

      // Untrained Output
      output1 = sut.forward(input1);
      output2 = sut.forward(input2);

      // Expect untrained output not to be matching the expected results (with a low precision of 1)
      expect(output1[0]).not.toBeCloseTo(expectedResultsForInput1[0], 1, '[untrained] output1 was close to');
      expect(output1[1]).not.toBeCloseTo(expectedResultsForInput1[1], 1, '[untrained] output1 was close to');
      expect(output1[2]).not.toBeCloseTo(expectedResultsForInput1[2], 1, '[untrained] output1 was close to');

      expect(output2[0]).not.toBeCloseTo(expectedResultsForInput2[0], 1, '[untrained] output2 was not close to');
      expect(output2[1]).not.toBeCloseTo(expectedResultsForInput2[1], 1, '[untrained] output2 was not close to');
      expect(output2[2]).not.toBeCloseTo(expectedResultsForInput2[2], 1, '[untrained] output2 was not close to');

      // Prepare for Training
      sut.setTrainability(true);

      // Start Training
      for (let i = 0; i < 20000; i++) {
        sut.forward(input1);
        sut.backward(expectedResultsForInput1);
        
        sut.forward(input2);
        sut.backward(expectedResultsForInput2);
      }

      output1 = sut.forward(input1);
      output2 = sut.forward(input2);

      // Expect trained output to be close to the expected results (with a high precision of 1e-11 resp. 10)
      expect(output1[0]).toBeCloseTo(expectedResultsForInput1[0], 10, '[trained] output1 was close to');
      expect(output1[1]).toBeCloseTo(expectedResultsForInput1[1], 10, '[trained] output1 was close to');
      expect(output1[2]).toBeCloseTo(expectedResultsForInput1[2], 10, '[trained] output1 was close to');

      expect(output2[0]).toBeCloseTo(expectedResultsForInput2[0], 10, '[trained] output2 was close to');
      expect(output2[1]).toBeCloseTo(expectedResultsForInput2[1], 10, '[trained] output2 was close to');
      expect(output2[2]).toBeCloseTo(expectedResultsForInput2[2], 10, '[trained] output2 was close to');

      // GAINED RESULTS:
      // output1 === Float64Array[
      //   -9.887809321318386e-12,
      //   0.9999999999930264,
      //   -9.887809321318386e-12]

      // output2 === Float64Array[
      //   1.0000000000025895,
      //   9.155343150268891e-12,
      //   1.0000000000025895]
    });

    it('given fresh (patched) instance >> forward([0,1]), backward([0, 1, 0]), forward([1,0]), backward([0, 1, 0]) >> should output expected results after 20.000 iterations', () => {
      sut = new DNN(config);
      // Output container
      let output1 = null;
      let output2 = null;

      // Training data
      const input1 = [0, 1];
      const expectedResultsForInput1 = [0, 1, 0];
      const input2 = [1, 0];
      const expectedResultsForInput2 = [1, 0, 1];

      // Untrained Output
      output1 = sut.forward(input1);
      output2 = sut.forward(input2);

      // Expect untrained output not to be matching the expected results (with a low precision of 1)
      expect(output1[0]).not.toBeCloseTo(expectedResultsForInput1[0], 1, '[untrained] output1 was close to');
      expect(output1[1]).not.toBeCloseTo(expectedResultsForInput1[1], 1, '[untrained] output1 was close to');
      expect(output1[2]).not.toBeCloseTo(expectedResultsForInput1[2], 1, '[untrained] output1 was close to');

      expect(output2[0]).not.toBeCloseTo(expectedResultsForInput2[0], 1, '[untrained] output2 was not close to');
      expect(output2[1]).not.toBeCloseTo(expectedResultsForInput2[1], 1, '[untrained] output2 was not close to');
      expect(output2[2]).not.toBeCloseTo(expectedResultsForInput2[2], 1, '[untrained] output2 was not close to');

      // Prepare for Training
      sut.setTrainability(true);

      // Start Training
      let alpha;
      for (let i = 0; i < 20000; i++) {
        alpha = Utils.randf(0.005, 0.01);
        sut.forward(input1);
        sut.backward(expectedResultsForInput1, alpha);

        sut.forward(input2);
        sut.backward(expectedResultsForInput2, alpha);
      }

      output1 = sut.forward(input1);
      output2 = sut.forward(input2);

      // Expect trained output to be close to the expected results (with a high precision of 1e-11 resp. 10)
      expect(output1[0]).toBeCloseTo(expectedResultsForInput1[0], 10, '[trained] output1 was close to');
      expect(output1[1]).toBeCloseTo(expectedResultsForInput1[1], 10, '[trained] output1 was close to');
      expect(output1[2]).toBeCloseTo(expectedResultsForInput1[2], 10, '[trained] output1 was close to');

      expect(output2[0]).toBeCloseTo(expectedResultsForInput2[0], 10, '[trained] output2 was close to');
      expect(output2[1]).toBeCloseTo(expectedResultsForInput2[1], 10, '[trained] output2 was close to');
      expect(output2[2]).toBeCloseTo(expectedResultsForInput2[2], 10, '[trained] output2 was close to');

      // GAINED RESULTS:
      // output1 === Float64Array[
      //   - 1.9799675787801618e-11,
      //   0.9999999999868354,
      //   -1.9799675787801618e-11]

      // output2 === Float64Array[
      //   1.0000000000179021,
      //   1.1319611914473171e-11,
      //   1.0000000000179021 ]
    });

    const patchFillRandn = () => {
      spyOn(Utils, 'fillRandn').and.callFake(fillConstOnes);
      spyOn(Utils, 'randf').and.callFake(() => { return 0.009; });
    };
  
    const fillConstOnes = (arr) => {
      Utils.fillConst(arr, 1);
    };
  });
});
