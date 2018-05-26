import { DNN, Mat, NetOpts, Utils, Graph } from '..';

describe('Deep Neural Network (DNN):', () => {

  let sut: DNN;

  describe('Instantiation:', () => {

    describe('Configuration with NetOpts:', () => {

      const config = {
        architecture: { inputSize: 2, hiddenUnits: [3, 4], outputSize: 3 }
      };

      beforeEach(() => {
        sut = new DNN(config);
      });

      it('fresh instance >> on creation >> should hold model with hidden layer, containing arrays of weight and bias matrices', () => {
        expect(sut.model).toBeDefined();
        expect(sut.model.hidden).toBeDefined();
        expect(sut.model.hidden.Wh).toBeDefined();
        expect(sut.model.hidden.Wh.length).toBe(2);
        expect(sut.model.hidden.bh).toBeDefined();
        expect(sut.model.hidden.bh.length).toBe(2);
      });

      it('fresh instance >> on creation >> should hold model with decoder layer, containing weight and bias matrices', () => {
        expect(sut.model.decoder).toBeDefined();
        expect(sut.model.decoder.Wh).toBeDefined();
        expect(sut.model.decoder.b).toBeDefined();
      });

      describe('Hidden Layer:', () => {

        it('fresh instance >> on creation >> model should hold hidden layer containing weight matrices with expected dimensions', () => {
          expectHiddenStatelessWeightMatricesToHaveColsOfSizeOfPrecedingLayerAndRowsOfConfiguredLength(2, [3, 4]);
        });

        it('fresh instance >> on creation >> model should hold hidden layer containing bias matrices with expected dimensions', () => {
          expectHiddenBiasMatricesToHaveRowsOfSizeOfPrecedingLayerAndColsOfSize1(2, [3, 4]);
        });

        const expectHiddenStatelessWeightMatricesToHaveColsOfSizeOfPrecedingLayerAndRowsOfConfiguredLength = (inputSize: number, hiddenUnits: Array<number>) => {
          let precedingLayerSize = inputSize;
          let expectedRows, expectedCols;
          for (let i = 0; i < config.architecture.hiddenUnits.length; i++) {
            expectedRows = hiddenUnits[i];
            expectedCols = precedingLayerSize;
            expect(sut.model.hidden.Wh[i].rows).toBe(expectedRows);
            expect(sut.model.hidden.Wh[i].cols).toBe(expectedCols);
            precedingLayerSize = expectedRows;
          }
        };

        const expectHiddenBiasMatricesToHaveRowsOfSizeOfPrecedingLayerAndColsOfSize1 = (inputSize: number, hiddenUnits: Array<number>) => {
          let expectedRows;
          const expectedCols = 1;
          for (let i = 0; i < config.architecture.hiddenUnits.length; i++) {
            expectedRows = hiddenUnits[i];
            expect(sut.model.hidden.bh[i].rows).toBe(expectedRows);
            expect(sut.model.hidden.bh[i].cols).toBe(expectedCols);
          }
        };
      });
    });
  });

  describe('Backpropagation:', () => {

    const config = { architecture: { inputSize: 2, hiddenUnits: [3, 4], outputSize: 3 }, training: { alpha: 0.01, loss: [1e6, 1e6, 1e6] } };

    describe('Update:', () => {

      beforeEach(() => {
        sut = new DNN(config);

        spyOnUpdateMethods();
      });

      describe('Hidden Layer:', () => {

        it('fresh instance >> update >> should call update methods of weight and bias matrices of all hidden layer', () => {
          sut.update(0.01);

          expectUpdateOfLayersMethodsToHaveBeenCalled();
        });

        it('fresh instance >> update >> should call update methods of weight and bias matrices of all hidden layer with given value', () => {
          sut.update(0.01);

          expectUpdateOfLayersMethodsToHaveBeenCalledWithValue(0.01);
        });

        const expectUpdateOfLayersMethodsToHaveBeenCalled = () => {
          for (let i = 0; i < config.architecture.hiddenUnits.length; i++) {
            expect(sut.model.hidden.Wh[i].update).toHaveBeenCalled();
            expect(sut.model.hidden.bh[i].update).toHaveBeenCalled();
          }
        };

        const expectUpdateOfLayersMethodsToHaveBeenCalledWithValue = (value: number) => {
          for (let i = 0; i < config.architecture.hiddenUnits.length; i++) {
            expect(sut.model.hidden.Wh[i].update).toHaveBeenCalledWith(value);
            expect(sut.model.hidden.bh[i].update).toHaveBeenCalledWith(value);
          }
        };
      });

      describe('Decoder Layer:', () => {

        it('fresh instance >> update >> should call update methods of weight and bias matrices of decoder layer', () => {
          sut.update(0.01);

          expectUpdateOfLayersMethodsToHaveBeenCalled();
        });

        it('fresh instance >> update >> should call update methods of weight and bias matrices of decoder layer with given value', () => {
          sut.update(0.01);

          expectUpdateOfLayersMethodsToHaveBeenCalledWithValue(0.01);
        });

        const expectUpdateOfLayersMethodsToHaveBeenCalled = () => {
          expect(sut.model.decoder.Wh.update).toHaveBeenCalled();
          expect(sut.model.decoder.b.update).toHaveBeenCalled();
        };

        const expectUpdateOfLayersMethodsToHaveBeenCalledWithValue = (value: number) => {
          expect(sut.model.decoder.Wh.update).toHaveBeenCalledWith(value);
          expect(sut.model.decoder.b.update).toHaveBeenCalledWith(value);
        };
      });

      const spyOnUpdateMethods = () => {
        for (let i = 0; i < config.architecture.hiddenUnits.length; i++) {
          spyOn(sut.model.hidden.Wh[i], 'update');
          spyOn(sut.model.hidden.bh[i], 'update');
        }

        spyOn(sut.model.decoder.Wh, 'update');
        spyOn(sut.model.decoder.b, 'update');
      };
    });


  });

  describe('Forward Pass:', () => {

    const netOpts: NetOpts = { architecture: { inputSize: 2, hiddenUnits: [3, 4], outputSize: 3 } };
    let sut: DNN;
    let input: Array<number>;

    beforeEach(() => {
      patchFillRandn();
      sut = new DNN(netOpts);
      input = [0, 1];
    });

    it('given fresh network instance and some input vector >> forward pass >> should call activation function as often as number of hidden layer', () => {
      patchNetworkGraph();
      sut.forward(input);

      expect(sut['graph'].relu).toHaveBeenCalledTimes(2);
    });

    it('given fresh network instance and some input vector >> forward pass >> should return output with given dimensions', () => {
      const output = sut.forward(input);

      expect(output.length).toBe(3);
    });

    it('given fresh network instance and some input vector >> forward pass >> should return Array filled with 12', () => {
      const output = sut.forward(input);

      expect(output[0]).toBe(12);
      expect(output[1]).toBe(12);
      expect(output[2]).toBe(12);
    });

    const patchFillRandn = () => {
      spyOn(Utils, 'fillRandn').and.callFake(fillConstOnes);
    };

    const patchNetworkGraph = () => {
      spyOn(sut['graph'], 'relu').and.callFake(fillMatConstOnes);
    };

    const fillConstOnes = (arr) => {
      Utils.fillConst(arr, 1);
    };

    const fillMatConstOnes = (mat) => {
      const out = new Mat(mat.rows, 1);
      fillConstOnes(out.w);
      return out;
    };
  });
});

