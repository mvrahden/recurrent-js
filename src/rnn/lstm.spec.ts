import { LSTM } from '..';

describe('Long Short-Term Memory Network (LSTM):', () => {

  let sut: LSTM;

  describe('Instantiation:', () => {

    describe('Configuration with NetOpts:', () => {

      const config = { architecture: { inputSize: 2, hiddenUnits: [3, 4], outputSize: 3 } };

      beforeEach(() => {
        sut = new LSTM(config);
      });

      describe('Hidden Layer:', () => {

        it('fresh instance >> on creation >> model should hold hidden layer containing weight matrices with expected dimensions', () => {
          expectHiddenWeightMatricesToHaveColsOfSizeOfPrecedingLayerAndRowsOfConfiguredLength(2, [3, 4]);
        });

        it('fresh instance >> on creation >> model should hold hidden layer containing weight matrices for preceding activations with expected quadratic dimensions', () => {
          expectHiddenWeightMatricesForPrecedingActivationToHaveQuadraticDimensionsAccordingToHiddenUnits([3, 4]);
        });

        it('fresh instance >> on creation >> model should hold hidden layer containing bias matrices with expected dimensions', () => {
          expectHiddenBiasMatricesToHaveRowsOfSizeOfPrecedingLayerAndColsOfSize1(2, [3, 4]);
        });

        const expectHiddenWeightMatricesToHaveColsOfSizeOfPrecedingLayerAndRowsOfConfiguredLength = (inputSize: number, hiddenUnits: Array<number>) => {
          let precedingLayerSize = inputSize;
          let expectedRows, expectedCols;
          for (let i = 0; i < config.architecture.hiddenUnits.length; i++) {
            expectedRows = hiddenUnits[i];
            expectedCols = precedingLayerSize;
            expect(sut.model.hidden.input.Wx[i].rows).toBe(expectedRows);
            expect(sut.model.hidden.input.Wx[i].cols).toBe(expectedCols);

            expect(sut.model.hidden.output.Wx[i].rows).toBe(expectedRows);
            expect(sut.model.hidden.output.Wx[i].cols).toBe(expectedCols);

            expect(sut.model.hidden.forget.Wx[i].rows).toBe(expectedRows);
            expect(sut.model.hidden.forget.Wx[i].cols).toBe(expectedCols);

            expect(sut.model.hidden.cell.Wx[i].rows).toBe(expectedRows);
            expect(sut.model.hidden.cell.Wx[i].cols).toBe(expectedCols);
            precedingLayerSize = expectedRows;
          }
        };

        const expectHiddenWeightMatricesForPrecedingActivationToHaveQuadraticDimensionsAccordingToHiddenUnits = (expected: Array<number>) => {
          for (let i = 0; i < config.architecture.hiddenUnits.length; i++) {
            expect(sut.model.hidden.input.Wh[i].rows).toBe(expected[i]);
            expect(sut.model.hidden.input.Wh[i].cols).toBe(expected[i]);

            expect(sut.model.hidden.output.Wh[i].rows).toBe(expected[i]);
            expect(sut.model.hidden.output.Wh[i].cols).toBe(expected[i]);

            expect(sut.model.hidden.forget.Wh[i].rows).toBe(expected[i]);
            expect(sut.model.hidden.forget.Wh[i].cols).toBe(expected[i]);

            expect(sut.model.hidden.cell.Wh[i].rows).toBe(expected[i]);
            expect(sut.model.hidden.cell.Wh[i].cols).toBe(expected[i]);
          }
        };

        const expectHiddenBiasMatricesToHaveRowsOfSizeOfPrecedingLayerAndColsOfSize1 = (inputSize: number, hiddenUnits: Array<number>) => {
          let expectedRows;
          const expectedCols = 1;
          for (let i = 0; i < config.architecture.hiddenUnits.length; i++) {
            expectedRows = hiddenUnits[i];
            expect(sut.model.hidden.input.bh[i].rows).toBe(expectedRows);
            expect(sut.model.hidden.input.bh[i].cols).toBe(expectedCols);

            expect(sut.model.hidden.output.bh[i].rows).toBe(expectedRows);
            expect(sut.model.hidden.output.bh[i].cols).toBe(expectedCols);

            expect(sut.model.hidden.forget.bh[i].rows).toBe(expectedRows);
            expect(sut.model.hidden.forget.bh[i].cols).toBe(expectedCols);

            expect(sut.model.hidden.cell.bh[i].rows).toBe(expectedRows);
            expect(sut.model.hidden.cell.bh[i].cols).toBe(expectedCols);
          }
        };
      });
    });
  });

  describe('Backpropagation:', () => {
    
    const config = { architecture: { inputSize: 2, hiddenUnits: [3, 4], outputSize: 3 } };

    beforeEach(() => {
      sut = new LSTM(config);

      spyOnUpdateMethods();
    });

    describe('Update:', () => {

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
            expect(sut.model.hidden.input.Wx[i].update).toHaveBeenCalled();
            expect(sut.model.hidden.input.Wh[i].update).toHaveBeenCalled();
            expect(sut.model.hidden.input.bh[i].update).toHaveBeenCalled();

            expect(sut.model.hidden.output.Wx[i].update).toHaveBeenCalled();
            expect(sut.model.hidden.output.Wh[i].update).toHaveBeenCalled();
            expect(sut.model.hidden.output.bh[i].update).toHaveBeenCalled();

            expect(sut.model.hidden.forget.Wx[i].update).toHaveBeenCalled();
            expect(sut.model.hidden.forget.Wh[i].update).toHaveBeenCalled();
            expect(sut.model.hidden.forget.bh[i].update).toHaveBeenCalled();

            expect(sut.model.hidden.cell.Wx[i].update).toHaveBeenCalled();
            expect(sut.model.hidden.cell.Wh[i].update).toHaveBeenCalled();
            expect(sut.model.hidden.cell.bh[i].update).toHaveBeenCalled();
          }
        };
        
        const expectUpdateOfLayersMethodsToHaveBeenCalledWithValue = (value: number) => {
          for (let i = 0; i < config.architecture.hiddenUnits.length; i++) {
            expect(sut.model.hidden.input.Wx[i].update).toHaveBeenCalledWith(value);
            expect(sut.model.hidden.input.Wh[i].update).toHaveBeenCalledWith(value);
            expect(sut.model.hidden.input.bh[i].update).toHaveBeenCalledWith(value);

            expect(sut.model.hidden.output.Wx[i].update).toHaveBeenCalledWith(value);
            expect(sut.model.hidden.output.Wh[i].update).toHaveBeenCalledWith(value);
            expect(sut.model.hidden.output.bh[i].update).toHaveBeenCalledWith(value);

            expect(sut.model.hidden.forget.Wx[i].update).toHaveBeenCalledWith(value);
            expect(sut.model.hidden.forget.Wh[i].update).toHaveBeenCalledWith(value);
            expect(sut.model.hidden.forget.bh[i].update).toHaveBeenCalledWith(value);

            expect(sut.model.hidden.cell.Wx[i].update).toHaveBeenCalledWith(value);
            expect(sut.model.hidden.cell.Wh[i].update).toHaveBeenCalledWith(value);
            expect(sut.model.hidden.cell.bh[i].update).toHaveBeenCalledWith(value);
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
    });

    const spyOnUpdateMethods = () => {
      for (let i = 0; i < config.architecture.hiddenUnits.length; i++) {
        spyOn(sut.model.hidden.input.Wx[i], 'update');
        spyOn(sut.model.hidden.input.Wh[i], 'update');
        spyOn(sut.model.hidden.input.bh[i], 'update');

        spyOn(sut.model.hidden.output.Wx[i], 'update');
        spyOn(sut.model.hidden.output.Wh[i], 'update');
        spyOn(sut.model.hidden.output.bh[i], 'update');

        spyOn(sut.model.hidden.forget.Wx[i], 'update');
        spyOn(sut.model.hidden.forget.Wh[i], 'update');
        spyOn(sut.model.hidden.forget.bh[i], 'update');

        spyOn(sut.model.hidden.cell.Wx[i], 'update');
        spyOn(sut.model.hidden.cell.Wh[i], 'update');
        spyOn(sut.model.hidden.cell.bh[i], 'update');
      }

      spyOn(sut.model.decoder.Wh, 'update');
      spyOn(sut.model.decoder.b, 'update');
    };
  });
});
