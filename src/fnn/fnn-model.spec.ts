import { DNN } from '../.';

/**
 * Tests are being executed on a DNN instance.
 * DNN Class fully delegates instantiation to FNNModel Class.
 */
describe('Feedforward Neural Network Model:', () => {
  let sut: DNN;

  describe('Instantiation:', () => {
    
    describe('Configuration with NetOpts:', () => {

      const config = {
        inputSize: 2, hiddenUnits: [3, 4], outputSize: 3
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
          expectHiddenWeightMatricesToHaveColsOfSizeOfPrecedingLayerAndRowsOfConfiguredLength(2, [3, 4]);
        });

        it('fresh instance >> on creation >> model should hold hidden layer containing bias matrices with expected dimensions', () => {
          expectHiddenBiasMatricesToHaveRowsOfSizeOfPrecedingLayerAndColsOfSize1(2, [3, 4]);
        });

        const expectHiddenWeightMatricesToHaveColsOfSizeOfPrecedingLayerAndRowsOfConfiguredLength = (inputSize: number, hiddenUnits: Array<number>) => {
          let precedingLayerSize = inputSize;
          let expectedRows, expectedCols;
          for (let i = 0; i < sut.model.hidden.Wh.length; i++) {
            expectedRows = hiddenUnits[i];
            expectedCols = precedingLayerSize;
            expect(sut.model.hidden.Wh[i].rows).toBe(expectedRows);
            expect(sut.model.hidden.Wh[i].cols).toBe(expectedCols);
            precedingLayerSize = expectedRows;
          }
        };

        const expectHiddenBiasMatricesToHaveRowsOfSizeOfPrecedingLayerAndColsOfSize1 = (inputSize: number, hiddenUnits: Array<number>) => {
          let expectedRows;
          let expectedCols = 1;
          for (let i = 0; i < sut.model.hidden.bh.length; i++) {
            expectedRows = hiddenUnits[i];
            expect(sut.model.hidden.bh[i].rows).toBe(expectedRows);
            expect(sut.model.hidden.bh[i].cols).toBe(expectedCols);
          }
        };
      });
    
      describe('Decoder Layer:', () => {

        it('fresh instance >> on creation >> model should hold decoder layer containing weight matrix with given dimensions', () => {
          sut = new DNN(config);

          expectDecoderWeightMatrixToHaveDimensionsOf(3, 4);
        });

        it('fresh instance >> on creation >> model should hold decoder layer containing bias matrix with given dimensions', () => {
          sut = new DNN(config);

          expectDecoderBiasMatrixToHaveDimensionsOf(3, 1);
        });
  
        const expectDecoderWeightMatrixToHaveDimensionsOf = (expectedRows: number, expectedCols: number) => {
          expect(sut.model.decoder.Wh.rows).toBe(expectedRows);
          expect(sut.model.decoder.Wh.cols).toBe(expectedCols);
        };
  
        const expectDecoderBiasMatrixToHaveDimensionsOf = (expectedRows: number, expectedCols: number) => {
          expect(sut.model.decoder.b.rows).toBe(expectedRows);
          expect(sut.model.decoder.b.cols).toBe(expectedCols);
        };
      });
    });

    describe('Configuration with JSON Object', () => {
      
    });
  });
  
  describe('Backpropagation:', () => {
  
    beforeEach(() => {
      const config = { inputSize: 2, hiddenUnits: [3, 4], outputSize: 3 };
      sut = new DNN(config);
  
      spyOnUpdateMethods();
    });
  
    describe('Backward:', () => {
  
      describe('Hidden Layer:', () => {

        it('after forward pass >> backward >> should call graph to execute', () => {

        })
        
      });
  
      describe('Decoder Layer:', () => {
  
      });
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
          expect(sut.model.hidden.Wh[0].update).toHaveBeenCalled();
          expect(sut.model.hidden.Wh[1].update).toHaveBeenCalled();
          expect(sut.model.hidden.bh[0].update).toHaveBeenCalled();
          expect(sut.model.hidden.bh[1].update).toHaveBeenCalled();
        };
  
        const expectUpdateOfLayersMethodsToHaveBeenCalledWithValue = (value: number) => {
          expect(sut.model.hidden.Wh[0].update).toHaveBeenCalledWith(value);
          expect(sut.model.hidden.Wh[1].update).toHaveBeenCalledWith(value);
          expect(sut.model.hidden.bh[0].update).toHaveBeenCalledWith(value);
          expect(sut.model.hidden.bh[1].update).toHaveBeenCalledWith(value);
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
      spyOn(sut.model.hidden.Wh[0], 'update');
      spyOn(sut.model.hidden.Wh[1], 'update');
      spyOn(sut.model.hidden.bh[0], 'update');
      spyOn(sut.model.hidden.bh[1], 'update');
  
      spyOn(sut.model.decoder.Wh, 'update');
      spyOn(sut.model.decoder.b, 'update');
    };
  });
});

