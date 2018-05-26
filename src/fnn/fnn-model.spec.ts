import { DNN } from '../.';

/**
 * Tests are being executed on a DNN instance.
 * DNN Class fully delegates instantiation to FNNModel Class.
 */
describe('Feedforward Neural Network Model:', () => {
  let sut: DNN;

  describe('Instantiation:', () => {
    
    describe('Configuration with NetOpts:', () => {

      describe('Initialization of NetOpts-Properties:', () => {

        it('given NetOpts with architecture >> on creation >> should define `sut.architecture` accordingly', () => {
          const config = { architecture: { inputSize: 2, hiddenUnits: [3, 4], outputSize: 3 } };

          sut = new DNN(config);

          expect(sut['architecture']).toBeDefined();
          expect(sut['architecture'].inputSize).toBe(2);
          expect(sut['architecture'].hiddenUnits[0]).toBe(3);
          expect(sut['architecture'].hiddenUnits[1]).toBe(4);
          expect(sut['architecture'].outputSize).toBe(3);
        });

        it('given NetOpts without `training` >> on creation >> should define `sut.training` with default values', () => {
          const config = { architecture: { inputSize: 2, hiddenUnits: [3, 4], outputSize: 3 } };

          sut = new DNN(config);

          expect(sut['training']).toBeDefined();
          expect(sut['training'].alpha).toBe(0.01);
          expect(sut['training'].loss).toBeDefined();
          expect(sut['training'].loss.rows).toBe(1);
          expect(sut['training'].loss.cols).toBe(3);
          expect(sut['training'].loss.w[0]).toBe(1e-6);
          expect(sut['training'].loss.w[1]).toBe(1e-6);
          expect(sut['training'].loss.w[2]).toBe(1e-6);
        });

        it('given NetOpts with `training` >> on creation >> should define `sut.training` with given values', () => {
          const config = { architecture: { inputSize: 2, hiddenUnits: [3, 4], outputSize: 3 }, training: { loss: 1 } };

          sut = new DNN(config);

          expect(sut['training']).toBeDefined();
          expect(sut['training'].alpha).toBe(0.01);
          expect(sut['training'].loss).toBeDefined();
          expect(sut['training'].loss.rows).toBe(1);
          expect(sut['training'].loss.cols).toBe(3);
          expect(sut['training'].loss.w[0]).toBe(1);
          expect(sut['training'].loss.w[1]).toBe(1);
          expect(sut['training'].loss.w[2]).toBe(1);
        });

        it('given NetOpts with `training` >> on creation >> should define `sut.training` with given values', () => {
          const config = { architecture: { inputSize: 2, hiddenUnits: [3, 4], outputSize: 3 }, training: { alpha: 1 } };

          sut = new DNN(config);

          expect(sut['training']).toBeDefined();
          expect(sut['training'].alpha).toBe(1);
          expect(sut['training'].loss).toBeDefined();
          expect(sut['training'].loss.rows).toBe(1);
          expect(sut['training'].loss.cols).toBe(3);
          expect(sut['training'].loss.w[0]).toBe(1e-6);
          expect(sut['training'].loss.w[1]).toBe(1e-6);
          expect(sut['training'].loss.w[2]).toBe(1e-6);
        });

        it('given NetOpts with `training` and loss as Array >> on creation >> should define `sut.training` with given values', () => {
          const config = { architecture: { inputSize: 2, hiddenUnits: [3, 4], outputSize: 3 }, training: { loss: [1, 2, 3] } };

          sut = new DNN(config);

          expect(sut['training']).toBeDefined();
          expect(sut['training'].alpha).toBe(0.01);
          expect(sut['training'].loss).toBeDefined();
          expect(sut['training'].loss.rows).toBe(1);
          expect(sut['training'].loss.cols).toBe(3);
          expect(sut['training'].loss.w[0]).toBe(1);
          expect(sut['training'].loss.w[1]).toBe(2);
          expect(sut['training'].loss.w[2]).toBe(3);
        });
      });
    
      describe('Decoder Layer:', () => {

        const config = { architecture: { inputSize: 2, hiddenUnits: [3, 4], outputSize: 3 } };

        beforeEach(() => {
          sut = new DNN(config);
        });

        it('fresh instance >> on creation >> model should hold decoder layer containing weight matrix with given dimensions', () => {
          expectDecoderWeightMatrixToHaveDimensionsOf(3, 4);
        });

        it('fresh instance >> on creation >> model should hold decoder layer containing bias matrix with given dimensions', () => {
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

    const config = { architecture: { inputSize: 2, hiddenUnits: [3, 4], outputSize: 3 } };
  
    beforeEach(() => {
      sut = new DNN(config);
  
      spyOnUpdateMethods();
    });
  
    describe('Backward:', () => {
  
      describe('Hidden Layer:', () => {

        it('after forward pass >> backward >> should call graph to execute', () => {

        });
        
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

