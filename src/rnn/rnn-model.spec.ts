import { RNN } from '../.';

/**
 * Tests are being executed on a RNN instance.
 * RNN Class fully delegates instantiation to FNNModel Class.
 */
describe('Recurrent Neural Network Model:', () => {
  let sut: RNN;

  describe('Instantiation:', () => {

    describe('Configuration with NetOpts:', () => {

      const config = {
        inputSize: 2, hiddenUnits: [3, 4], outputSize: 3
      };

      beforeEach(() => {
        sut = new RNN(config);
      });

      describe('Hidden Layer:', () => {
        // Hidden Layer are responsibility of concrete RNN implementations
      });

      describe('Decoder Layer:', () => {

        it('fresh instance >> on creation >> model should hold decoder layer containing weight matrix with given dimensions', () => {
          sut = new RNN(config);

          expectDecoderWeightMatrixToHaveDimensionsOf(3, 4);
        });

        it('fresh instance >> on creation >> model should hold decoder layer containing bias matrix with given dimensions', () => {
          sut = new RNN(config);

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
});