import { Mat, Utils } from '..';
import { MatOps } from './mat-ops';

describe('MatOps:', () => {

  const sut = MatOps;
  let actual: Mat;
  let expected: Mat;
  let mat1: Mat;

  beforeEach(() => {
    mat1 = new Mat(2, 4);
    mat1.setFrom([1, 4, 6, 10, 2, 7, 5, 3]);
  });

  describe('Single Matrix Operations:', ()  => {

    describe('Row Pluck:', () => {

      let rowIndex: number;

      beforeEach(() => {
        rowIndex = 0;
      });

      it('given a matrix >> rowPluck >> should return new instance of matrix-object (reference)', () => {
        actual = sut.rowPluck(mat1, rowIndex);

        expectOperationHasReturnedNewInstance();
      });

      it('given a matrix with dimensions (2,4) >> rowPluck >> should return matrix with dimensions (4,1)', () => {
        actual = sut.rowPluck(mat1, rowIndex);

        expectOperationHasReturnedMatrixWithDimensions(4, 1);
      });

      it('given a matrix with dimensions (2,4) and incompatible rowIndex >> rowPluck >> should throw error', () => {
        const incompatible = 2;

        const callFunction = () => { sut.rowPluck(mat1, incompatible); };

        expect(callFunction).toThrowError('[class:MatOps] rowPluck: dimensions misaligned');
      });

      it('given a matrix >> rowPluck >> should return matrix with expected content', () => {
        actual = sut.rowPluck(mat1, rowIndex);

        expected = new Mat(4, 1);
        expectRowpluckHasReturnedMatrixWithContent([1, 4, 6, 10]);
      });

      const expectRowpluckHasReturnedMatrixWithContent = (content: Array<number>) => {
        expected.setFrom(content);

        for (let i = 0; i < actual.w.length; i++) {
          expect(actual.w[i]).toBe(expected.w[i]);
        }
      };
    });

    describe('Gauss noise-addition:', () => {

      let std: Mat;

      beforeEach(() => {
        std = new Mat(2, 4);
        std.setFrom([0.1, 0.2, 0.02, 0.5, 1, 0.01, 0, 1]);
        spyOn(Utils, 'randn').and.callFake((mu: number, std: number) => { return mu + std; });
      });

      it('given a matrix >> gauss >> should return new instance of matrix-object (reference)', () => {
        actual = sut.gauss(mat1, std);

        expectOperationHasReturnedNewInstance();
      });

      it('given a matrix with dimensions (2,4) >> gauss >> should return matrix with dimensions (2,4)', () => {
        actual = sut.gauss(mat1, std);

        expectOperationHasReturnedMatrixWithDimensions(2, 4);
      });

      it('given a matrix with dimensions (2,4) and (3,3) >> gauss >> should throw error', () => {
        const incompatible = new Mat(3, 3);

        const callFunction = () => { sut.gauss(mat1, incompatible); };

        expect(callFunction).toThrowError('[class:MatOps] gauss: dimensions misaligned');
      });

      it('given a matrix >> gauss >> should return matrix with expected content', () => {
        actual = sut.gauss(mat1, std);

        expected = new Mat(2, 4);
        expectGaussHasReturnedMatrixWithGaussianDistributedContent([1, 4, 6, 10, 2, 7, 5, 3], [0.1, 0.2, 0.02, 0.5, 1, 0.01, 0, 1]);
      });

      const expectGaussHasReturnedMatrixWithGaussianDistributedContent = (content: Array<number>, std: Array<number>) => {
        expected.setFrom(content);

        for (let i = 0; i < actual.w.length; i++) {
          expect(actual.w[i]).toBe(expected.w[i] + std[i]);
        }
      };
    });

    describe('Monadic Operations', () => {

      describe('Hyperbolic Tangens', () => {

        it('given a matrix >> tanh >> should return new instance of matrix-object (reference)', () => {
          actual = sut.tanh(mat1);

          expectOperationHasReturnedNewInstance();
        });

        it('given a matrix with dimensions (2,4) >> tanh >> should return matrix with dimensions (2,4)', () => {
          actual = sut.tanh(mat1);

          expectOperationHasReturnedMatrixWithDimensions(2, 4);
        });

        it('given a matrix with dimensions (2,4) >> tanh >> should return matrix with dimensions (2,4)', () => {
          actual = sut.tanh(mat1);

          expectMonadicOperationHasReturnedMatrixWithContent([0.761594, 0.999329, 0.999987, 0.999999, 0.964027, 0.999998, 0.999909, 0.995054]);
        });
      });
  
      describe('Sigmoid', () => {

        it('given a matrix >> sig >> should return new instance of matrix-object (reference)', () => {
          actual = sut.sig(mat1);

          expectOperationHasReturnedNewInstance();
        });

        it('given a matrix with dimensions (2,4) >> sig >> should return matrix with dimensions (2,4)', () => {
          actual = sut.sig(mat1);

          expectOperationHasReturnedMatrixWithDimensions(2, 4);
        });

        it('given a matrix with dimensions (2,4) >> sig >> should return matrix with dimensions (2,4)', () => {
          actual = sut.sig(mat1);

          expectMonadicOperationHasReturnedMatrixWithContent([0.731058, 0.982013, 0.997527, 0.999954, 0.880797, 0.999088, 0.993307, 0.952574]);
        });
      });
  
      describe('Rectified Linear Units (ReLU)', () => {

        beforeEach(() => {
          // Mat with some negative values
          mat1.setFrom([1, -4, 6, 10, 2, -7, 5, 3]);
        });

        it('given a matrix >> relu >> should return new instance of matrix-object (reference)', () => {
          actual = sut.relu(mat1);

          expectOperationHasReturnedNewInstance();
        });

        it('given a matrix with dimensions (2,4) >> relu >> should return matrix with dimensions (2,4)', () => {
          actual = sut.relu(mat1);

          expectOperationHasReturnedMatrixWithDimensions(2, 4);
        });

        it('given a matrix with dimensions (2,4) >> relu >> should return matrix with dimensions (2,4)', () => {
          actual = sut.relu(mat1);

          expectMonadicOperationHasReturnedMatrixWithContent([1, 0, 6, 10, 2, 0, 5, 3]);
        });
      });

      const expectMonadicOperationHasReturnedMatrixWithContent = (content: Array<number>) => {
        expected.setFrom(content);

        for (let i = 0; i < actual.w.length; i++) {
          expect(actual.w[i]).toBeCloseTo(expected.w[i], 5);
        }
      };
    });
  });

  describe('Dual Matrix Operations:', ()  => {

    let mat2: Mat;

    describe('Multiplication:', () => {

      beforeEach(() => {
        mat2 = new Mat(4, 3);
        mat2.setFrom([1, 4, 6, 2, 7, 5, 9, 0, 11, 3, 1, 0]);
      });

      it('given two matrices >> multiply >> should return new instance of matrix-object (reference)', () => {
        actual = sut.mul(mat1, mat2);
  
        expectDualMatrixOperationHasReturnedNewInstance();
      });
  
      it('given two matrices with dimensions (2,4)*(4,3) >> multiply >> should return matrix with dimensions (2,3)', () => {
        actual = sut.mul(mat1, mat2);
  
        expectOperationHasReturnedMatrixWithDimensions(2, 3);
      });
  
      it('given two matrices with incompatible dimensions (2,4)*(3,3) >> multiply >> should throw error', () => {
        const incompatible = new Mat(3,3);

        const callFunction = () => { sut.mul(mat1, incompatible); };

        expect(callFunction).toThrowError('[class:MatOps] mul: dimensions misaligned');
      });
  
      it('given two matrices >> multiply >> should return matrix with expected content', () => {
        actual = sut.mul(mat1, mat2);
  
        expected = new Mat(2, 3);
        expectMultiplicationHasReturnedMatrixWithContent([93, 42, 92, 70, 60, 102]);
      });
      
      const expectMultiplicationHasReturnedMatrixWithContent = (content: Array<number>) => {
        expected.setFrom(content);

        for (let i = 0; i < actual.w.length; i++) {
          expect(actual.w[i]).toBe(expected.w[i]);
        }
      };
    });

    describe('Addition:', () => {

      beforeEach(() => {
        mat2 = new Mat(2, 4);
        mat2.setFrom([1, 4, 6, 10, 2, 7, 5, 3]);
      });

      it('given two matrices >> add >> should return new instance of matrix-object (reference)', () => {
        actual = sut.add(mat1, mat2);
  
        expectDualMatrixOperationHasReturnedNewInstance();
      });
  
      it('given two matrices with dimensions (2,4)*(2,4) >> add >> should return matrix with dimensions (2,4)', () => {
        actual = sut.add(mat1, mat2);
  
        expectOperationHasReturnedMatrixWithDimensions(2, 4);
      });

      it('given two matrices with incompatible dimensions (2,4)*(3,3) >> add >> should throw error', () => {
        const incompatible = new Mat(3, 3);

        const callFunction = () => { sut.add(mat1, incompatible); };

        expect(callFunction).toThrowError('[class:MatOps] add: dimensions misaligned');
      });
  
      it('given two matrices >> add >> should return matrix with expected content', () => {
        actual = sut.add(mat1, mat2);
  
        expected = new Mat(2, 4);
        expectAdditionHasReturnedMatrixWithContent([2, 8, 12, 20, 4, 14, 10, 6]);
      });
      
      const expectAdditionHasReturnedMatrixWithContent = (content: Array<number>) => {
        expected.setFrom(content);

        for (let i = 0; i < actual.w.length; i++) {
          expect(actual.w[i]).toBe(expected.w[i]);
        }
      };
    });

    describe('Dot Product:', () => {

      beforeEach(() => {
        mat2 = new Mat(2, 4);
        mat2.setFrom([1, 4, 6, 10, 2, 7, 5, 3]);
      });

      it('given two matrices >> dot >> should return new instance of matrix-object (reference)', () => {
        actual = sut.dot(mat1, mat2);
  
        expectDualMatrixOperationHasReturnedNewInstance();
      });
  
      it('given two matrices with dimensions (2,4)*(2,4) >> dot >> should return matrix with dimensions (1,1)', () => {
        actual = sut.dot(mat1, mat2);
  
        expectOperationHasReturnedMatrixWithDimensions(1, 1);
      });

      it('given two matrices with incompatible dimensions (2,4)*(3,3) >> dot >> should throw error', () => {
        const incompatible = new Mat(3, 3);

        const callFunction = () => { sut.dot(mat1, incompatible); };

        expect(callFunction).toThrowError('[class:MatOps] dot: dimensions misaligned');
      });
  
      it('given two matrices >> dot >> should return matrix with expected content', () => {
        actual = sut.dot(mat1, mat2);
  
        expected = new Mat(1, 1);
        expectDotHasReturnedMatrixWithContent([1 + 16 + 36 + 100 + 4 + 49 + 25 + 9]);
      });
      
      const expectDotHasReturnedMatrixWithContent = (content: Array<number>) => {
        expected.setFrom(content);

        for (let i = 0; i < actual.w.length; i++) {
          expect(actual.w[i]).toBe(expected.w[i]);
        }
      };
    });

    describe('Elementwise Multiplication:', () => {
      
      beforeEach(() => {
        mat2 = new Mat(2, 4);
        mat2.setFrom([1, 4, 6, 10, 2, 7, 5, 3]);
      });

      it('given two matrices >> eltmul >> should return new instance of matrix-object (reference)', () => {
        actual = sut.eltmul(mat1, mat2);
  
        expectDualMatrixOperationHasReturnedNewInstance();
      });
  
      it('given two matrices with dimensions (2,4)*(2,4) >> eltmul >> should return matrix with dimensions (2,4)', () => {
        actual = sut.eltmul(mat1, mat2);
  
        expectOperationHasReturnedMatrixWithDimensions(2, 4);
      });

      it('given two matrices with incompatible dimensions (2,4)*(3,3) >> eltmul >> should throw error', () => {
        const incompatible = new Mat(3, 3);

        const callFunction = () => { sut.eltmul(mat1, incompatible); };

        expect(callFunction).toThrowError('[class:MatOps] eltmul: dimensions misaligned');
      });
  
      it('given two matrices >> eltmul >> should return matrix with expected content', () => {
        actual = sut.eltmul(mat1, mat2);
  
        expected = new Mat(2, 4);
        expectEltmulHasReturnedMatrixWithContent([1, 16, 36, 100, 4, 49, 25, 9]);
      });
      
      const expectEltmulHasReturnedMatrixWithContent = (content: Array<number>) => {
        expected.setFrom(content);

        for (let i = 0; i < actual.w.length; i++) {
          expect(actual.w[i]).toBe(expected.w[i]);
        }
      };
    });

    const expectDualMatrixOperationHasReturnedNewInstance = (): void => {
      expectOperationHasReturnedNewInstance();
      expect(actual === mat2).toBe(false);
    };
  });

  const expectOperationHasReturnedNewInstance = (): void => {
    expect(actual === mat1).toBe(false);
  };

  const expectOperationHasReturnedMatrixWithDimensions = (rows: number, cols: number): void => {
    expected = new Mat(rows, cols);

    expect(actual.rows).toBe(expected.rows);
    expect(actual.cols).toBe(expected.cols);
  };
});

