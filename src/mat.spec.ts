import { Mat, Utils } from '.';

describe('Mat:', () => {
  let sut = Mat;
  let actual: Mat;
  let expected: Mat;

  describe('Single Matrix Operations', ()  => {
    let mat1: Mat;

    describe('Row Pluck:', () => {
      let rowIndex: number;

      beforeEach(() => {
        mat1 = new Mat(2, 4);
        mat1.setFrom([1, 4, 6, 10, 2, 7, 5, 3]);
        rowIndex = 0;
      });

      it('given a matrix >> rowPluck >> should return new instance of matrix-object (reference)', () => {
        actual = sut.rowPluck(mat1, rowIndex);

        expectRowpluckHadReturnedNewInstance();
      });

      it('given a matrix with dimensions (2,4) >> rowPluck >> should return matrix with dimensions (4,1)', () => {
        actual = sut.rowPluck(mat1, rowIndex);

        expectRowpluckHasReturnedMatrixWithDimensions(4, 1);
      });

      it('given a matrix with dimensions (2,4) and incompatible rowIndex >> rowPluck >> should throw error', () => {
        const incompatible = 2;

        const callFunction = () => { sut.rowPluck(mat1, incompatible); };

        expect(callFunction).toThrowError('[class:Mat] rowPluck: dimensions misaligned');
      });

      it('given a matrix >> rowPluck >> should return matrix with expected content', () => {
        actual = sut.rowPluck(mat1, rowIndex);

        expected = new Mat(4, 1);
        expectRowpluckHasReturnedMatrixWithContent([1, 4, 6, 10]);
      });

      const expectRowpluckHadReturnedNewInstance = (): void => {
        expect(actual === mat1).toBe(false);
      }

      const expectRowpluckHasReturnedMatrixWithDimensions = (rows: number, cols: number): void => {
        expected = new Mat(rows, cols);

        expect(actual.rows).toBe(expected.rows);
        expect(actual.cols).toBe(expected.cols);
      }

      const expectRowpluckHasReturnedMatrixWithContent = (content: Array<number>) => {
        expected.setFrom(content);

        for (let i = 0; i < actual.w.length; i++) {
          expect(actual.w[i]).toBe(expected.w[i]);
        }
      }
    });

    describe('Gauss noise-addition:', () => {
      let std: Mat;
      beforeEach(() => {
        mat1 = new Mat(2, 4);
        mat1.setFrom([1, 4, 6, 10, 2, 7, 5, 3]);
        std = new Mat(2, 4);
        std.setFrom([0.1, 0.2, 0.02, 0.5, 1, 0.01, 0, 1]);
      });

      it('given a matrix >> gauss >> should return new instance of matrix-object (reference)', () => {
        actual = sut.gauss(mat1, std);

        expectGaussHadReturnedNewInstance();
      });

      it('given a matrix with dimensions (2,4) >> gauss >> should return matrix with dimensions (2,4)', () => {
        actual = sut.gauss(mat1, std);

        expectGaussHasReturnedMatrixWithDimensions(2, 4);
      });

      it('given a matrix with dimensions (2,4) and (3,3) >> gauss >> should throw error', () => {
        const incompatible = new Mat(3, 3);

        const callFunction = () => { sut.gauss(mat1, incompatible); };

        expect(callFunction).toThrowError('[class:Mat] gauss: dimensions misaligned');
      });

      it('given a matrix >> gauss >> should return matrix with expected content', () => {
        patchUtilsRandn();

        actual = sut.gauss(mat1, std);

        expected = new Mat(2, 4);
        expectGaussHasReturnedMatrixWithGaussianDistributedContent([1, 4, 6, 10, 2, 7, 5, 3], [0.1, 0.2, 0.02, 0.5, 1, 0.01, 0, 1]);
      });

      const expectGaussHadReturnedNewInstance = (): void => {
        expect(actual === mat1).toBe(false);
      }

      const expectGaussHasReturnedMatrixWithDimensions = (rows: number, cols: number): void => {
        expected = new Mat(rows, cols);

        expect(actual.rows).toBe(expected.rows);
        expect(actual.cols).toBe(expected.cols);
      }

      const expectGaussHasReturnedMatrixWithGaussianDistributedContent = (content: Array<number>, std: Array<number>) => {
        expected.setFrom(content);

        for (let i = 0; i < actual.w.length; i++) {
          expect(actual.w[i]).toBe(expected.w[i] + std[i]);
        }
      }

      const patchUtilsRandn = (): void => {
        Utils.randn = (mu: number, std: number) => { return mu + std; };
      }
    });

    describe('Hyperbolic Tangens', () => {
      
    });

    describe('Sigmoid', () => {
      
    });

    describe('Rectified Linear Units (ReLU)', () => {
      
    });
  });

  describe('Dual Matrix Operations:', ()  => {
    let mat1: Mat;
    let mat2: Mat;

    describe('Multiplication:', () => {
      beforeEach(() => {
        prepareMatricesForMultiplication();
      });

      it('given two matrices >> multiply >> should return new instance of matrix-object (reference)', () => {
        actual = sut.mul(mat1, mat2);
  
        expectMultiplicationHadReturnedNewInstance();
      });
  
      it('given two matrices with dimensions (2,4)*(4,3) >> multiply >> should return matrix with dimensions (2,3)', () => {
        actual = sut.mul(mat1, mat2);
  
        expectMultiplicationHasReturnedMatrixWithDimensions(2, 3);
      });
  
      it('given two matrices with incompatible dimensions (2,4)*(3,3) >> multiply >> should throw error', () => {
        const incompatible = new Mat(3,3);

        const callFunction = () => { sut.mul(mat1, incompatible); };

        expect(callFunction).toThrowError('[class:Mat] mul: dimensions misaligned');
      });
  
      it('given two matrices >> multiply >> should return matrix with expected content', () => {
        actual = sut.mul(mat1, mat2);
  
        expected = new Mat(2, 3);
        expectMultiplicationHasReturnedMatrixWithContent([93, 42, 92, 70, 60, 102]);
      });

      const prepareMatricesForMultiplication = (): void => {
        mat1 = new Mat(2, 4);
        mat1.setFrom([1, 4, 6, 10, 2, 7, 5, 3]);
        mat2 = new Mat(4, 3);
        mat2.setFrom([1, 4, 6, 2, 7, 5, 9, 0, 11, 3, 1, 0]);
      }
  
      const expectMultiplicationHadReturnedNewInstance = (): void => {
        // test dimension
        expect(actual === mat1).toBe(false);
        expect(actual === mat2).toBe(false);
      }
  
      const expectMultiplicationHasReturnedMatrixWithDimensions = (rows: number, cols: number): void => {
        expected = new Mat(rows, cols);
  
        // test dimension
        expect(actual.rows).toBe(expected.rows);
        expect(actual.cols).toBe(expected.cols);
      }
      
      const expectMultiplicationHasReturnedMatrixWithContent = (content: Array<number>) => {
        expected.setFrom(content);
        // test content
        for (let i = 0; i < actual.w.length; i++) {
          expect(actual.w[i]).toBe(expected.w[i]);
        }
      }
    });

    describe('Addition:', () => {
      beforeEach(() => {
        mat1 = new Mat(2, 4);
        mat1.setFrom([1, 4, 6, 10, 2, 7, 5, 3]);
        mat2 = new Mat(2, 4);
        mat2.setFrom([1, 4, 6, 10, 2, 7, 5, 3]);
      });

      it('given two matrices >> add >> should return new instance of matrix-object (reference)', () => {
        actual = sut.add(mat1, mat2);
  
        expectAdditionHadReturnedNewInstance();
      });
  
      it('given two matrices with dimensions (2,4)*(2,4) >> add >> should return matrix with dimensions (2,4)', () => {
        actual = sut.add(mat1, mat2);
  
        expectAdditionHasReturnedMatrixWithDimensions(2, 4);
      });

      it('given two matrices with incompatible dimensions (2,4)*(3,3) >> add >> should throw error', () => {
        const incompatible = new Mat(3, 3);

        const callFunction = () => { sut.add(mat1, incompatible); };

        expect(callFunction).toThrowError('[class:Mat] add: dimensions misaligned');
      });
  
      it('given two matrices >> add >> should return matrix with expected content', () => {
        actual = sut.add(mat1, mat2);
  
        expected = new Mat(2, 4);
        expectAdditionHasReturnedMatrixWithContent([2, 8, 12, 20, 4, 14, 10, 6]);
      });
  
      const expectAdditionHadReturnedNewInstance = (): void => {
        expect(actual === mat1).toBe(false);
        expect(actual === mat2).toBe(false);
      }
  
      const expectAdditionHasReturnedMatrixWithDimensions = (rows: number, cols: number): void => {
        expected = new Mat(rows, cols);
  
        expect(actual.rows).toBe(expected.rows);
        expect(actual.cols).toBe(expected.cols);
      }
      
      const expectAdditionHasReturnedMatrixWithContent = (content: Array<number>) => {
        expected.setFrom(content);

        for (let i = 0; i < actual.w.length; i++) {
          expect(actual.w[i]).toBe(expected.w[i]);
        }
      }
    });

    describe('Dot Product:', () => {
      beforeEach(() => {
        mat1 = new Mat(2, 4);
        mat1.setFrom([1, 4, 6, 10, 2, 7, 5, 3]);
        mat2 = new Mat(2, 4);
        mat2.setFrom([1, 4, 6, 10, 2, 7, 5, 3]);
      });

      it('given two matrices >> dot >> should return new instance of matrix-object (reference)', () => {
        actual = sut.dot(mat1, mat2);
  
        expectDotHadReturnedNewInstance();
      });
  
      it('given two matrices with dimensions (2,4)*(2,4) >> dot >> should return matrix with dimensions (1,1)', () => {
        actual = sut.dot(mat1, mat2);
  
        expectDotHasReturnedMatrixWithDimensions(1, 1);
      });

      it('given two matrices with incompatible dimensions (2,4)*(3,3) >> dot >> should throw error', () => {
        const incompatible = new Mat(3, 3);

        const callFunction = () => { sut.dot(mat1, incompatible); };

        expect(callFunction).toThrowError('[class:Mat] dot: dimensions misaligned');
      });
  
      it('given two matrices >> dot >> should return matrix with expected content', () => {
        actual = sut.dot(mat1, mat2);
  
        expected = new Mat(1, 1);
        expectDotHasReturnedMatrixWithContent([1 + 16 + 36 + 100 + 4 + 49 + 25 + 9]);
      });
  
      const expectDotHadReturnedNewInstance = (): void => {
        expect(actual === mat1).toBe(false);
        expect(actual === mat2).toBe(false);
      }
  
      const expectDotHasReturnedMatrixWithDimensions = (rows: number, cols: number): void => {
        expected = new Mat(rows, cols);
  
        expect(actual.rows).toBe(expected.rows);
        expect(actual.cols).toBe(expected.cols);
      }
      
      const expectDotHasReturnedMatrixWithContent = (content: Array<number>) => {
        expected.setFrom(content);

        for (let i = 0; i < actual.w.length; i++) {
          expect(actual.w[i]).toBe(expected.w[i]);
        }
      }
    });

    describe('Elementwise Multiplication:', () => {
      beforeEach(() => {
        mat1 = new Mat(2, 4);
        mat1.setFrom([1, 4, 6, 10, 2, 7, 5, 3]);
        mat2 = new Mat(2, 4);
        mat2.setFrom([1, 4, 6, 10, 2, 7, 5, 3]);
      });

      it('given two matrices >> eltmul >> should return new instance of matrix-object (reference)', () => {
        actual = sut.eltmul(mat1, mat2);
  
        expectEltmulHadReturnedNewInstance();
      });
  
      it('given two matrices with dimensions (2,4)*(2,4) >> eltmul >> should return matrix with dimensions (2,4)', () => {
        actual = sut.eltmul(mat1, mat2);
  
        expectEltmulHasReturnedMatrixWithDimensions(2, 4);
      });

      it('given two matrices with incompatible dimensions (2,4)*(3,3) >> eltmul >> should throw error', () => {
        const incompatible = new Mat(3, 3);

        const callFunction = () => { sut.eltmul(mat1, incompatible); };

        expect(callFunction).toThrowError('[class:Mat] eltmul: dimensions misaligned');
      });
  
      it('given two matrices >> eltmul >> should return matrix with expected content', () => {
        actual = sut.eltmul(mat1, mat2);
  
        expected = new Mat(2, 4);
        expectEltmulHasReturnedMatrixWithContent([1, 16, 36, 100, 4, 49, 25, 9]);
      });
  
      const expectEltmulHadReturnedNewInstance = (): void => {
        expect(actual === mat1).toBe(false);
        expect(actual === mat2).toBe(false);
      }
  
      const expectEltmulHasReturnedMatrixWithDimensions = (rows: number, cols: number): void => {
        expected = new Mat(rows, cols);
  
        expect(actual.rows).toBe(expected.rows);
        expect(actual.cols).toBe(expected.cols);
      }
      
      const expectEltmulHasReturnedMatrixWithContent = (content: Array<number>) => {
        expected.setFrom(content);

        for (let i = 0; i < actual.w.length; i++) {
          expect(actual.w[i]).toBe(expected.w[i]);
        }
      }
    });
  });
});

