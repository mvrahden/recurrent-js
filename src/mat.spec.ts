import { Mat } from ".";


describe('Matrix Object:', () => {
  let sut: Mat;
  describe('Instantiation:', () => {

    beforeEach(() => {
      sut = new Mat(2, 3);
    });

    it('fresh instance >> on creation >> should have expected rows, cols and array length', () => {
      expect(sut.rows).toBe(2);
      expect(sut.cols).toBe(3);
      expect(sut['_length']).toBe(6);
    });
    
    it('fresh instance >> on creation >> should have array of values [w] and derivatives [dw] with given length', () => {
      expect(sut.w.length).toBe(6);
      expect(sut.dw.length).toBe(6);
    });
    
    it('fresh instance >> on creation >> should have array of values [w] populated with zeros', () => {
      expectValuesToBe(0);
    });
    
    it('fresh instance >> on creation >> should have array of derivatives [dw] populated with zeros', () => {
      expectDerivativesToBe(0);
    });

    const expectValuesToBe = (value: number) => {
      for(let i = 0; i < sut.w.length; i++) {
        expect(sut.w[i]).toBe(value);
      }
    };

    const expectDerivativesToBe = (value: number) => {
      for(let i = 0; i < sut.w.length; i++) {
        expect(sut.w[i]).toBe(value);
      }
    };

  });

  describe('Get and Set:', () => {

    beforeEach(() => {
      sut = new Mat(2, 3);
      sut.setFrom([0, 3, 2, 4, 5, 1]);
    });

    it('fresh instance >> setFrom >> values should be populated in same order as given Array', () => {
      expectValuesToBe([0, 3, 2, 4, 5, 1]);
    });

    it('fresh instance >> setFrom >> derivatives should stay untouched', () => {
      expectDerivativesToBe([0, 0, 0, 0, 0, 0]);
    });

    it('with [0, 3, 2, 4, 5, 1] populated instance >> get value with given row and col >> should return value at position', () => {
      expectValueAtGivenPosition({row: 1, col: 2}, 1);
      expectValueAtGivenPosition({row: 0, col: 1}, 3);
    });

    it('with [0, 3, 2, 4, 5, 1] populated instance >> set value at given row and col >> should mutate value at given position', () => {
      sut.set(1, 2, 4);
      expect(sut.w[5]).toBe(4);
      expect(sut.dw[5]).toBe(0);

      sut.set(0, 1, 4);
      expect(sut.w[1]).toBe(4);
      expect(sut.dw[1]).toBe(0);
    });

    it('with [0, 3, 2, 4, 5, 1] populated instance >> set value at given row and col >> should not mutate derivative at position', () => {
      sut.set(1, 2, 4);
      expect(sut.dw[5]).toBe(0);

      sut.set(0, 1, 4);
      expect(sut.dw[1]).toBe(0);
    });

    it('with [0, 3, 2, 4, 5, 1] populated instance >> setColumn with given values >> should mutate the values of given column', () => {
      const m = new Mat(2, 1);
      m.setFrom([10, 20]);

      sut.setColumn(m, 1);

      expectColumnToContain(1, [10, 20]);
    });

    const expectColumnToContain = (col: number, expected: Array<number>) => {
      for(let i = 0; i < expected.length; i++) {
        expect(sut.get(i, col)).toBe(expected[i]);
      }
    };
    
    const expectValueAtGivenPosition = (given: any, expected: number) => {
      let actual = sut.get(given.row, given.col);
      expect(actual).toBe(expected);
    };

    const expectValuesToBe = (expected: Array<number>) => {
      for(let i = 0; i < expected.length; i++) {
        expect(sut.w[i]).toBe(expected[i]);
      }
    };

    const expectDerivativesToBe = (expected: Array<number>) => {
      for(let i = 0; i < expected.length; i++) {
        expect(sut.dw[i]).toBe(expected[i]);
      }
    };
  });
});
