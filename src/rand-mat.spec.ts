import { RandMat, Utils, Mat } from '.';

describe('RandMat:', () => {
  let sut: Mat;

  describe('Check usage of Utils:', () => {
    
    beforeEach(() => {
      spyOn(Utils, 'fillRandn');
      sut = new RandMat(2, 4, 0, 1);
    });

    it('given a fresh instance >> on creation >> should have created Mat of expected size', () => {
      expect(sut.rows).toBe(2);
      expect(sut.cols).toBe(4);
    });

    it('given a fresh instance >> on creation >> should have called Utils.fillRandn', () => {
      expect(Utils.fillRandn).toHaveBeenCalled();
      expect(Utils.fillRandn).toHaveBeenCalledWith(sut.w, 0, 1);
    });
  });
  
  describe('Check population of values:', () => {
  
    beforeEach(() => {
      spyOn(Utils, 'randn').and.callFake((mu: number, std: number) => { return mu + std; });
    });
  
    it('given a fresh instance >> on creation >> should have populated Matrix with values', () => {
      sut = new RandMat(1, 3, 1, 0.123);
  
      expectMatrixToBePopulatedWith([1.123, 1.123, 1.123]);
    });
  
    const expectMatrixToBePopulatedWith = (expected: Array<number>) => {
      for (let i = 0; i < sut.w.length; i++) {
        expect(sut.w[i]).toBeCloseTo(expected[i], 5);
      }
    };
  });
});

