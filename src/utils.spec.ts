import { Utils } from '.';

const gaussRandomBackup = Utils['gaussRandom'];

describe('Utils:', () => {

  const sut = Utils;

  beforeAll(() => {
    patchPrivateFunctionUtilsGaussRandom();
  });
  
  describe('Random Number Functions:', () => {
    beforeEach(() => {
      spyOn(Math, 'random').and.callFake(() => { return 1; });
    });

    it('no instance >> randn >> should give back ', () => {
      const actual = sut.randn(1, 1.123);
  
      expect(actual).toBe(2.123);
    });

    it('no instance >> randi >> should give back ', () => {
      const actual = sut.randi(1, 1.123);
  
      expect(actual).toBe(1);
    });

    it('no instance >> randf >> should give back ', () => {
      const actual = sut.randf(1, 1.123);
  
      expect(actual).toBe(1.123);
    });
  });
  
  describe('Array Filler:', () => {
    let actual: Array<number>;

    beforeEach(() => {
      spyOn(sut, 'randn').and.callFake(() => { return 1; });
      spyOn(sut, 'randf').and.callFake(() => { return 1; });
      actual = new Array<number>(5);
    });

    it('Unpopulated Array of size 5 >> fillRandn >> should call Utils.randn 5 times', () => {
      sut.fillRandn(actual, 0, 5);
  
      expect(sut.randn).toHaveBeenCalled();
      expect(sut.randn).toHaveBeenCalledTimes(5);
      expect(sut.randn).toHaveBeenCalledWith(0, 5);
    });

    it('Unpopulated Array of size 5 >> fillRandn >> should be filled with values from Utils.randn', () => {
      sut.fillRandn(actual, 0, 5);

      expectSutToBeFilledWith(1);
    });

    it('Unpopulated Array of size 5 >> fillRand >> should call Utils.randf 5 times', () => {
      sut.fillRand(actual, 0, 5);
  
      expect(sut.randf).toHaveBeenCalled();
      expect(sut.randf).toHaveBeenCalledTimes(5);
      expect(sut.randf).toHaveBeenCalledWith(0, 5);
    });

    it('Unpopulated Array of size 5 >> fillRand >> should be filled with values from Utils.randf', () => {
      sut.fillRand(actual, 0, 5);

      expectSutToBeFilledWith(1);
    });

    it('Unpopulated Array of size 5 >> fillConst >> should be filled with constant value', () => {
      sut.fillConst(actual, 1);

      expectSutToBeFilledWith(1);
    });

    const expectSutToBeFilledWith = (expected: number) => {
      for (let i = 0; i < sut.length; i++) {
        expect(actual[i]).toBe(expected);
      }
    };
  });

  describe('Array Creation Functions:', () => {
    let actual: Array<number> | Float64Array;

    it('no instance >> zeros >> should be of expected length', () => {
      actual = sut.zeros(4);

      expect(actual.length).toBe(4);
    });

    it('no instance >> zeros >> should be filled with constant zeros', () => {
      actual = sut.zeros(4);

      expectSutToBeFilledWith(0);
    });

    it('no instance >> ones >> should be of expected length', () => {
      actual = sut.ones(4);

      expect(actual.length).toBe(4);
    });

    it('no instance >> ones >> should be filled with constant ones', () => {
      actual = sut.ones(4);

      expectSutToBeFilledWith(1);
    });

    const expectSutToBeFilledWith = (expected: number) => {
      for (let i = 0; i < sut.length; i++) {
        expect(actual[i]).toBe(expected);
      }
    };
  });

  describe('Argmax:', () => {
    it('Populated Array >> argmax >> should return index of field with max value', () => {
      const actual = sut.argmax([0, 1, 10, 3, 4]);

      expect(actual).toBe(2);
    });
  });

  describe('Weighted Sampling:', () => {
    beforeEach(() => {
      spyOn(Math, 'random').and.callFake(() => { return 1.2; });
    });

    it('Populated Array >> sampleWeighted >> should return index of field with accumulated value greater than pseudo-random number', () => {
      const actual = sut.sampleWeighted([0, 1, 10, 3, 4]);

      expect(actual).toBe(2);
    });

    it('Populated Array >> sampleWeighted >> should return of Array of zeros (fallback)', () => {
      const actual = Utils.sampleWeighted([0, 0, 0, 0]);

      expect(actual).toBe(0);
    });
  });

  afterAll(() => {
    reversePatchUtilsGaussRandom();
  });
  
  const patchPrivateFunctionUtilsGaussRandom = (): void => {
    Utils['gaussRandom'] = () => { return 1; };
  };

  const reversePatchUtilsGaussRandom = (): void => {
    Utils['gaussRandom'] = gaussRandomBackup;
  };
});

