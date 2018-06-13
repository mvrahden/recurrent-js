import { Utils } from '.';

describe('Utils:', () => {

  const sut = Utils;

  describe('Public Methods:', () => {

    describe('Random Number Functions:', () => {

      describe('Gaussian Random Number Generator (gaussRandom):', () => {

        let actual = { min: null, max: null, std: null, var: null, mean: null };

        it('Unpatched random number generator >> 100000 iterations of gaussRandom (private function) >> should comply to given statistical restrictions', () => {
          let actualSamples = [];

          for (let i = 0; i < 100000; i++) {
            actualSamples.push(sut['gaussRandom']());
          }

          actual = determineBasicStatistics(actualSamples);

          expect(actual.min).toBeGreaterThan(-5, 'Min');
          expect(actual.min).toBeLessThan(-3.5, 'Min');
          expect(actual.max).toBeGreaterThan(3.5, 'Max');
          expect(actual.max).toBeLessThan(5, 'Max');
          expect(actual.mean).toBeGreaterThan(-0.1, 'Mean');
          expect(actual.mean).toBeLessThan(0.1, 'Mean');
          expect(actual.std).toBeGreaterThan(0.9, 'Std');
          expect(actual.std).toBeLessThan(1.1, 'Std');
          expect(actual.var).toBeGreaterThan(0.9, 'Var');
          expect(actual.var).toBeLessThan(1.1, 'Var');
        });

        it('Unpatched random number generator >> box_muller (private function) >> should comply to given statistical restrictions', () => {
          let actualSamples = [];

          for (let i = 0; i < 100000; i++) {
            actualSamples.push(sut['box_muller']());
          }

          actual = determineBasicStatistics(actualSamples);

          expect(actual.min).toBeGreaterThan(0, 'Min');
          expect(actual.min).toBeLessThan(0.12, 'Min');
          expect(actual.max).toBeGreaterThan(0.87, 'Max');
          expect(actual.max).toBeLessThan(1, 'Max');
          expect(actual.mean).toBeGreaterThan(0.49, 'Mean');
          expect(actual.mean).toBeLessThan(0.51, 'Mean');
          expect(actual.std).toBeGreaterThan(0.09, 'Std');
          expect(actual.std).toBeLessThan(0.11, 'Std');
          expect(actual.var).toBeGreaterThan(0.009, 'Var');
          expect(actual.var).toBeLessThan(0.011, 'Var');
        });
      });

      describe('With Unpatched Random Number Generator:', () => {

        let actual = { min: null, max: null, std: null, var: null, mean: null };

        it('no arrangement >> randn >> should comply to given statistical restrictions', () => {
          let actualSamples = [];

          for (let i = 0; i < 100000; i++) {
            actualSamples.push(sut.randn(0, 1));
          }
          actual = determineBasicStatistics(actualSamples);

          expect(actual.min).toBeGreaterThan(-5, 'Min');
          expect(actual.min).toBeLessThan(-3.5, 'Min');
          expect(actual.max).toBeGreaterThan(3.5, 'Max');
          expect(actual.max).toBeLessThan(5, 'Max');
          expect(actual.mean).toBeGreaterThan(-0.1, 'Mean');
          expect(actual.mean).toBeLessThan(0.1, 'Mean');
          expect(actual.std).toBeGreaterThan(0.9, 'Std');
          expect(actual.std).toBeLessThan(1.1, 'Std');
          expect(actual.var).toBeGreaterThan(0.9, 'Var');
          expect(actual.var).toBeLessThan(1.1, 'Var');
        });

        it('no arrangement >> randf >> should comply to given statistical restrictions', () => {
          let actualSamples = [];

          for (let i = 0; i < 100000; i++) {
            actualSamples.push(sut.randf(0, 100));
          }
          actual = determineBasicStatistics(actualSamples);

          expect(actual.min).toBeGreaterThan(0, 'Min');
          expect(actual.min).toBeLessThan(0.01, 'Min');
          expect(actual.max).toBeGreaterThan(99.9, 'Max');
          expect(actual.max).toBeLessThan(100, 'Max');
          expect(actual.mean).toBeGreaterThan(49.7, 'Mean');
          expect(actual.mean).toBeLessThan(50.3, 'Mean');
        });

        it('no arrangement >> randi >> should comply to given statistical restrictions', () => {
          let actualSamples = [];

          for (let i = 0; i < 100000; i++) {
            actualSamples.push(sut.randi(0, 100));
          }
          actual = determineBasicStatistics(actualSamples);

          expect(actual.min).toBeGreaterThanOrEqual(0, 'Min');
          expect(actual.min).toBeLessThanOrEqual(1, 'Min');
          expect(actual.max).toBeGreaterThanOrEqual(98, 'Max');
          expect(actual.max).toBeLessThanOrEqual(99, 'Max');
          expect(actual.mean).toBeGreaterThan(49, 'Mean');
          expect(actual.mean).toBeLessThan(51, 'Mean');
        });

        it('no arrangement >> skewedRandn (skewness factor 1) >> should comply to given statistical restrictions', () => {
          let actualSamples = [];

          for (let i = 0; i < 100000; i++) {
            actualSamples.push(sut.skewedRandn(0, 10, 5));
          }
          actual = determineBasicStatistics(actualSamples);

          expect(actual.min).toBeGreaterThan(0, 'Min');
          expect(actual.min).toBeLessThan(0.1, 'Min');
          expect(actual.max).toBeGreaterThan(5, 'Max');
          expect(actual.max).toBeLessThan(10, 'Max');
          expect(actual.std).toBeGreaterThan(0.42, 'Std');
          expect(actual.std).toBeLessThan(0.47, 'Std');
          expect(actual.var).toBeGreaterThan(0.11, 'Var');
          expect(actual.var).toBeLessThan(0.25, 'Var');
          expect(actual.mean).toBeGreaterThan(0.4, 'Mean');
          expect(actual.mean).toBeLessThan(0.5, 'Mean');
        });
      });

      describe('With Patched Random Number Generator:', () => {

        beforeEach(() => {
          spyOn(Utils, 'gaussRandom' as any).and.callFake(() => { return 1; });
          spyOn(Math, 'random').and.callFake(() => { return 1; });
        });

        it('patched random number generator >> randn >> should give back 2.123', () => {
          const actual = sut.randn(1, 1.123);

          expect(actual).toBe(2.123);
        });

        it('patched random number generator >> randi >> should give back ', () => {
          const actual = sut.randi(1, 1.123);

          expect(actual).toBe(1);
        });

        it('patched random number generator >> randf >> should give back ', () => {
          const actual = sut.randf(1, 1.123);

          expect(actual).toBe(1.123);
        });
      });

      const determineBasicStatistics = (sample: Array<number>): { min, max, std, var, mean } => {
        let out = { min: null, max: null, std: null, var: null, mean: null };
        // RANGE: MIN & MAX
        out.min = Number.POSITIVE_INFINITY, out.max = Number.NEGATIVE_INFINITY;
        for (let i = 0; i < sample.length; i++) {
          if (sample[i] > out.max)
            out.max = sample[i];
          if (sample[i] < out.min)
            out.min = sample[i];
        }
        // MEAN, MEDIAN & MODE
        out.std = Utils.std(sample);
        out.var = Utils.var(sample);
        out.mean = Utils.mean(sample);
        return out;
      }
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

      it('Unpopulated Array of size 5 >> fillConst >> should be filled with constant value 2', () => {
        sut.fillConst(actual, 2);

        expectSutToBeFilledWith(2);
      });

      const expectSutToBeFilledWith = (expected: number) => {
        for (let i = 0; i < sut.length; i++) {
          expect(actual[i]).toBe(expected);
        }
      };
    });

    describe('Statistic Tools:', () => {

      it('some Array [3, 5, 4, 4, 1, 1, 2, 3] >> sum >> should return 23', () => {
        const input = [3, 5, 4, 4, 1, 1, 2, 3];

        let actual = sut.sum(input);

        expect(actual).toBe(23)
      });

      it('some Array [3, 5, 4, 4, 1, 1, 2, 3] >> mean >> should return 2.875', () => {
        const input = [3, 5, 4, 4, 1, 1, 2, 3];

        let actual = sut.mean(input);

        expect(actual).toBe(2.875)
      });

      it('some Array [3, 5, 4, 4, 1, 1, 2, 3] >> median >> should return 3', () => {
        const evenInput = [3, 5, 4, 4, 1, 1, 2, 3];

        let actual = sut.median(evenInput);

        expect(actual).toBe(3);
      });

      it('some Array [3, 5, 4, 4, 1, 1, 2, 3, 9] >> median >> should return 3', () => {
        const oddInput = [3, 5, 4, 4, 1, 1, 2, 3, 9];

        let actual = sut.median(oddInput);

        expect(actual).toBe(3);
      });

      it('some Array [3, 5, 4, 4, 1, 1, 2, 3] >> var >> should return 2.125 an thus default to `unbiased` form', () => {
        const input = [3, 5, 4, 4, 1, 1, 2, 3];

        let actual = sut.var(input);

        expect(actual).toBe(2.125);
      });

      it('some Array [3, 5, 4, 4, 1, 1, 2, 3] >> var >> should return 1.859375', () => {
        const input = [3, 5, 4, 4, 1, 1, 2, 3];

        let actual = sut.var(input, 'uncorrected');

        expect(actual).toBe(1.859375);
      });

      it('some Array [3, 5, 4, 4, 1, 1, 2, 3] >> std >> should return 1.45722 and thus default to `unbiased` form', () => {
        const input = [3, 5, 4, 4, 1, 1, 2, 3];

        let actual = sut.std(input);

        expect(actual).toBeCloseTo(1.45722);
      });

      it('some Array [3, 5, 4, 4, 1, 1, 2, 3] >> mode >> should return [1, 3, 4]', () => {
        const input = [3, 5, 4, 4, 1, 1, 2, 3];

        let actual = sut.mode(input);

        expect(actual).toEqual([1, 3, 4]);
      });

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

      it('Populated Array >> patched sampleWeighted >> should return index of field with accumulated value greater than the given output of patch', () => {
        const actual = sut.sampleWeighted([0, 1, 10, 3, 4]);

        // 0 + 1 + 10 > 1.2 >> 10 is in field 2
        expect(actual).toBe(2);
      });

      it('Populated Array of zeros >> patched sampleWeighted >> should return 0 (fallback)', () => {
        const actual = Utils.sampleWeighted([0, 0, 0, 0]);

        expect(actual).toBe(0);
      });
    });
  });
});
