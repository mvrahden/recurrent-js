import { Utils } from '..';

const patchFillRandn = () => {
  spyOn(Utils, 'fillRandn').and.callFake(fillConstOnes);
};

const fillConstOnes = (arr) => {
  Utils.fillConst(arr, 1);
};
