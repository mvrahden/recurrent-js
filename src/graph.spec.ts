// import { Graph, Mat } from '../../dist/index';

// describe('Graph:', () => {
//   it('should multiply matrices', () => {
//     // initialize
//     const mat1 = new Mat(2, 4);
//     mat1.setFrom([1, 4, 6, 10, 2, 7, 5, 3]);
//     const mat2 = new Mat(4, 3);
//     mat2.setFrom([1, 4, 6, 2, 7, 5, 9, 0, 11, 3, 1, 0]);

//     // create actual output
//     const graph = new Graph(false);
//     const actual = graph.mul(mat1, mat2);

//     // create expected output
//     const expected = new Mat(2, 3);
//     expected.setFrom([93, 42, 92, 70, 60, 102]);

//     // test
//     actual.w.forEach((value, i) => {
//       expect(value).toBe(expected.w[i]);
//     });
//     expect(actual.rows).toBe(expected.rows);
//     expect(actual.cols).toBe(expected.cols);
//   });
// });
