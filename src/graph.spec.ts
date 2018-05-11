import { Graph, Mat } from '.';

/**
 * TEST MAT method call
 * TEST GRAPH derivative functions --> Outsource to Mat??
 */

describe('Graph:', () => {
  describe('Matrix Operation Call:', () => {
    let sut: Graph;
    let mat1: Mat;
    
    beforeEach(() => {
      // Turn Matrix into Test Double through Method-Patching
      initializeSpyFunctions();

      sut = new Graph(true);
      mat1 = new Mat(2, 4);
    });

    describe('Single Matrix Operations:', () => {
      it('given a graph and a matrices >> rowPluck >> should have called Mat.rowPluck', () => {
        const anyIndex = 0;
        sut.rowPluck(mat1, anyIndex);
  
        expect(Mat.rowPluck).toHaveBeenCalled();
        expect(Mat.rowPluck).toHaveBeenCalledWith(mat1, 0);
      });
      
      describe('Monadic Matrix Operations', () => {
        it('given a graph and 2 matrices >> tanh >> should have called Mat.tanh', () => {
          sut.tanh(mat1);
          
          expectSpyMethodToHaveBeenCalled(Mat.tanh);
        });
        
        it('given a graph and 2 matrices >> sig >> should have called Mat.sig', () => {
          sut.sig(mat1);
          
          expectSpyMethodToHaveBeenCalled(Mat.sig);
        });
    
        it('given a graph and 2 matrices >> relu >> should have called Mat.relu', () => {
          sut.relu(mat1);
          
          expectSpyMethodToHaveBeenCalled(Mat.relu);
        });
    
        const expectSpyMethodToHaveBeenCalled = (spy: Function): void => {
          expect(spy).toHaveBeenCalled();
          expect(spy).toHaveBeenCalledWith(mat1);
        }
      });
    });

    describe('Dual Matrix Operations:', () => {
      let mat2: Mat;

      beforeEach(() => {
        mat2 = new Mat(2, 4);
      });

      it('given a graph and 2 matrices >> mul >> should have called Mat.mul', () => {
        sut.mul(mat1, mat2);
  
        expectSpyMethodToHaveBeenCalled(Mat.mul);
      });
      
      it('given a graph and 2 matrices >> add >> should have called Mat.add', () => {
        sut.add(mat1, mat2);
  
        expectSpyMethodToHaveBeenCalled(Mat.add);
      });
      
      it('given a graph and 2 matrices >> dot >> should have called Mat.dot', () => {
        sut.dot(mat1, mat2);
        
        expectSpyMethodToHaveBeenCalled(Mat.dot);
      });
      
      it('given a graph and 2 matrices >> eltmul >> should have called Mat.eltmul', () => {
        sut.eltmul(mat1, mat2);
        
        expectSpyMethodToHaveBeenCalled(Mat.eltmul);
      });
  
      const expectSpyMethodToHaveBeenCalled = (spy: Function): void => {
        expect(spy).toHaveBeenCalled();
        expect(spy).toHaveBeenCalledWith(mat1, mat2);
      }
    });
  
    const initializeSpyFunctions = (): void => {
      spyOn(Mat, 'rowPluck');
      spyOn(Mat, 'tanh');
      spyOn(Mat, 'sig');
      spyOn(Mat, 'relu');
      spyOn(Mat, 'mul');
      spyOn(Mat, 'add');
      spyOn(Mat, 'dot');
      spyOn(Mat, 'eltmul');
    };
  });

  describe('Graph Stack', () => {
    xit('given 2 matrices >> multiply >> should add derivative function on stack', () => {
      /**
       * TEST GRAPH stack increase
       * TEST GRAPH derivative functions --> Outsource to Mat??
       */
    });
  });
});

