import { Graph, Mat } from '.';
import { MatOps } from './utils/mat-ops';

/**
 * TEST GRAPH derivative functions --> Outsource to Mat??
 */

describe('Graph Operations:', () => {

  let sut: Graph;
  let mat: Mat;

  beforeEach(() => {
    // Turn Matrix into Test Double through Method-Patching
    initializeMatrixSpyFunctions();

    sut = new Graph();
    mat = new Mat(2, 4);
  });

  describe('Matrix Operation Call:', () => {  

    describe('Single Matrix Operations:', () => {
      it('given a graph and a matrix >> rowPluck >> should have called MatOps.rowPluck', () => {
        const anyIndex = 0;
        sut.rowPluck(mat, anyIndex);
  
        expect(MatOps.rowPluck).toHaveBeenCalled();
        expect(MatOps.rowPluck).toHaveBeenCalledWith(mat, 0);
      });
      
      describe('Monadic Matrix Operations', () => {
        it('given a graph and a matrix >> tanh >> should have called MatOps.tanh', () => {
          sut.tanh(mat);
          
          expectSpyMethodToHaveBeenCalled(MatOps.tanh);
        });
        
        it('given a graph and a matrix >> sig >> should have called MatOps.sig', () => {
          sut.sig(mat);
          
          expectSpyMethodToHaveBeenCalled(MatOps.sig);
        });
    
        it('given a graph and a matrix >> relu >> should have called MatOps.relu', () => {
          sut.relu(mat);
          
          expectSpyMethodToHaveBeenCalled(MatOps.relu);
        });
    
        const expectSpyMethodToHaveBeenCalled = (spy: Function): void => {
          expect(spy).toHaveBeenCalled();
          expect(spy).toHaveBeenCalledWith(mat);
        }
      });
    });

    describe('Dual Matrix Operations:', () => {

      let mat2: Mat;

      beforeEach(() => {
        mat2 = new Mat(2, 4);
      });

      it('given a graph and 2 matrices >> mul >> should have called MatOps.mul', () => {
        sut.mul(mat, mat2);
  
        expectSpyMethodToHaveBeenCalled(MatOps.mul);
      });
      
      it('given a graph and 2 matrices >> add >> should have called MatOps.add', () => {
        sut.add(mat, mat2);
  
        expectSpyMethodToHaveBeenCalled(MatOps.add);
      });
      
      it('given a graph and 2 matrices >> dot >> should have called MatOps.dot', () => {
        sut.dot(mat, mat2);
        
        expectSpyMethodToHaveBeenCalled(MatOps.dot);
      });
      
      it('given a graph and 2 matrices >> eltmul >> should have called MatOps.eltmul', () => {
        sut.eltmul(mat, mat2);
        
        expectSpyMethodToHaveBeenCalled(MatOps.eltmul);
      });
  
      const expectSpyMethodToHaveBeenCalled = (spy: Function): void => {
        expect(spy).toHaveBeenCalled();
        expect(spy).toHaveBeenCalledWith(mat, mat2);
      }
    });
  });
  
  describe('Backpropagation Stack:', () => {

    beforeEach(() => {
      sut = new Graph(); // create a Graph without backpropagation

      // Turn Graph Property into Test Double through Method-Patching
      spyOn(sut['backpropagationStack'], 'push');
    });

    describe('Without Backpropagation:', () => {

      describe('Single Matrix Operations:', () => {
        
        it('given a graph without backpropagation >> gauss >> should add function on stack', () => {
          const std = new Mat(2, 4);
          sut.gauss(mat, std); // Exception to the rule, gauss adds noise but does not change the slope.

          expectOperationNotToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> rowPluck >> should add function on stack', () => {
          const rowIndex = 1;
          sut.rowPluck(mat, rowIndex);

          expectOperationNotToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> tanh >> should add function on stack', () => {
          sut.tanh(mat);

          expectOperationNotToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> sig >> should add function on stack', () => {
          sut.sig(mat);

          expectOperationNotToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> relu >> should add function on stack', () => {
          sut.relu(mat);

          expectOperationNotToBePushedToBackpropagationStack();
        });
      });
      
      describe('Dual Matrix Operations:', () => {

        let mat2: Mat;

        beforeEach(() => {
          mat2 = new Mat(2,4);
        });

        it('given a graph without backpropagation >> add >> should add function on stack', () => {
          sut.add(mat, mat2);

          expectOperationNotToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> mul >> should add function on stack', () => {
          sut.mul(mat, mat2);

          expectOperationNotToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> dot >> should add function on stack', () => {
          sut.dot(mat, mat2);

          expectOperationNotToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> eltmul >> should add function on stack', () => {
          sut.eltmul(mat, mat2);

          expectOperationNotToBePushedToBackpropagationStack();
        });
      });

    });

    describe('With Backpropagation:', () => {

      beforeEach(() => {
        sut = new Graph();
        sut.setOperationSequenceMemoryTo(true);
        // Turn Graph Property into Test Double through Method-Patching
        spyOn(sut['backpropagationStack'], 'push');
      });

      describe('Single Matrix Operations:', () => {

        it('given a graph without backpropagation >> gauss >> should NOT add function on stack', () => {
          const std = new Mat(2, 4);
          sut.gauss(mat, std); // Exception to the rule, gauss adds noise but does not change the slope.

          expectOperationNotToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> rowPluck >> should add function on stack', () => {
          const rowIndex = 1;
          sut.rowPluck(mat, rowIndex);

          expectOperationToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> tanh >> should add function on stack', () => {
          sut.tanh(mat);

          expectOperationToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> sig >> should add function on stack', () => {
          sut.sig(mat);

          expectOperationToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> relu >> should add function on stack', () => {
          sut.relu(mat);

          expectOperationToBePushedToBackpropagationStack();
        });
      });
      
      describe('Dual Matrix Operations:', () => {
        
        let mat2: Mat;

        beforeEach(() => {
          mat2 = new Mat(2,4);
        });

        it('given a graph without backpropagation >> add >> should add function on stack', () => {
          sut.add(mat, mat2);

          expectOperationToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> mul >> should add function on stack', () => {
          sut.mul(mat, mat2);

          expectOperationToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> dot >> should add function on stack', () => {
          sut.dot(mat, mat2);

          expectOperationToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> eltmul >> should add function on stack', () => {
          sut.eltmul(mat, mat2);

          expectOperationToBePushedToBackpropagationStack();
        });
      });

    });

    const expectOperationToBePushedToBackpropagationStack = () => {
      expect(sut['backpropagationStack'].push).toHaveBeenCalled();
    };

    const expectOperationNotToBePushedToBackpropagationStack = () => {
      expect(sut['backpropagationStack'].push).not.toHaveBeenCalled();
    };
  });

  describe('Backwards Differentiation:', () => {

  });

  const initializeMatrixSpyFunctions = (): void => {
    spyOn(MatOps, 'rowPluck');
    spyOn(MatOps, 'tanh');
    spyOn(MatOps, 'sig');
    spyOn(MatOps, 'relu');
    spyOn(MatOps, 'mul');
    spyOn(MatOps, 'add');
    spyOn(MatOps, 'dot');
    spyOn(MatOps, 'eltmul');
  };
});
