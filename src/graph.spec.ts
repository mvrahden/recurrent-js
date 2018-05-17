import { Graph, Mat, MatOps } from '.';

/**
 * TEST GRAPH derivative functions --> Outsource to Mat??
 */

describe('Graph Operations:', () => {

  let sut: Graph;
  let mat1: Mat;

  beforeEach(() => {
    // Turn Matrix into Test Double through Method-Patching
    initializeMatrixSpyFunctions();

    sut = new Graph(true);
    mat1 = new Mat(2, 4);
  });

  describe('Matrix Operation Call:', () => {  

    describe('Single Matrix Operations:', () => {
      it('given a graph and a matrix >> rowPluck >> should have called MatOps.rowPluck', () => {
        const anyIndex = 0;
        sut.rowPluck(mat1, anyIndex);
  
        expect(MatOps.rowPluck).toHaveBeenCalled();
        expect(MatOps.rowPluck).toHaveBeenCalledWith(mat1, 0);
      });
      
      describe('Monadic Matrix Operations', () => {
        it('given a graph and a matrix >> tanh >> should have called MatOps.tanh', () => {
          sut.tanh(mat1);
          
          expectSpyMethodToHaveBeenCalled(MatOps.tanh);
        });
        
        it('given a graph and a matrix >> sig >> should have called MatOps.sig', () => {
          sut.sig(mat1);
          
          expectSpyMethodToHaveBeenCalled(MatOps.sig);
        });
    
        it('given a graph and a matrix >> relu >> should have called MatOps.relu', () => {
          sut.relu(mat1);
          
          expectSpyMethodToHaveBeenCalled(MatOps.relu);
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

      it('given a graph and 2 matrices >> mul >> should have called MatOps.mul', () => {
        sut.mul(mat1, mat2);
  
        expectSpyMethodToHaveBeenCalled(MatOps.mul);
      });
      
      it('given a graph and 2 matrices >> add >> should have called MatOps.add', () => {
        sut.add(mat1, mat2);
  
        expectSpyMethodToHaveBeenCalled(MatOps.add);
      });
      
      it('given a graph and 2 matrices >> dot >> should have called MatOps.dot', () => {
        sut.dot(mat1, mat2);
        
        expectSpyMethodToHaveBeenCalled(MatOps.dot);
      });
      
      it('given a graph and 2 matrices >> eltmul >> should have called MatOps.eltmul', () => {
        sut.eltmul(mat1, mat2);
        
        expectSpyMethodToHaveBeenCalled(MatOps.eltmul);
      });
  
      const expectSpyMethodToHaveBeenCalled = (spy: Function): void => {
        expect(spy).toHaveBeenCalled();
        expect(spy).toHaveBeenCalledWith(mat1, mat2);
      }
    });
  });
  
  describe('Backpropagation Stack:', () => {

    describe('Without Backpropagation:', () => {

      beforeEach(() => {
        sut = new Graph(false); // create a Graph without backpropagation

        // Turn Graph Property into Test Double through Method-Patching
        spyOn(sut['backpropagationStack'], 'push');
      });

      describe('Single Matrix Operations:', () => {
        it('given a graph without backpropagation >> tanh >> should add function on stack', () => {
          sut.tanh(mat1);

          expectOperationNotToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> sig >> should add function on stack', () => {
          sut.sig(mat1);

          expectOperationNotToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> relu >> should add function on stack', () => {
          sut.relu(mat1);

          expectOperationNotToBePushedToBackpropagationStack();
        });
      });
      
      describe('Dual Matrix Operations:', () => {

        let mat2: Mat;

        beforeEach(() => {
          mat2 = new Mat(2,4);
        });

        it('given a graph without backpropagation >> add >> should add function on stack', () => {
          sut.add(mat1, mat2);

          expectOperationNotToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> mul >> should add function on stack', () => {
          sut.mul(mat1, mat2);

          expectOperationNotToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> dot >> should add function on stack', () => {
          sut.dot(mat1, mat2);

          expectOperationNotToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> eltmul >> should add function on stack', () => {
          sut.eltmul(mat1, mat2);

          expectOperationNotToBePushedToBackpropagationStack();
        });
      });

      const expectOperationNotToBePushedToBackpropagationStack = () => { 
        expect(sut['backpropagationStack'].push).not.toHaveBeenCalled();
      };
    });

    describe('With Backpropagation:', () => {

      beforeEach(() => {
        // Turn Graph Property into Test Double through Method-Patching
        sut = new Graph(true);
        spyOn(sut['backpropagationStack'], 'push');
      });

      describe('Single Matrix Operations:', () => {

        it('given a graph without backpropagation >> tanh >> should add function on stack', () => {
          sut.tanh(mat1);

          expectOperationToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> sig >> should add function on stack', () => {
          sut.sig(mat1);

          expectOperationToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> relu >> should add function on stack', () => {
          sut.relu(mat1);

          expectOperationToBePushedToBackpropagationStack();
        });
      });
      
      describe('Dual Matrix Operations:', () => {
        
        let mat2: Mat;

        beforeEach(() => {
          mat2 = new Mat(2,4);
        });

        it('given a graph without backpropagation >> add >> should add function on stack', () => {
          sut.add(mat1, mat2);

          expectOperationToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> mul >> should add function on stack', () => {
          sut.mul(mat1, mat2);

          expectOperationToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> dot >> should add function on stack', () => {
          sut.dot(mat1, mat2);

          expectOperationToBePushedToBackpropagationStack();
        });

        it('given a graph without backpropagation >> eltmul >> should add function on stack', () => {
          sut.eltmul(mat1, mat2);

          expectOperationToBePushedToBackpropagationStack();
        });
      });

      const expectOperationToBePushedToBackpropagationStack = () => { 
        expect(sut['backpropagationStack'].push).toHaveBeenCalled();
      };
    });
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
