# Class: `MatOpts`

The class `MatOpts` holds all the necessary matrix operations for assambling a forward-pass and their respective derivatives to perform backpropagation.
All methods are static (stateindependent) methods.

* Matrix Operations:
  * Execute the actual Matrix Operation
  * Throw an Error Message if dimensions are not aligned
  * Available Matrix Operations are:
    * `[static] rowPluck(m: Mat, rowIndex: number): Mat`
    * `[static] getRowPluckBackprop(m: Mat, rowIndex: number, out: Mat): Mat`
    * `[static] gauss(m: Mat, std: Mat): Mat`
    * `[static] tanh(m: Mat): Mat`
    * `[static] getTanhBackprop(m: Mat, out: Mat): Mat`
    * `[static] sig(m: Mat): Mat`
    * `[static] getSigmoidBackprop(m: Mat, out: Mat): Mat`
    * `[static] relu(m: Mat): Mat`
    * `[static] getReluBackprop(m: Mat, out: Mat): Mat`
    * `[static] add(mat1: Mat, mat2: Mat): Mat`
    * `[static] getAddBackprop(mat1: Mat, mat2: Mat, out: Mat): Mat`
    * `[static] mul(mat1: Mat, mat2: Mat): Mat`
    * `[static] getMulBackprop(mat1: Mat, mat2: Mat, out: Mat): Mat`
    * `[static] dot(mat1: Mat, mat2: Mat): Mat`
    * `[static] getDotBackprop(mat1: Mat, mat2: Mat, out: Mat): Mat`
    * `[static] eltmul(mat1: Mat, mat2: Mat): Mat`
    * `[static] getEltmulBackprop(mat1: Mat, mat2: Mat, out: Mat): Mat`
    