CompleteThat (v0.1dev) 
====================
CompleteThat is a python package that solves the low rank matrix completion
problem. Given a low rank matrix with partial entries the package solves an
optimization problem to estimate the missing entries.

Mathematically, the package solves a relaxation (using the nuclear norm or the 
Frobenius norm of the objective matrix) of the following problem:

  minimize_{X} ||X||
  st. X(i,j) = M(i,j) \forall (i,j)\in \Omega,
  where, M represents the data matrix and \Omega represents the set of p
  observed entries of M

Usage
====================

For matrices that fit into memory use the `MatrixCompletion`

```python
# MatrixCompletion
from completethat import MatrixCompletion
problem = MatrixCompletion(M)
problem.complete_it(algo_name)
X = problem.get_matrix()
out_info = problem.get_out() #Extrainformation (number of iterations, ect)

# MatrixCompletionBD
from completethat import MatrixCompletionBD
problem = MatrixCompletionBD('input_data.txt')
problem.train_sgd(dimension=6,init_step_size=.01,min_step=.000001, reltol=.001,rand_init_scale=10,   maxiter=1000,batch_size_sgd=50000,shuffle=True)
problem.validate_sgd('test_data.txt')
problem.save_model()
```
Authors 
====================

This package was written by Joshua Edgerton and Esteban Fajardo

Acknowledgments
====================

This package is the result of the final project for the class EEOR E4650: Convex
Optimization at Columbia University, Fall 2014. We would like to thank the
authors of the different algorithms used in the package to solve the problem.
