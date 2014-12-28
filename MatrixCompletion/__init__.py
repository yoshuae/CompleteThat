""" CompleteThat is a python package that solves the low rank matrix completion
problem. Given a low rank matrix with partial entries the package solves an
optimization problem to estimate the missing entries.

Mathematically, the package solves a relaxation (using the nuclear norm or the
Frobenius norm of the objective matrix) of the following problem:
    minimize_{X} X
    st. X(i,j) = M(i,j) \forall (i,j)\in \Omega,

Where, M represents the data matrix and \Omega represents the set of p observed entries of M

    Usage:

    >>> from completethat import MatrixCompletion
    >>> problem = MatrixCompletion(M, Omega) 
    >>> problem.complete_it(algo_name) 
    >>> X = problem.get_matrix() 
    >>> out_info = problem.get_out() #Extra info (iterations, ect)

"""
from matrix_completion import MatrixCompletion
