""" CompleteThat is a python package that solves the low rank matrix completion
problem. Given a low rank matrix with partial entries the package solves an
optimization problem to estimate the missing entries.

Mathematically, the package solves a relaxation (using the nuclear norm or the
Frobenius norm of the objective matrix) of the following problem:
    minimize_{X} ||X||
    st. X(i,j) = M(i,j) \forall (i,j)\in \Omega,

Where, M represents the data matrix and \Omega represents the set of p observed entries of M

    Usage:
    # MatrixCompletion
    >>> from completethat import MatrixCompletion
    >>> problem = MatrixCompletion(M) 
    >>> problem.complete_it(algo_name) 
    >>> X = problem.get_matrix() 
    >>> out_info = problem.get_out() #Extra info (iterations, ect)

    # MatrixCompletionBD
    >>>from matrix_completion_BD import MatrixCompletionBD 
    >>>temp=MatrixCompletionBD('input_data.txt')
    >>>temp.train_sgd(dimension=6,init_step_size=.01,min_step=.000001, reltol=.001,rand_init_scale=10,   maxiter=1000,batch_size_sgd=50000,shuffle=True):
    >>>temp.validate_sgd('test_data.txt')
    >>>temp.save_model()

"""
from matrix_completion import MatrixCompletion
from matrix_completion import MatrixCompletionBD
