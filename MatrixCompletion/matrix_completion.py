import numpy as np
import importlib
from scipy import linalg

class MatrixCompletion:
    """ A general class to represent a matrix completion problem

    Data members (private)
    ==================== 
    M:= data matrix (numpy array).
    X:= optimized data matrix (numpy array)
    out_info:= output information for the optimization (list) 


    Class methods
    ====================
    complete_it():= method to complete the matrix
    get_optimized_matrix():= method to get the solution to the problem
    get_matrix():= method to get the original matrix
    get_out():= method to get extra information on the optimization (iter
    number, convergence, objective function)

    """

    def __init__(self, X,*args, **kwargs):
        """ Constructor for the problem instance

            Inputs:
             1) X: known data matrix. Numpy array with np.nan on the unknow entries. 
                example: 
                    X = np.random.randn(5, 5)
                    X[1][3] = np.nan
                    X[0][0] = np.nan
                    X[4][4] = np.nan

        """

        # Initialization of the members
        self._M = X
        self._X = np.array(X, copy = True) #Initialize with ini data matrix
        self._out_info = []

    def get_optimized_matrix(self):
        """ Getter function to return the optimized matrix X 

            Ouput:
             1) Optimized matrix
        """
        return self._X

    def get_matrix(self):
        """ Getter function that returns the original matrix M

            Output:
            1) Original matrix M
        """
        return self._M

    def get_out(self):
        """ Getter function to return the output information 
            of the optimization

            Output:
             1) List of length 2: number of iterations and relative residual

        """
        return self._out_info

    def complete_it(self, algo_name, r = None, reltol=1e-5, maxiter=5000):
 
        """ Function to solve the optimization with the choosen algorithm 

            Input:
             1) algo_name: Algorithm name (ASD, sASD, SVT)
             2) r: rank of the matrix if performing alternating algorithm
        """
        module = importlib.import_module(algo_name)
        self._X, self._out_info = getattr(module, algo_name)(self._M, r, reltol, maxiter)

