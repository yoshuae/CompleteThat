import numpy as np
from scipy import misc
from completethat import MatrixCompletion

def blur(imgpath, delta = 0.65):
    """
        Function to blur the image
    """
    photo = scipy.misc.imread(imgpath, flatten=True).astype(float) #Read as grayscale
    m, n = photo.shape
    p = round(delta * (m * n - 1)) #Number of non-blanks
    A = np.zeros((m * n, 1), dtype=bool)
    A[0:p] = True
    ind = np.random.permutation(range(m*n))
    A = A[ind]
    A = A.reshape((m,n))
    return photo, A
	

if __name__ == '__main__':
    
    # Read image and randomly erase 35% of the pixels
    photo, A = blur('./columbia_1.png')
    M = np.copy(photo)
    M[~A] = np.nan
    problem = MatrixCompletion(M)

    # Solve the problem
    problem.complete_it('ASD')
    X = np.copy(problem.get_optimized_matrix())