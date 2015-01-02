import numpy as np
import os
from scipy import linalg

class MatrixCompletion:
    """ A general class to represent a matrix completion problem

    Data members 
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


    def _ASD(self, M, r = None, reltol=1e-5, maxiter=5000):
        """
        Alternating Steepest Descent (ASD)
        Taken from Low rank matrix completion by alternating steepest descent methods
        Jared Tanner and Ke Wei
        SIAM J. IMAGING SCIENCES (2014)
        
        We have a matrix M with incomplete entries,
        and want to estimate the full matrix
        
        Solves the following relaxation of the problem:
        minimize_{X,Y} \frac{1}{2} ||P_{\Omega}(Z^0) - P_\{Omega}(XY)||_F^2
        Where \Omega represents the set of m observed entries of the matrix M
        and P_{\Omega}() is an operator that represents the observed data. 
        
        Inputs:
         M := Incomplete matrix, with NaN on the unknown matrix
         r := hypothesized rank of the matrix
        
        Usage:
         Just call the function _ASD(M)
        """
    
        # Get shape and Omega
        m, n = M.shape
        if r == None:
            r = min(m, n, 5)
    
        Omega = ~np.isnan(M)
        relres = reltol * linalg.norm(M[Omega]) #set relative error
    
        # Initialize
        X = np.random.randn(m, r)
        Y = np.random.randn(r, n)
        itres = np.zeros((maxiter+1, 1)) 
    
        XY = np.dot(X, Y)
        diff_on_omega = M[Omega] - XY[Omega]
        res = linalg.norm(diff_on_omega)
        iter = 0
        itres[iter] = res
    
        while iter < maxiter and res >= relres:
            
            # Gradient for X
            diff_on_omega_matrix = np.zeros((m,n))
            diff_on_omega_matrix[Omega] = diff_on_omega
            grad_X = np.dot(diff_on_omega_matrix, np.transpose(Y))
            
            # Stepsize for X
            delta_XY = np.dot(grad_X, Y)
            tx = linalg.norm(grad_X,'fro')**2/linalg.norm(delta_XY)**2
        
            # Update X
            X = X + tx*grad_X;
            diff_on_omega = diff_on_omega-tx*delta_XY[Omega]
        
            # Gradient for Y
            diff_on_omega_matrix = np.zeros((m,n))
            diff_on_omega_matrix[Omega] = diff_on_omega
            Xt = np.transpose(X)
            grad_Y = np.dot(Xt, diff_on_omega_matrix)
        
            # Stepsize for Y
            delta_XY = np.dot(X, grad_Y)
            ty = linalg.norm(grad_Y,'fro')**2/linalg.norm(delta_XY)**2
        
            # Update Y
            Y = Y + ty*grad_Y
            diff_on_omega = diff_on_omega-ty*delta_XY[Omega]
            
            res = linalg.norm(diff_on_omega)
            iter = iter + 1
            itres[iter] = res
    
        M_out = np.dot(X, Y)
    
        out = [iter, itres[iter]/linalg.norm(M[Omega])]
    
        return M_out, out    

    def _sASD(self, M, r = None, reltol=1e-5, maxiter=10000):
        """
        Scaled Alternating Steepest Descent (ScaledASD)
        Taken from:
        Low rank matrix completion by alternating steepest descent methods
        Jared Tanner and Ke Wei
        SIAM J. IMAGING SCIENCES (2014)
        
        We have a matrix M with incomplete entries,
        and want to estimate the full matrix
        
        Solves the following relaxation of the problem:
        minimize_{X,Y} \frac{1}{2} ||P_{\Omega}(Z^0) - P_\{Omega}(XY)||_F^2
        Where \Omega represents the set of m observed entries of the matrix M
        and P_{\Omega}() is an operator that represents the observed data. 
        
        Inputs:
         M := Incomplete matrix, with NaN on the unknown matrix
         r := hypothesized rank of the matrix
        
        Usage:
         Just call the function _sASD(M)
        """
    
    
        # Get shape and Omega
        m, n = M.shape
        if r == None:
            r = min(m, n, 5)
    
        Omega = ~np.isnan(M)
        relres = reltol * linalg.norm(M[Omega]) #set relative error
    
        # Initialize
        identity = np.identity(r);
        X = np.random.randn(m, r)
        Y = np.random.randn(r, n)
        itres = np.zeros((maxiter+1, 1)) 
    
        XY = np.dot(X, Y)
        diff_on_omega = M[Omega] - XY[Omega]
        res = linalg.norm(diff_on_omega)
        iter = 0
        itres[iter] = res
    
        while iter < maxiter and res >= relres:
    
            # Gradient for X
            diff_on_omega_matrix = np.zeros((m,n))
            diff_on_omega_matrix[Omega] = diff_on_omega
            grad_X = np.dot(diff_on_omega_matrix, np.transpose(Y))
    
            # Scaled gradient
            scale = linalg.solve(np.dot(Y, np.transpose(Y)), identity)
            dx = np.dot(grad_X, scale) 
    
            delta_XY = np.dot(dx, Y)
            tx = np.trace(np.dot(np.transpose(dx),grad_X))/linalg.norm(delta_XY[Omega])**2
    
            # Update X
            X = X + tx*dx
            diff_on_omega = diff_on_omega-tx*delta_XY[Omega]
    
            # Gradient for Y
            diff_on_omega_matrix = np.zeros((m,n))
            diff_on_omega_matrix[Omega] = diff_on_omega
            Xt = np.transpose(X)
            grad_Y = np.dot(Xt, diff_on_omega_matrix)
    
            # Scaled gradient
            scale = linalg.solve(np.dot(Xt, X), identity)
            dy = np.dot(scale, grad_Y) 
    
            # Stepsize for Y
            delta_XY = np.dot(X, dy)
            ty = np.trace(np.dot(dy,np.transpose(grad_Y)))/linalg.norm(delta_XY[Omega])**2
    
            # Update Y
            Y = Y + ty*dy
            diff_on_omega = diff_on_omega-ty*delta_XY[Omega]
    
            # Update iteration information
            res = linalg.norm(diff_on_omega)
            iter = iter + 1
            itres[iter] = res 
    
        M_out = np.dot(X, Y)
    
        out = [iter, itres[iter]/linalg.norm(M[Omega])]
    
        return M_out, out

    def complete_it(self, algo_name, r = None, reltol=1e-5, maxiter=5000):
 
        """ Function to solve the optimization with the choosen algorithm 

            Input:
             1) algo_name: Algorithm name (ASD, sASD, SVT)
             2) r: rank of the matrix if performing alternating algorithm
        """
        if algo_name == "ASD":
            self._X, self._out_info = self._ASD(self._M, r, reltol, maxiter)
        elif algo_name == "sASD":
            self._X, self._out_info = self._sASD(self._M, r, reltol, maxiter)
        else:
            raise NameError("Algorithm name not recognized")

class MatrixCompletionBD:		
	""" 
	A general class for matrix factorization via stochastic gradient descent

	Class members
	==================== 
	file: three column file of user, item, and value to build models


	Class methods
	====================
	train_sgd():= method to complete the matrix via sgd
	shuffle_file():= method to 'psuedo' shuffle input file in chunks
	file_split():= method to split input file into training and test set
	save_model():= save user and items parameters to text file
	validate_sgd():= validate sgd model on test set 
	build_matrix():= for smaller data build complete matrix in pandas df or numpy matrix? 
	"""


	def __init__(self,file_path,delimitter='\t',*args, **kwargs):
		 """ 
		     Object constructor
		     Initialize Matrix Completion BD object
		 """
		 self._file = file_path
		 self._delimitter = '\t'
		 self._users = dict()
		 self._items = dict()

	def shuffle_file(self,batch_size=50000):
		"""

		Shuffle line of file for sgd method, improves performance/convergence

		"""
		data = open(self._file)
		temp_file=open('temp_shuffled.txt','w')
		temp_array=[]
		counter=0
		for line in data:
			counter+=1
			temp_array.append(line)
			if counter==batch_size : 	
				random.shuffle(temp_array)
				for entry in temp_array:
					temp_file.write(entry)
				temp_array=[]
				counter=0

		if len(temp_array)>0:
			random.shuffle(temp_array)
			for entry in temp_array:
				temp_file.write(entry)

		data.close()
		temp_file.close()
		system_string='mv temp_shuffled.txt ' + self.file 
		os.system(system_string)

	def file_split(self,percent_train=.80, train_file='data_train.csv', test_file='data_test.csv'):
		"""

		split input file randomly into training and test set for cross validation

		"""
		train=open(train_file,'w')
		test=open(test_file,'w')
		temp_file=open(self.file)
		for line in temp_file:
			if np.random.rand()<percent_train:
				train.write(line)
			else:
				test.write(line)

		train.close()
		test.close()
		print('test file written as ' + train_file)
		print('test file written as ' + test_file)
		temp_file.close()

	def train_sgd(self,dimension=6,init_step_size=.01,min_step=.000001,reltol=.05,rand_init_scalar=1, maxiter=100,batch_size_sgd=50000,shuffle=True):

		alpha=init_step_size
		iteration=0
		delta_err=1
		new_mse=reltol+10
		counter=0

		while iteration != maxiter and delta_err > reltol :

			data=open(self.file)
			total_err=[0]
			if alpha>=min_step: alpha*=.3
			else: alpha=min_step

			for line in data:

				record=line[0:len(line)-1].split(self.delimitter)
				record[2]=float(record[2])
				# format : user, movie,5-point-ratings
				ratings.append(record[2])
				#if record[0] in self.users and record[1] in self.items :
				try:
					# do some updating
					# updates
					error=record[2]-4-np.dot(self.users[record[0]],self.items[record[1]])
					self.users[record[0]]=self.users[record[0]]+alpha*2*error*self.items[record[1]]
					self.items[record[1]]=self.items[record[1]]+alpha*2*error*self.users[record[0]]
					total_err.append(error**2)
				except:
					#else:
					counter+=1
					if record[0] not in self.users:
						self.users[record[0]]=np.random.rand(dimension)*rand_init_scalar
					if record[1] not in self.items:
						self.items[record[1]]=np.random.rand(dimension)*rand_init_scalar

			data.close()
			if shuffle: 
				self.shuffle_file(batch_size=batch_size_sgd)
		iteration+=1
		old_mse=new_mse
		new_mse=sum(total_err)*1.0/len(total_err)
		delta_err=abs(old_mse-new_mse)

    
	def save_model(self,user_out='user_params.txt',item_out='item_params.txt'):
		"""
		save model user and item parameters to text file	
		user_key, user_vector entries
		item_key, item_vector entries 
		"""
		users=open(user_out,'w')
		items=open(item_out,'w')
		for key in self.users:
			user_string= key+ self.delimitter + self.delimitter.join(map(str,list(self.users[key]))) + '\n'
			users.write(user_string)

		for key in self.items:
			item_string=key+ self.delimitter + self.delimitter.join(map(str,list(self.items[key]))) + '\n'
			items.write(item_string)

		users.close()
		items.close()

	## read saved model, particularly useful for fitting very large files! 
	def read_model(self,dimension=6,saved_user_params='user_params.txt',saved_item_params='item_params.txt'):
		"""
		
		Read the saved user and item parameters from text files to the item and user dictionaries
		
		"""
		#populate users:
		user_data=open(saved_user_params)
		for line in user_data:
			record=line[0:len(line)-1].split(self.delimitter)
			key=record.pop(0)
			params=np.array(map(float,record))
			self.users[key]=params

		user_data.close()

		#populate items:
		item_data=open(saved_item_params)
		for line in item_data:
			record=line[0:len(line)-1].split(self.delimitter)
			key=record.pop(0)
			params=np.array(map(float,record))
			self.items[key]=params

		item_data.close()

	def clear_model(self):
		"""
		
		clear the user and item parameters
		
		"""
		del self.items, self.users
		self.items=dict()
		self.users=dict()

	def validate_sgd(self,test_file_path):
		"""

		run model on test/validation set	

		"""
		mse=[]
		test_set=open(test_file_path)
		for line in test_set: 
			record=line[0:len(line)-1].split(self._delimitter)
			record[2]=float(record[2])
			error=record[2]-4-np.dot(self.users[record[0]],self.items[record[1]])
			mse.append(error**2)
		return sum(mse)/len(mse)

	def build_matrix(self):
		pass
