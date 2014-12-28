import numpy as np
import os, random
#from MatrixCompletion import MatrixCompletion
## for tab delimitted input files 

#class MatrixCompletionBD(MatrixCompletion):

""" A general class for matrix factorization via stochastic gradient descent
==================== 
Add Info:
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

class MatrixCompletionBD:		
	#initialize Matrix Completion BD object
	def __init__(self,file_path='/Users/joshua/Desktop/School/Big_Data_Analytics/Data/user_movie_data/movie_train.txt',delimitter='\t',*args, **kwargs):
		self.file=file_path
		self.delimitter='\t'
		## initialize users and objects dictionaries
		self.users=dict()
		self.items=dict()
		
	# shuffle line of file for sgd method, improves performance/convergence
	def shuffle_file(self,batch_size=50000):
		data=open(self.file)
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
		
	# split input file randomly into training and test set for cross validation
	def file_split(self,percent_train=.80, train_file='data_train.csv', test_file='data_test.csv'):
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
		
	def train_sgd(self,dimension=6,reltol=.1, maxiter=1000,batch_size_sgd=50000,shuffle=True):
		
		ratings=[]
		alpha=.00001
		iteration=0
		delta_err=1
		counter=0
		while iteration != maxiter and delta_err > reltol :

			data=open(self.file)
			total_err=[0]
	
			for line in data:
				#line=data.readline()
				record=line[0:len(line)-1].split(self.delimitter)
				record[2]=float(record[2])
				#print temp
				# format : user, movie,5-point-ratings
				ratings.append(record[2])
				# start with 5-point ratings
				#if record[0] in self.users and record[1] in self.items :
				try:
					# do some updating
					# updates
					error=record[2]-4-np.dot(self.users[record[0]],self.items[record[1]])
					#if np.isnan(error):
					#	print (record,self.users[record[0]],self.items[record[1]])
					#print(error)
					self.users[record[0]]=self.users[record[0]]+alpha*2*error*self.items[record[1]]
					#print self.users[record[0]]
					self.items[record[1]]=self.items[record[1]]+alpha*2*error*self.users[record[0]]
					total_err.append(error**2)
				except:
				#else:
					counter+=1
					if record[0] not in self.users:
						self.users[record[0]]=np.random.rand(dimension)
					if record[1] not in self.items:
						self.items[record[1]]=np.random.rand(dimension)
					#self.users[record[0]][params]=np.random.rand(dimension)
					#self.items[record[1]][params]=np.random.rand(dimension)
					#self.users[record[0]][count]+=1
					#self.items[record[1]][count]+=1
					
					
			data.close()
			if shuffle: self.shuffle_file(batch_size=batch_size_sgd)
			iteration+=1
			mse=sum(total_err)*1.0/len(total_err)
			delta_err=abs(delta_err-mse)
			# we are printing error after each pass, lets make it 
			print ('Delta Error: %f ' % delta_err)
		print ('iterations: %f ' % iteration)
		print ('Delta Error: %f ' % delta_err)
		print ('MSE: %f ' % mse)
		print counter
		
	#save model user and item parameters to text file	
	# user_key, user_vector entries
	# item_key, item_vector entries 
	def save_model(self,user_out='user_params.txt',item_out='item_params.txt'):
		
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
			
	## run model on test/validation set	
	def validate_sgd(self,test_file_path='/Users/joshua/Desktop/School/Big_Data_Analytics/Data/user_movie_data/movie_test.txt'):
		
		mse=[]
		test_set=open(test_file_path)
		for line in test_set: 
			record=line[0:len(line)-1].split(self.delimitter)
			record[2]=float(record[2])
			error=record[2]-4-np.dot(self.users[record[0]],self.items[record[1]])
			mse.append(error**2)
		print 'avg mean squared:'	
		return sum(mse)/len(mse)
		
	# method to build a full matrix to mirror Esteban's MatrixCompletion package
	def build_matrix(self):
		pass
		
		


