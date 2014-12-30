def shuffle_file(self,batch_size=50000):
	data=open(self.file)
	cat_file=open('shuffled_file.txt','w')
	cat_file.close()
	temp_file=open('temp_file.txt','w')
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
			
			if random.random()>.5:
				os.system('cat temp_file.txt shuffled_file.txt > shuffled_file.txt')
				temp_file.close()
			 	temp_file=open('temp_file.txt','w') 
			else:
				os.system('cat shuffled_file.txt temp_file.txt  > shuffled_file.txt')
				temp_file.close()
			 	temp_file=open('temp_file.txt','w')		
	
	if len(temp_array)>0:
		random.shuffle(temp_array)
		for entry in temp_array:
			temp_file.write(entry)
	
		if random.random()>.5:
			os.system('cat temp_file.txt shuffled_file.txt > shuffled_file.txt')
		else:
			os.system('cat shuffled_file.txt temp_file.txt  > shuffled_file.txt')
		
	data.close()
	temp_file.close()
	os.system('rm temp_file.txt')
	system_string='mv shuffled_file.txt ' + self.file 
	os.system(system_string)