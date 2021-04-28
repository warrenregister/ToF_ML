'''
Methods for scraping TOF spectra data out of raw cas files.
'''
import os
import pandas as pd 
import itertools

def main():
    generate_csv()


def generate_csv(dir='/Users/warren/Desktop/Phi/TOF_ML/data/CdbSpectra/'):
	i = 0
	df = None
	for file_path in os.listdir(dir):
		data = process_cas(dir, file_path)
		if i == 0:
			df = data
		else:
			print(df, data)
			df = pd.concat([df, data])
		i += 1
	print(df)
	df.to_csv('processed_cas.csv')


def process_cas(dir, file_name):
	with open(dir + file_name) as file:
			last = False
			dic = {}
			dic['file_name'] = file_name
			try:
				for line in file.readlines():
					if ':' in line:
						contents = line.split(': ')

						try:
							contents[1] = float(contents[1])
						except:
							pass 

						dic[contents[0]] = [contents[1]]
					elif line.strip() == 'EOFH':
						last = True
					elif last:
						dic['sequence'] = line.strip()
						return(pd.DataFrame.from_dict(dic))
			except:
				pass



if __name__ == '__main__':
	main()