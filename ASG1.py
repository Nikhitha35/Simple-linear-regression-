import numpy as np
import pandas as pd

def importdata(path = '/content/enjoysport.csv'):

	data = pd.read_csv(path, header = 0, skiprows = 0)

	print ("Dataset Length: ", len(data))
	print ("Dataset Shape: ", data. shape)

	data.target = data['play']

	return data

def Entropy(data): #Entropy(data) Entropy(data. Loc[data['sky']=='sunny'])

	d = data.iloc[:,-1]
	d = d.value_counts()
	s = 0

	for v in d.keys():

		p = d[v]/sum(d)
		s -= p*np.log2(p)

	return(s)



def IG(data,A): #IG(data, 'sky')

	Es = Entropy(data)
	val = values(data[A])
	s_c = data[A].value_counts()
	s_v = []

	for v in range(len(val)):
		ds = data[data[A] == val[v]]
		s = 0;
		for res in values(data.iloc[:,-1]):
			try:
				pi = ds.iloc[:, -1].value_counts()[res]/len(ds)
				s -= pi*np.log2(pi)
			except:
				s = 0
		s_v.append(s)

	for i in range (len(val)):
		Es = Es - s_c[val[i]]*s_v[i]/sum(s_c)

	return Es



class Node():

	def __init__(self,name = None, attr=None):
		self.name = name
		self.attr = attr

	def call_(self):
		return self.name

def DTNode(data, features_used):
	node = Node()
	IGmax = 0; vbest = None
	val_list = [v for v in values(data)[:-1] if v not in features_used]

	if val_list != []:

		for v in val_list:

			if IG(data,v) > IGmax:
				IGmax = IG(data,v)
				v_best = v

		if v_best:

			features_used.append(v_best)
			node.name = v_best
			node.attr = values(data[v_best])

			return (node)
		else:
			return (None)

	return (None)

def DTClassifier(data, features_used):

	root = DTNode(data, features_used)

	DT_dict = {}

	if root != None:
		item = []
		for attr in root.attr:
			dataN = data[data[root.name] == attr]
			if Entropy(dataN) == 0:
				item.append((attr,values(dataN.iloc[:,-1])[0]))
			else:
				dt = DTClassifier(dataN, features_used)
				item.append((attr,dt))

		DT_dict[root.name] = item

	return (DT_dict)
