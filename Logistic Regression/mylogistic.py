''' 
This program is simple implementation of Logistic Regression

Data: 
	Features= 2 => grade1,grade2
	Label => (0,1)
'''
import math
import numpy as np 
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

df = pd.read_csv("data.csv", header=0)	#read csv file and returns dataFrame
#df.columns = ['grade1', 'grade2', 'label;;;;'] Actual present
df.columns = ["grade1","grade2","label"]  #we have to do this so that label;;;; can we written as label

#print("Columns:")
#print(df.columns)		
#print(df.index)			#RangeIndex(start=0, stop=100, step=1)
#for i in df.index:
#	print(df["grade1"][i])

x= df["label"].map(lambda x: float(x.rstrip(';')))
# x.rstrip(";")  is to trim ';' from the end of string all occurences
# here x is <class 'pandas.core.series.Series'>
#print(type(x))
#print(x)

#Data is cleaned up
#Format data a/c to our need
#Need: 2D array of 'grade1','grade2' and 1D array of label
X= df[["grade1","grade2"]]	#returns the Data frame object with only these two columns
X= np.array(X)			#convert to np array

X = min_max_scaler.fit_transform(X)

Y= df["label"].map(lambda x: float(x.rstrip(';')))
Y= np.array(Y)
#X is 2D array and Contains Features 'grade1','grade2'
#Y is 1D array and contains label

#creating testing and training sets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)

# train scikit learn model
clf = LogisticRegression()
clf.fit(X_train,Y_train)	#fit model a/c to training data
#print(Y_test)			#actual label
#print(clf.predict(X_test))	#predicted label
scikit_score = clf.score(X_test,Y_test)
print('Scikit Score: ',scikit_score)#mean accuracy

def Sigmoid(z):
	#G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0*z))))# why use extra bracket
	G_of_Z = float(1.0 / float(1.0 + math.exp(-1.0*z)))
	return G_of_Z

def Hypotheses(theta, x):
	z = 0
	#for i in range(len(theta)):	try range instead of xrange
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return Sigmoid(z)

def Cost_Function(X,Y,theta,m):
	sumOfErrors = 0
	for i in range(m):
		xi = X[i]
		hi = Hypotheses(theta,xi)
		if Y[i] == 1:
			error = Y[i] * math.log(hi)
			#error = 1.0*math.log(hi)
		elif Y[i] == 0:
			error = (1 - Y[i]) * math.log(1 - hi)
		# error = Y[i]*math.log(hi) + (1 - Y[i])*math.log(1 - hi)
		sumOfErrors += error
	const = -1/m 		#should not we use float(-1/m)
	J = const * sumOfErrors
	print( 'Cost is ', J)
	return J

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
	''' derivative del(J)/del(thetaj) = 1/m Summ((hi-yi)*xij '''
	sumErrors=0
	for i in range(m):
		xi = X[i]
		xij = xi[j]
		hi = Hypotheses(theta,X[i])
		error = (hi - Y[i])* xij
		sumErrors +=error	
	m = len(Y)	# Do we need this ??
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J

def Gradient_Descent(X,Y,theta,m,alpha):
	new_theta = []
	for j in range(len(theta)):
		CFDerivative = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
		new_theta_value = theta[j] - CFDerivative
		new_theta.append(new_theta_value)
	return new_theta

def Logistic_Regression(X,Y,alpha,theta,num_iters):
	m= len(Y)
	for x in range(num_iters):
		new_theta = Gradient_Descent(X,Y,theta,m,alpha)
		theta = new_theta
		if x % 100 == 0:
			#print cost, theta after every 100th iteration
			cost = Cost_Function(X,Y,theta,m)
			print ('theta :',theta)
			print ('cost :', cost)
	test_model(theta)

def test_model(theta):
	score = 0
	length = len(X_test)
	for i in range(length):
		prediction = round(Hypotheses(X_test[i],theta))
		answer = Y_test[i]
		if prediction == answer:
			score += 1

	print ('total testing eg : ',length)
	print ('total correct response :',score)

	acc = (float(score) / float(length))*100
	print('MyAccuracy :',acc)
	print('Scikit Accuracy :', scikit_score*100)		

def main():
	initial_theta = [0,0]
	alpha = 0.1
	iterations = 10000
	Logistic_Regression(X,Y,alpha,initial_theta,iterations)

main()