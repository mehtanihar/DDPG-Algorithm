import numpy as np
from thread import *
import time
import socket
import sys
import argparse
import random
import ast
import math





#iterate through rows of X

def scalar_multiply(X,Y):
	#print(len(X))
	#print(len(X[0]))

	result=[[0 for j in range(len(X[0]))] for i in range(len(X))]
	
	for i in range(len(X)):
	   # iterate through columns of Y
	   for j in range(len(X[0])):
	   		result[i][j] = X[i][j] * Y

	#print(len(X))
	#print(len(X[0]))
	return result
def multiply(X,Y):
	
	
	result=[[0 for j in range(len(Y[0]))] for i in range(len(X))]
	#print(result)
	for i in range(len(X)):
	   # iterate through columns of Y
	   for j in range(len(Y[0])):
	       # iterate through rows of Y
	       for k in range(len(Y)):
	    
	        result[i][j]= result[i][j]+ (X[i][k] * Y[k][j])


	return result

def element_multiply(X,Y):
	result=[[0 for j in range(len(Y[0]))] for i in range(len(X))]
	
	for i in range(len(X)):
	   # iterate through columns of Y
	   for j in range(len(Y[0])):
	   		result[i][j] = X[i][j] * Y[i][j]

	return result

def element_subtract(X,Y):
	
	
	result=[[0 for j in range(len(Y[0]))] for i in range(len(X))]

	for i in range(len(X)):
	   # iterate through columns of Y
	   for j in range(len(Y[0])):
	   		result[i][j] = X[i][j] - Y[i][j]

	return result
def element_add(X,Y):
	result=[[0 for j in range(len(Y[0]))] for i in range(len(X))]
	
	for i in range(len(X)):
	   # iterate through columns of Y
	   for j in range(len(Y[0])):
	   		result[i][j] = X[i][j] + Y[i][j]

	return result
def get_action(curr_state,hidden_layers,theta):
	a=[]
	z=[]
	
	a.append(curr_state)

	for j in range(0,hidden_layers+1):
		#print(theta[j])
		#print(a[j])
		z.append(multiply(a[j],theta[j])) 
		a.append(np.tanh(z[j]))
	

	#print(a[hidden_layers_1+1])

	action=a[hidden_layers+1]
	

	return action


def get_value(state,action,hidden_layers,theta):
	
	curr_state=[[]]

	for i in range(0,38):
		curr_state[0].append(state[0][i])
	
	curr_state[0].append(action[0][0])
	curr_state[0].append(action[0][1])
	curr_state[0].append(action[0][2])
	
	a=[]
	a.append(curr_state)
	#print(current_state)

	for j in range(0,hidden_layers+1):
	    a.append(np.tanh(multiply(a[j],theta[j])))       
	    

	

	action=(a[hidden_layers+1][0][0])
	return action

def transpose(z):
	n=len(z)
	m=len(z[0])
	res=[]
	for i in range(0,m):
		r=[]
		for j in range(0,n):
			r.append(z[j][i])
		res.append(r)
	return res
def act(x):
	res=[]
	for i in range(0,len(x[0])):
		res.append(np.tanh(x[0][i]))
	return [res]
def getdiff(x):
	res=[]
	for i in range(0,len(x[0])):
		res.append(1-pow(x[0][i],2))
	return [res]





outputs_1=3
inputs_1=38
hidden_layers_1=1
hidden_units_1=[inputs_1,15,outputs_1]
alpha=0.001
outputs_2=1
inputs_2=41
hidden_layers_2=2
hidden_units_2=[inputs_2,15,15,outputs_2]

N=16



theta_1=[]
theta_2=[]
theta_3=[]
theta_4=[]

gamma=0.9
tau=0.001



for i in range(0,hidden_layers_1+1):
	theta_1.append([[random.random() for k in range(hidden_units_1[i+1])] for j in range(hidden_units_1[i])])


for i in range(0,hidden_layers_2+1):
	theta_2.append([[random.random() for k in range(hidden_units_2[i+1])] for j in range(hidden_units_2[i])])

for i in range(0,hidden_layers_1+1):
	theta_3.append([[random.random() for k in range(hidden_units_1[i+1])] for j in range(hidden_units_1[i])])


for i in range(0,hidden_layers_2+1):
	theta_4.append([[random.random() for k in range(hidden_units_2[i+1])] for j in range(hidden_units_2[i])])




def ddpg(file_state,file_action,minibatch):

	for i in range(len(minibatch)):


		#compute action using s[i+1]*theta_3
		new_action=get_action(minibatch[i][3],hidden_layers_1,theta_3)
		
		#compute Q using s[i+1] and a[i+1]
		q=get_value(minibatch[i][3],new_action,hidden_layers_2,theta_4)
		#ri+gamma*Q
		

		y=reward+gamma*q
		#si, ai = q 
		Q=get_value(minibatch[i][0],minibatch[i][1],hidden_layers_2,theta_2)
		
		error_outputs=error_outputs+(y-Q)


	error_outputs=error_outputs*2.0/len(minibatch)

	dJdO=[[error_outputs]]

	I=file_state
	I[0].append(file_action[0][0])
	I[0].append(file_action[0][1])
	I[0].append(file_action[0][2])

	o1=multiply(I,theta_2[0])
	o2=multiply(o1,theta_2[1])
	O=multiply(o2,theta_2[2])
	dJdt2=multiply(transpose(o2),dJdO) #15*1
	dJdo2=transpose(multiply(theta_2[2],dJdO)) #1*15
	dJdt1=multiply(transpose(o1),dJdo2)#15*15
	dJdo1=transpose(multiply(theta_2[1],transpose(dJdo2))) #1*15
	dJdt0=multiply(transpose(I),dJdo1)#41*15
	alp1=[alpha]*len(theta_2[0][0])#1*15
	alp1=[alp1]*len(theta_2[0])#41*15
	alp2=[alpha]*len(theta_2[1][0])#1*15
	alp2=[alp2]*len(theta_2[1])#15*15
	alp3=[alpha]*len(theta_2[2][0])#1*1
	alp3=[alp3]*len(theta_2[2])#15*1



	theta_2[0]=element_subtract(theta_2[0],element_multiply(alp1,dJdt0)) 
	theta_2[1]=element_subtract(theta_2[1],element_multiply(alp2,dJdt1))
	theta_2[2]=element_subtract(theta_2[2],element_multiply(alp3,dJdt2))
	dt1=[0]*len(theta_1[1][0])
	dt1=[dt1]*len(theta_1[1])#15*3

	dt0=[0]*len(theta_1[0][0])

	dt0=[dt0]*len(theta_1[0])#38*15

	for i in range(len(minibatch)):
		
		

		I[0].append(minibatch[i][1][0][0])
		I[0].append(minibatch[i][1][0][1])
		I[0].append(minibatch[i][1][0][2])
		
		o1=multiply(I,theta_2[0])
		o2=multiply(o1,theta_2[1])
		O=multiply(o2,theta_2[2])
		dJdt2=multiply(transpose(o2),dJdO) #15*1
		dJdo2=transpose(multiply(theta_2[2],dJdO)) #1*15
		dJdt1=multiply(transpose(o1),dJdo2)#15*15
		dJdo1=transpose(multiply(theta_2[1],transpose(dJdo2))) #1*15
		dJdt0=multiply(transpose(I),dJdo1)#41*15
		dJdI=transpose(multiply(theta_2[0],transpose(dJdo1)))#1-41
		
		dQda=dJdI[0][38]+dJdI[0][39]+dJdI[0][40]
		J=minibatch[i][0]#1*38
		
		net1=multiply(J,theta_1[0]) #1*15
		o1=act(net1)#1*15
		net0=multiply(o1,theta_1[1])#1*3
		O=act(net0)#1*3
		dOdnet0=getdiff(O)#1*3
		dOdt1=multiply(transpose(o1),dOdnet0)#15*3
		alp1=[dQda]*len(theta_1[1][0])
		alp1=[alp1]*len(theta_1[1])#15*3
		dt1=element_add(dt1,element_multiply(alp1,dOdt1))#15*3
		dOdo1=transpose(multiply(theta_1[1],transpose(dOdnet0)))#1*15
		dOdnet1=element_multiply(getdiff(o1),dOdo1)#1*15
		dOdt0=multiply(transpose(I),dOdnet1)#38*15
		alp1=[dQda]*len(theta_1[0][0])
		alp1=[alp1]*len(theta_1[0])
				
		dt0=element_add(dt0,element_multiply(alp1,dOdt0))#38*15
	alp1=[alpha/len(minibatch)]*len(theta_1[1][0])
	alp1=[alp1]*len(theta_1[1])	
	dt1=element_multiply(alp1,dt1)
	alp1=[alpha/len(minibatch)]*len(theta_1[0][0])
	alp1=[alp1]*len(theta_1[0])	
	dt0=element_multiply(alp1,dt0)
		
	theta_1[1]=element_subtract(theta_1[1],dt1)

	theta_1[0]=element_subtract(theta_1[0],dt0)

	theta_3[0]=element_add(scalar_multiply(theta_3[0],(1-tau)),scalar_multiply(theta_1[0],tau))
	theta_3[1]=element_add(scalar_multiply(theta_3[1],(1-tau)),scalar_multiply(theta_1[1],tau))
	theta_4[0]=element_add(scalar_multiply(theta_4[0],(1-tau)),scalar_multiply(theta_2[0],tau))
	theta_4[1]=element_add(scalar_multiply(theta_4[1],(1-tau)),scalar_multiply(theta_2[1],tau))
	theta_4[2]=element_add(scalar_multiply(theta_4[2],(1-tau)),scalar_multiply(theta_2[2],tau))

