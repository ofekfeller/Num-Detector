# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 07:54:48 2020

@author: ofekf
"""
import numpy as np
import random
import json

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def vectoraized_results(j):
    e=np.zeros((10,1))
    e[j]=1.0
    return e

class CostEntropyCost:
    def fn(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    def delta( a, y):
        return a-y

class Network:
    def __init__(self,sizes, cost=CostEntropyCost):
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.cost=cost
        self.weights_initializer()
        
    def weights_initializer(self):
        self.biases=[np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights=[np.random.randn(y, x) /np.sqrt(x)
                      for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        
    
    def feedforward(self, a):
        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a
    
    def SGD(self,training_data,epochs,mini_batch_size, eta, lmbda=0.0, evaluation_data=None):
        if evaluation_data: n_test=len(evaluation_data)
        n=len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size]
                          for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if evaluation_data:
                print ("Epoch {0}:{1}/{2}"
                       .format(j,self.evaluate(evaluation_data),n_test))
            else: print ("Epoch {0} complete".format(j))
        
            
    def update_mini_batch(self, mini_batch, eta):
        
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w= self.backprop(x,y)
            nabla_b=[a+b for a,b in zip(nabla_b,delta_nabla_b)]
            nabla_w=[a+b for a,b in zip(nabla_w,delta_nabla_w)]
        self.weights=[w-(eta/len(mini_batch))*nw for w,nw in 
                      zip(self.weights,nabla_w)]
        self.biases=[b-(eta/len(mini_batch))*nb for b,nb in 
                      zip(self.biases,nabla_b)]
        
    def backprop(self,x,y):
        #ds
        nabla_w=[np.zeros(b.shape) for b in self.weights]
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        #ds
        activation=x
        activations=[x]
        zs=[]
        #ds
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)
            #ds   
        delta = (self.cost).delta(activations[-1], y)
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())
        #ds
        for i in range(2,self.num_layers):
            z=zs[-i]
            sp=sigmoid_prime(z)
            delta=np.dot(self.weights[-i+1].transpose(), delta)*sp
            nabla_b[-i]=delta
            nabla_w[-i]=np.dot(delta,activations[-i-1].transpose())
        
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        test_results=[(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    
    def total_cost(self, data, lmbda, convert=False):
        cost=0.0
        for x,y in data:
            a=self.feedforward(x)
            if convert: y=vectoraized_results(y)
            cost+=self.cost.fn(a,y)/len(data)
            cost+=0.5*lmbda*(1/len(data))*sum(np.lialg.norm(w)**2 for w in self.weights)
            return cost
            
    def save(self, filename):
        data = {"sizes": self.sizes,
        "weights": [w.tolist() for w in self.weights],
        "biases": [b.tolist() for b in self.biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
                
            
            
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
            
            

                
                
            
            
    