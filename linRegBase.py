import numpy as np
import tensorflow as tf
import pandas as pd
from gloveProject import loadWordVectors

glove = loadWordVectors()

def initialize_parameters(vector_dim = 50):
	w = np.random.randn(vector_dim, 1).reshape(-1,1)
	b = 0.0
	return w, b

def forward_propagate(inputs, weights, bias):
	score = inputs.T.dot(weights)+bias
	pred_class = int(round(score))
	return score, pred_class

def get_loss(score, label):
	return (score-label)**2

def back_propagate(weights, bias, score, label, inputs, learning_rate = 0.001):
	dw = 2*(score - label)*inputs
	db = 2*(score - label)
	new_weights = weights - learning_rate*dw
	new_bias = bias - learning_rate*db
	return new_weights, new_bias

def get_av_vec(tokens):
	av_vec = None
	for token in tokens:
		if av_vec is None and glove[token.lower()] is not None:
			av_vec = glove[token.lower()].reshape(-1,1).T
		elif glove[token.lower()] is not None:
			av_vec = np.concatenate((av_vec,glove[token.lower()].reshape(-1,1).T)) 
	av_vec = np.mean(av_vec, axis=0, keepdims=True)
	return av_vec.T

def train(train_set):
	w, b = initialize_parameters()
	for i in range(1000):
		smoothed_cost_list = []
		correct_class = 0
		attempts = 0
		for example in train_set:
			review_str, label = example
			inputs = get_av_vec(review_str.split(" ")) 
			score, pred = forward_propagate(inputs, w, b)
			if pred == label:
				correct_class += 1
			attempts += 1
			curr_loss = get_loss(score, label)
			smoothed_cost_list.append(curr_loss)
			w, b = back_propagate(w, b, score, label, inputs)
		smoothed_cost = float(sum(smoothed_cost_list))/len(smoothed_cost_list)
		print "Gradient Descent Iteration " + str(i) + ": Accuracy = " + str(float(correct_class)*100/attempts) + ", Smoothed Cost:"  + str(smoothed_cost)

def main():
	train_set = [("This was delicious", 5), ("They should pay you to eat the food",1), ("Overrated yet still solid", 3), ("Good price to quality ratio",4)]
	train(train_set)
  
if __name__== "__main__":
	main()



#print forward_propagate(["It","was","awesome"], np.random.randn(50, 1), 0)
