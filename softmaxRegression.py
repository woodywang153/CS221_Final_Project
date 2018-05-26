import numpy as np
import tensorflow as tf
import pandas as pd
import math
from gloveProject import loadWordVectors

glove = loadWordVectors()

def initialize_parameters(vector_dim = 50, num_classes = 5):
	w = np.random.randn(num_classes, vector_dim)
	b = np.zeros((num_classes, 1)).reshape(-1,1)
	return w, b

def softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits), axis=0)

def forward_propagate(inputs, weights, bias):
	score = weights.dot(inputs)+bias
	y_hat = softmax(score)
	pred = np.argmax(y_hat)+1
	return y_hat, pred

def get_loss(y_hat, y):
	return -1*(np.sum(y*np.log(y_hat)))

def back_propagate(weights, bias, y_hat, y, inputs, learning_rate = 0.01):
	db = y_hat-y
	dw = db.dot(inputs.T)
	new_weights = weights - learning_rate*dw
	new_bias = bias - learning_rate*db
	return new_weights, new_bias

def get_av_vec(tokens):
	av_vec = None
	for token in tokens:
		if av_vec is None and token.lower() in glove:
			av_vec = glove[token.lower()].reshape(-1,1).T
		elif token.lower() in glove:
			av_vec = np.concatenate((av_vec,glove[token.lower()].reshape(-1,1).T))
	if av_vec is None:
		return None 
	av_vec = np.mean(av_vec, axis=0, keepdims=True)
	return av_vec.T

def get_max_vec(tokens):
	max_vec = None
	for token in tokens:
		if max_vec is None and token.lower() in glove:
			max_vec = glove[token.lower()].reshape(-1,1)
		elif token.lower() in glove: 
			max_vec = np.maximum(max_vec,glove[token.lower()].reshape(-1,1))
	if max_vec is None:
		return None
	return max_vec

def get_one_hot(label_num, num_classes = 5):
	if math.isnan(label_num):
		label_num = 3
	one_hot = np.zeros((num_classes,1))
	one_hot[int(label_num)-1,0] = 1
	return one_hot 


def train(train_set):
	w, b = initialize_parameters()
	for i in range(1000):
		smoothed_cost_list = []
		correct_class = 0
		attempts = 0
		for _, row in train_set.iterrows():
			review_str = row['text']
			label_number = row['stars']
			label = get_one_hot(label_number)
			if type(review_str) != str:
				continue 
			inputs = get_av_vec(review_str.split(" ")) 
			#inputs = get_max_vec(review_str.split(" ")) 
			if inputs is None:
				continue
			y_hat, pred = forward_propagate(inputs, w, b)
			if pred == label_number:
				correct_class += 1
			attempts += 1
			curr_loss = get_loss(y_hat, label)
			smoothed_cost_list.append(curr_loss)
			w, b = back_propagate(w, b, y_hat, label, inputs)
		smoothed_cost = float(sum(smoothed_cost_list))/len(smoothed_cost_list)
		print "Epoch " + str(i) + ": Accuracy = " + str(float(correct_class)*100/attempts) + ", Smoothed Cost : "  + str(smoothed_cost)

def main():
	train_set = [("This was delicious", 5), ("They should pay you to eat the food",1), ("Overrated yet still solid", 3), ("Good price to quality ratio",4)]
	yelp = pd.read_csv("newreviews.csv")
	yelp_reduced = yelp.loc[100:199]
	#yelp_reduced = yelp.sample(n=1000)
	#train(yelp_reduced)
	train(yelp_reduced)
  
if __name__== "__main__":
	main()



#print forward_propagate(["It","was","awesome"], np.random.randn(50, 1), 0)
