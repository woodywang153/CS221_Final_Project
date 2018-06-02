import numpy as np
import tensorflow as tf
import pandas as pd
import math
from gloveProject import loadWordVectors

glove = loadWordVectors()
alpha = 0.01
batch_size = 1024
num_epochs = 100
lstm_units = 64
hidden_size = 64

def get_all_vecs(tokens):
	all_vecs = [] 
	for token in tokens:
		if token.lower() in glove:
			all_vecs.append(glove[token.lower()].reshape(-1,1).T)
	if all_vecs is None:
		return None 
	return all_vecs

def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res

def get_one_hot(label_num, num_classes = 5):
	if math.isnan(label_num):
		label_num = 3
	one_hot = np.zeros((num_classes,1))
	one_hot[int(label_num)-1,0] = 1
	return one_hot 

def initialize_parameters(inputs_size = 50, labels_size = 5):
	W_f = tf.get_variable(name="Wf",shape=(inputs_size, hidden_size), initializer = tf.contrib.layers.xavier_initializer())
	b_f = tf.zeros(name="bf",shape=(1, hidden_size))
	W_l = tf.get_variable(name="Wl",shape=(hidden_size, labels_size), initializer = tf.contrib.layers.xavier_initializer())
	b_l = tf.zeros(name="bl",shape=(1, labels_size))
	# W_f = tf.get_variable(name="Wf",shape=(inputs_size, hidden_size*2), initializer = tf.contrib.layers.xavier_initializer())
	# b_f = tf.zeros(name="bf",shape=(1, hidden_size*2))
	# W_l = tf.get_variable(name="Wl",shape=(hidden_size*2, labels_size), initializer = tf.contrib.layers.xavier_initializer())
	# b_l = tf.zeros(name="bl",shape=(1, labels_size))
	return W_f, b_f, W_l, b_l

def get_placeholders(inputs_size = 50, labels_size = 5):
	inputs_placeholder = tf.placeholder(tf.float32,(1, None , inputs_size))
	labels_placeholder = tf.placeholder(tf.float32,(1, None , labels_size))
	return inputs_placeholder, labels_placeholder

def forward_propagate(inputs, W_f, b_f, W_l, b_l):
	print inputs.get_shape()
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_units)
	lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
	# lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lstm_units)
	# lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
	_, states = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
	a_last = states.h
	# output_state_fw, output_state_bw = states
	# a_last_fw = output_state_fw.h
	# a_last_bw = output_state_bw.h
	# a_last = tf.concat(concat_dim=1,values=[a_last_fw, a_last_bw])
	z_out = tf.matmul(a_last, W_l) + b_l
	return z_out

def train(train_set):
	train_set = train_set.drop(["useful","funny","cool", "date", "review_id", "user_id", "business_id"],axis=1)
	print len(train_set)
	inputs, labels = get_placeholders(inputs_size = 50, labels_size = 5)
	parameters = initialize_parameters(inputs_size = 50, labels_size = 5)
	logits = forward_propagate(inputs, *parameters)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits))
	optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)
	init = tf.global_variables_initializer()


	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		for i in range(num_epochs):
			smoothed_cost_list = []
			correct_class = 0
			attempts = 0
			for _, row in train_set.iterrows():
				review_str = row['text']
				label_number = row['stars']
			 	curr_label = get_one_hot(label_number)
			 	if type(review_str) != str:
			 		continue 
			 	curr_input = get_all_vecs(review_str.split(" "))
			 	curr_input = np.concatenate(curr_input, axis=0)
			 	curr_label = np.expand_dims(curr_label.T,axis=0) 
			 	curr_input = np.expand_dims(curr_input, axis=0)
			 	#print curr_input, curr_label
			 	#print curr_input.shape, curr_label.shape
			 	if curr_input is None:
			 		continue
			 	#print curr_input.shape, curr_label.shape
				#inputs_batch, labels_batch, correct_labels = get_batch(batch)
				y_, _, curr_loss = sess.run([logits,optimizer,loss], feed_dict={inputs: curr_input, labels: curr_label})
			 	preds = tf.argmax(tf.nn.softmax(y_),axis=1)
			 	pred = preds.eval(session=sess)+1#convert to prediction and normalize
			 	correct_class += (float(pred[0])==label_number)
			 	attempts += 1
			 	smoothed_cost_list.append(curr_loss)
			smoothed_cost = float(sum(smoothed_cost_list))/len(smoothed_cost_list)
			if i % 10 == 0:
				saver.save(sess, 'basicRNN', global_step = i)
			print "Epoch " + str(i) + ": Accuracy = " + str(float(correct_class)*100/attempts) + ", Smoothed Cost : "  + str(smoothed_cost)

def main():
	train_set = [("This was delicious", 5), ("They should pay you to eat the food",1), ("Overrated yet still solid", 3), ("Good price to quality ratio",4)]
	yelp = pd.read_csv("newreviews.csv")
	yelp_reduced = yelp.loc[0:499]
	#yelp_reduced = yelp.sample(n=20000)
	train(yelp_reduced)
  
if __name__== "__main__":
	main()


