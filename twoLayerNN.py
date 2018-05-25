import numpy as np
import tensorflow as tf
import pandas as pd
import math
from gloveProject import loadWordVectors
import matplotlib.pyplot as plt

glove = loadWordVectors()
#alpha = 0.01
# Try lower learning rate
alpha = 0.01
batch_size = 1024
num_epochs = 100

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

def get_one_hot(label_num, num_classes = 5):
	if math.isnan(label_num):
		label_num = 3
	one_hot = np.zeros((num_classes,1))
	one_hot[int(label_num)-1,0] = 1
	return one_hot 

def initialize_parameters(inputs_size = None, labels_size = None):
	W_1 = tf.get_variable(name="W1",shape=(inputs_size, 100), initializer = tf.contrib.layers.xavier_initializer())
	b_1 = tf.zeros(name="b1",shape=(1, 100))
	W_2 = tf.get_variable(name="W2",shape=(100, labels_size), initializer = tf.contrib.layers.xavier_initializer())
	b_2 = tf.zeros(name="b2",shape=(1, labels_size))
	return W_1, b_1, W_2, b_2

def get_placeholders(inputs_size = None, labels_size = None):
	inputs_placeholder = tf.placeholder(tf.float32,(None, inputs_size))
	labels_placeholder = tf.placeholder(tf.float32,(None, labels_size))
	return inputs_placeholder, labels_placeholder

def forward_propagate(inputs, W_1, b_1, W_2, b_2):
	z_1 = tf.matmul(inputs, W_1) + b_1
	a_1 = tf.nn.relu(z_1)
	z_2 = tf.matmul(a_1, W_2)+b_2
	return z_2

def chunker(seq, size=64):
    return [seq[pos:pos + size] for pos in xrange(0, len(seq), size)]

def get_batch(df_batch):
	inputs_batch = None
	labels_batch = None
	correct_labels_batch = []
	for index, row in df_batch.iterrows():
		if inputs_batch is None and labels_batch is None:
			review_str = row['text']
			label_number = row['stars']
			curr_label = get_one_hot(label_number)
			if type(review_str) != str:
				continue 
			curr_input = get_av_vec(review_str.split(" ")) 
			if curr_input is None:
				continue
			inputs_batch = curr_input
			labels_batch = curr_label
			correct_labels_batch.append(int(label_number))
		else:
			review_str = row['text']
			label_number = row['stars']
			curr_label = get_one_hot(label_number)
			if type(review_str) != str:
				continue 
			curr_input = get_av_vec(review_str.split(" ")) 
			if curr_input is None:
				continue
			inputs_batch = np.concatenate((inputs_batch, curr_input), axis=1)
			labels_batch = np.concatenate((labels_batch, curr_label), axis=1)
			correct_labels_batch.append(int(label_number))
	return inputs_batch, labels_batch, correct_labels_batch





def train(train_set):
	train_set = train_set.drop(["useful","funny","cool", "date", "review_id", "user_id", "business_id"],axis=1)
	print len(train_set)
	inputs, labels = get_placeholders(inputs_size = 50, labels_size = 5)
	parameters = initialize_parameters(inputs_size = 50, labels_size = 5)
	logits = forward_propagate(inputs, *parameters)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits))
	optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)
	init = tf.global_variables_initializer()

	# Used to save the model as a checkpoint, added by Woody
	saver = tf.train.Saver(max_to_keep = 3)
	costs = []


	with tf.Session() as sess:
		sess.run(init)
		for i in range(num_epochs):
			smoothed_cost_list = []
			correct_class = 0
			attempts = 0
			for batch in chunker(train_set, size=batch_size):
				inputs_batch, labels_batch, correct_labels = get_batch(batch)
				y_, _, curr_loss = sess.run([logits,optimizer,loss], feed_dict={inputs: inputs_batch.T, labels: labels_batch.T})
			 	preds = tf.argmax(tf.nn.softmax(y_),axis=1)
			 	pred = preds.eval(session=sess)+1#convert to prediction and normalize
			 	correct_class += np.sum(pred==np.array(correct_labels))
			 	attempts += batch_size
			 	smoothed_cost_list.append(curr_loss)
			 	#intermediate printing
			 	#if attempts % batch_size == 0:
			 	#	print float(sum(smoothed_cost_list))/len(smoothed_cost_list)
			smoothed_cost = float(sum(smoothed_cost_list))/len(smoothed_cost_list)
			costs.append(smoothed_cost)
			print "Epoch " + str(i) + ": Accuracy = " + str(float(correct_class)*100/attempts) + ", Smoothed Cost : "  + str(smoothed_cost)
			#if i % 10 == 0:
				#saver.save(sess, 'twoLayerNN', global_step = i)

		plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.title("Learning rate =" + str(alpha))
        plt.show()

def main():
	train_set = [("This was delicious", 5), ("They should pay you to eat the food",1), ("Overrated yet still solid", 3), ("Good price to quality ratio",4)]
	yelp = pd.read_csv("newreviews.csv")
	#yelp_reduced = yelp.loc[0:499]
	# The training set has 20,000 examples from the newreviews.csv file
	yelp_train_partition = yelp.loc[0:19999]
	train(yelp_train_partition)
	#yelp_reduced = yelp.sample(n=20000)
	#train(yelp_reduced)
  
if __name__== "__main__":
	main()

#UNBATCHED CODE
# for _, row in train_set.iterrows():
			# 	review_str = row['text']
			# 	label_number = row['stars']
			# 	curr_label = get_one_hot(label_number)
			# 	if type(review_str) != str:
			# 		continue 
			# 	curr_input = get_av_vec(review_str.split(" ")) 
			# 	if curr_input is None:
			# 		continue
			# 	y_, _, curr_loss = sess.run([logits,optimizer,loss], feed_dict={inputs: curr_input.T, labels: curr_label.T})
			# 	preds = tf.argmax(tf.nn.softmax(y_),axis=1)
			# 	pred = int(preds.eval(session=sess)[0])+1 #convert to prediction and normalize
			# 	if pred == label_number:
			# 		correct_class += 1
			# 	attempts += 1
			# 	smoothed_cost_list.append(curr_loss)
			# 	if attempts % 100 == 0:
			# 		print float(sum(smoothed_cost_list))/len(smoothed_cost_list)
			# smoothed_cost = float(sum(smoothed_cost_list))/len(smoothed_cost_list)
			# print "Epoch " + str(i) + ": Accuracy = " + str(float(correct_class)*100/attempts) + ", Smoothed Cost : "  + str(smoothed_cost)

