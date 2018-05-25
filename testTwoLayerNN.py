import numpy as np
import tensorflow as tf
import pandas as pd
import math
from gloveProject import loadWordVectors
from twoLayerNN import get_batch, chunker, forward_propagate, get_placeholders
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

glove = loadWordVectors()

# Borrowed from scikit-learn implementation of plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def test(test_set):
	test_set = test_set.drop(["useful","funny","cool", "date", "review_id", "user_id", "business_id"],axis=1)
	print len(test_set)

	smoothed_cost_list = []
	correct_class = 0
	attempts = 0

	with tf.Session() as sess:
		new_saver = tf.train.import_meta_graph('twoLayerNN-90.meta')
		new_saver.restore(sess, tf.train.latest_checkpoint('./'))
		graph = tf.get_default_graph()
		inputs, labels = get_placeholders(inputs_size = 50, labels_size = 5)
		W1 = graph.get_tensor_by_name("W1:0")
		b1 = graph.get_tensor_by_name("b1:0")
		W2 = graph.get_tensor_by_name("W2:0")
		b2 = graph.get_tensor_by_name("b2:0")
		logits = forward_propagate(inputs, W1, b1, W2, b2)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits))
		inputs_batch, labels_batch, correct_labels = get_batch(test_set)
		yhat, curr_loss = sess.run([logits, loss], feed_dict={inputs: inputs_batch.T, labels: labels_batch.T})
		preds = tf.argmax(tf.nn.softmax(yhat),axis=1)
		pred = preds.eval(session=sess)+1#convert to prediction and normalize
		correct_class += np.sum(pred==np.array(correct_labels))
		attempts += len(test_set)
		smoothed_cost_list.append(curr_loss)
		print ": Accuracy = " + str(float(correct_class)*100/attempts) + ", Smoothed Cost : "  + str(smoothed_cost_list[0])
		
		#Computes confusion matrix
		#class_names = ["1 star", "2 star", "3 star", "4 star", "5 star"]
		class_names = [1, 2, 3, 4, 5]
		confmat = confusion_matrix(correct_labels, pred, class_names)
		np.set_printoptions(precision=2)
		plt.figure()
		plot_confusion_matrix(confmat, classes=class_names, normalize=False, title='Confusion matrix, without normalization')
		plt.figure()
		plot_confusion_matrix(confmat, classes=class_names, normalize=True, title='Normalized confusion matrix')
		# Commented out to prevent showing the matrix everytime code is run
		#plt.show()

		precision = []
		recall = []
		for i in range(len(class_names)):
			pi = float(confmat[i][i]) / (np.sum(confmat, axis = 0))[i]
			ri = float(confmat[i][i]) / (np.sum(confmat, axis = 1))[i]
			precision.append(pi)
			recall.append(ri)
		print("Precision: ", precision)
		print("Recall: ", recall)

		


def predict(pred_set):
	# Fill in with basically the exact same code as above but over prediction examples
	raise Exception("Not implemented yet")


def main():
	yelp = pd.read_csv("newreviews.csv")
	#yelp_reduced = yelp.loc[0:499]
	# The train_dev set uses locations 700,000 to 710,000
	yelp_train_dev = yelp.loc[700000:709999]
	test(yelp_train_dev)
  
if __name__== "__main__":
	main()