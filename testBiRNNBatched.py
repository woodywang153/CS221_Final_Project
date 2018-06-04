import numpy as np
import tensorflow as tf
import pandas as pd
import math
from gloveProject import loadWordVectors
#import matplotlib.pyplot as plt
import itertools
from BiRNNbatched import get_one_hot, get_all_vecs, chunker, get_batch

batch_size = 64
gloveSize = 200

def test(test_set):
    test_set = test_set.drop(["useful","funny","cool", "date", "review_id", "user_id", "business_id"],axis=1)
    print len(test_set)

    smoothed_cost_list = []
    correct_class = 0
    attempts = 0

    with tf.Session() as sess:

        new_saver = tf.train.import_meta_graph('./experiment_logs/biRNN-10.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./experiment_logs'))
        graph = tf.get_default_graph()

        summary_writer = tf.summary.FileWriter('logs', sess.graph)
        inputs = tf.get_collection('inputs_placeholder')[0]
        labels = tf.get_collection('labels_placeholder')[0]
        sequence_lengths = tf.get_collection('sequence_lengths')[0]
        keep_prob = tf.get_collection('keep_prob')[0]
        
        logits = tf.get_collection('logits')[0]
        loss = tf.get_collection('loss')[0]
    
        smoothed_cost_list = []
        correct_class = 0
        attempts = 0
        for batch in chunker(test_set, size=batch_size):
            inputs_batch, labels_batch, correct_labels, pad_until = get_batch(batch)
            seq_lens = np.array([inputs_batch[k].shape[1] for k in range(len(inputs_batch))])
            #print(inputs_batch[0].shape)
            inputs_batch = [np.concatenate((inputs_batch[j],np.zeros((1,pad_until-inputs_batch[j].shape[1],gloveSize),np.float32)), axis=1) for j in range(len(inputs_batch))]
            inputs_batch = np.concatenate(inputs_batch, axis=0)
            labels_batch = np.concatenate(labels_batch, axis=0)
            yhat, curr_loss = sess.run([logits, loss], feed_dict={inputs: inputs_batch, labels: labels_batch, sequence_lengths: seq_lens, keep_prob: 1.0})
            preds = tf.argmax(tf.nn.softmax(yhat),axis=1)
            pred = preds.eval(session=sess)+1#convert to prediction and normalize
            correct_class += np.sum(pred==np.array(correct_labels))
            attempts += min(batch_size, inputs_batch.shape[0])
            smoothed_cost_list.append(curr_loss)
        smoothed_cost = float(sum(smoothed_cost_list))/len(smoothed_cost_list)
        objectives_summary = tf.Summary()
        objectives_summary.value.add(tag='train_dev_smoothed_cost', simple_value=smoothed_cost)
        summary_writer.add_summary(objectives_summary)
        summary_writer.flush()
        print ": Accuracy = " + str(float(correct_class)*100/attempts) + ", Smoothed Cost : "  + str(smoothed_cost)
        
        #Computes confusion matrix
        #class_names = ["1 star", "2 star", "3 star", "4 star", "5 star"]
        '''class_names = [1, 2, 3, 4, 5]
        confmat = confusion_matrix(correct_labels, pred, class_names)
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(confmat, classes=class_names, normalize=False, title='Confusion matrix, without normalization')
        plt.figure()
        plot_confusion_matrix(confmat, classes=class_names, normalize=True, title='Normalized confusion matrix')'''
        # Commented out to prevent showing the matrix everytime code is run
        #plt.show()

        '''precision = []
        recall = []
        for i in range(len(class_names)):
            pi = float(confmat[i][i]) / (np.sum(confmat, axis = 0))[i]
            ri = float(confmat[i][i]) / (np.sum(confmat, axis = 1))[i]
            precision.append(pi)
            recall.append(ri)
        print("Precision: ", precision)
        #print("Recall: ", recall)'''


def main():
    yelp = pd.read_csv("newreviews.csv")
    #yelp_reduced = yelp.loc[0:499]
    # The train_dev set uses locations 700,000 to 710,000
    #yelp_train_dev = yelp.loc[700000:709999]
    yelp_train_dev = yelp.loc[700000:709999]
    test(yelp_train_dev)
  
if __name__== "__main__":
    main()