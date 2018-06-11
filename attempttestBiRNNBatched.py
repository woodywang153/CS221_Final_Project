import numpy as np
import tensorflow as tf
import pandas as pd
import math
from gloveProject import loadWordVectors
import matplotlib.pyplot as plt
import itertools
from BiRNNbatched import get_one_hot, get_all_vecs, chunker
from sklearn.metrics import confusion_matrix

batch_size = 64
gloveSize = 200

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

def get_batch(df_batch):
    inputs_batch = []
    OH_labels_batch = []
    correct_labels_batch = []
    restaurant_id = []
    user_id = []
    text_list = []
    longest_seq = -1
    for index, row in df_batch.iterrows():
        review_str = row['text']
        label_number = row['stars']
        #res_id = row['restaurant_id']
        #us_id = row['user_id']
        curr_label = get_one_hot(label_number)
        if type(review_str) != str:
            continue 
        curr_input = get_all_vecs(review_str.split(" "))
        if curr_input is None or len(curr_input)<=0:
            continue
        #text_list.append(review_str)
        #restaurant_id.append(res_id)
        #user_id.append(us_id)
        if len(curr_input) > longest_seq:
            longest_seq = len(curr_input)
        curr_input = np.concatenate(curr_input, axis=0)
        curr_label = np.expand_dims(curr_label.T,axis=0) 
        curr_input = np.expand_dims(curr_input, axis=0) 
        inputs_batch.append(curr_input)
        OH_labels_batch.append(curr_label)
        correct_labels_batch.append(int(label_number))
    return inputs_batch, OH_labels_batch, correct_labels_batch, longest_seq#, text_list, restaurant_id, user_id

def test(test_set):
    #test_set = test_set.drop(["useful","funny","cool", "date", "review_id", "user_id", "business_id"],axis=1)
    print len(test_set)

    smoothed_cost_list = []
    correct_class = 0
    attempts = 0

    with tf.Session() as sess:

        new_saver = tf.train.import_meta_graph('./BestBiRNN/biRNN-11.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./BestBiRNN'))
        #new_saver = tf.train.import_meta_graph('./biRNNWeights/biRNN-10.meta')
        #new_saver.restore(sess, tf.train.latest_checkpoint('./biRNNWeights'))

        graph = tf.get_default_graph()

        summary_writer = tf.summary.FileWriter('tensorboardlogs/biRNN', sess.graph)
        inputs = tf.get_collection('inputs_placeholder')[0]
        labels = tf.get_collection('labels_placeholder')[0]
        sequence_lengths = tf.get_collection('sequence_lengths')[0]
        keep_prob = tf.get_collection('keep_prob')[0]
        
        logits = tf.get_collection('logits')[0]
        loss = tf.get_collection('loss')[0]
    
        smoothed_cost_list = []
        correct_class = 0
        attempts = 0
        for batch in chunker(test_set, size=len(test_set)):
            #inputs_batch, labels_batch, correct_labels, pad_until, text_list, restaurant_id, user_id = get_batch(batch)
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
            #attempts += min(batch_size, inputs_batch.shape[0])
            attempts += min(len(test_set), inputs_batch.shape[0])
            smoothed_cost_list.append(curr_loss)
        smoothed_cost = float(sum(smoothed_cost_list))/len(smoothed_cost_list)
        objectives_summary = tf.Summary()
        objectives_summary.value.add(tag='train_dev_smoothed_cost', simple_value=smoothed_cost)
        summary_writer.add_summary(objectives_summary)
        summary_writer.flush()
        print ": Accuracy = " + str(float(correct_class)*100/attempts) + ", Smoothed Cost : "  + str(smoothed_cost)

        # Code for qualitative analysis of reviews we got wrong
        numwrong = 0
        MAXNUMWRONG = 20
        '''print(test_set.loc[21]["text"])
        print(test_set.loc[21]['stars'])
        print(test_set.loc[22]["text"])
        print(test_set.loc[22]['stars'])
        print(correct_labels[21])
        print(pred[21])
        print(len(correct_labels))
        print(len(pred))
        print(len(text_list))'''
        '''restaurant_id = np.asarray(restaurant_id).reshape(-1,1)
        print restaurant_id.shape
        user_id = np.asarray(user_id).reshape(-1,1)
        print restaurant_id.shape
        text_list = np.asarray(text_list).reshape(-1,1)
        print text_list.shape
        pred = np.asarray(pred).reshape(-1,1)
        print pred.shape
        stackedarray = np.hstack((user_id, restaurant_id, text_list, pred))
        print stackedarray.shape
        print stackedarray.T.shape
        columns = ['user_id', 'restaurant_id', 'text', 'pred']
        df = pd.DataFrame(stackedarray, columns=columns)
        df.to_csv('predictionStanford.csv', encoding = 'utf-8', index=False)
        for i in range(len(pred)):
            if correct_labels[i] != pred[i]:
                print("prediction: ", pred[i])
                print("correct label :", correct_labels[i])
                print(text_list[i])
                numwrong += 1
                if numwrong == MAXNUMWRONG:
                    break'''
        
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
        plt.show()

        precision = []
        recall = []
        for i in range(len(class_names)):
            pi = float(confmat[i][i]) / (np.sum(confmat, axis = 0))[i]
            ri = float(confmat[i][i]) / (np.sum(confmat, axis = 1))[i]
            precision.append(pi)
            recall.append(ri)
        print("Precision: ", precision)
        print("Recall: ", recall)


def main():
    #stanfordtestset = pd.read_csv('groundtruthsdata.csv')
    yelp = pd.read_csv("newreviews.csv")
    #yelp_reduced = yelp.loc[0:499]
    # The train_dev set uses locations 700,000 to 710,000
    yelp_train_dev = yelp.loc[700000:709999]
    test(yelp_train_dev)
    #nohup python attempttestBiRNNBatched.py > log.txt &
    #test(stanfordtestset)
  
if __name__== "__main__":
    main()