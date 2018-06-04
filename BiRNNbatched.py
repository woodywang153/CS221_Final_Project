import numpy as np
import tensorflow as tf
import pandas as pd
import math
from gloveProject import loadWordVectors
from testBiRNNBatch import test

glove = loadWordVectors()
alpha = 0.005
batch_size = 64
num_epochs = 50
lstm_units = 64
hidden_size = 64
gloveSize = 50

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

def initialize_parameters(inputs_size = gloveSize, labels_size = 5):
    W_f = tf.get_variable(name="Wf",shape=(inputs_size, hidden_size*2), initializer = tf.contrib.layers.xavier_initializer())
    b_f = tf.zeros(name="bf",shape=(1, hidden_size*2))
    W_l = tf.get_variable(name="Wl",shape=(hidden_size*2, labels_size), initializer = tf.contrib.layers.xavier_initializer())
    b_l = tf.zeros(name="bl",shape=(1, labels_size))
    return W_f, b_f, W_l, b_l

def get_placeholders(batch_size = None, inputs_size = gloveSize, labels_size = 5):
    inputs_placeholder = tf.placeholder(tf.float32,(batch_size, None , inputs_size), name='inputs')
    labels_placeholder = tf.placeholder(tf.float32,(batch_size, None , labels_size), name='labels')
    sequence_lengths = tf.placeholder(tf.int32,(batch_size), name='sequence_lengths')
    keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
    tf.add_to_collection('inputs_placeholder', inputs_placeholder)
    tf.add_to_collection('labels_placeholder', labels_placeholder)
    tf.add_to_collection('keep_prob', keep_prob)
    tf.add_to_collection('sequence_lengths', sequence_lengths)
    return inputs_placeholder, labels_placeholder, sequence_lengths, keep_prob

def forward_propagate(inputs, keep_prob, W_f, b_f, W_l, b_l, sequence_lengths):
    seq_len = inputs.get_shape()[2]
    lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lstm_units)
    lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_fw, output_keep_prob=keep_prob)
    lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lstm_units)
    lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_bw, output_keep_prob=keep_prob)
    _, states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs, sequence_length = sequence_lengths, dtype=tf.float32)
    output_state_fw, output_state_bw = states
    a_last_fw = output_state_fw.h
    a_last_bw = output_state_bw.h
    a_last = tf.concat(axis=1,values=[a_last_fw, a_last_bw])
    z_out = tf.matmul(a_last, W_l) + b_l
    return z_out

def chunker(seq, size=64):
    return [seq[pos:pos + size] for pos in xrange(0, len(seq), size)]

def get_batch(df_batch):
    inputs_batch = []
    OH_labels_batch = []
    correct_labels_batch = []
    longest_seq = -1
    for index, row in df_batch.iterrows():
        review_str = row['text']
        label_number = row['stars']
        curr_label = get_one_hot(label_number)
        if type(review_str) != str:
            continue 
        curr_input = get_all_vecs(review_str.split(" "))
        if curr_input is None or len(curr_input)<=1:
            continue
        if len(curr_input) > longest_seq:
            longest_seq = len(curr_input)
        curr_input = np.concatenate(curr_input, axis=0)
        curr_label = np.expand_dims(curr_label.T,axis=0) 
        curr_input = np.expand_dims(curr_input, axis=0) 
        inputs_batch.append(curr_input)
        OH_labels_batch.append(curr_label)
        correct_labels_batch.append(int(label_number))
    return inputs_batch, OH_labels_batch, correct_labels_batch, longest_seq

def train(train_set, test_set):
    tf.reset_default_graph()
    train_set = train_set.drop(["useful","funny","cool", "date", "review_id", "user_id", "business_id"],axis=1)
    print len(train_set)
    inputs, labels, sequence_lengths, keep_prob = get_placeholders(inputs_size = gloveSize, labels_size = 5)
    parameters = initialize_parameters(inputs_size = gloveSize, labels_size = 5)
    logits = forward_propagate(inputs, keep_prob, *parameters, sequence_lengths=sequence_lengths)
    tf.add_to_collection("logits",logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits))
    tf.add_to_collection("loss",loss)
    optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)
    init = tf.global_variables_initializer()


    saver = tf.train.Saver()
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('logs', sess.graph)
        sess.run(init)
        for i in range(num_epochs):
            #print i
            smoothed_cost_list = []
            correct_class = 0
            attempts = 0
            for batch in chunker(train_set, size=batch_size):
                inputs_batch, labels_batch, correct_labels, pad_until = get_batch(batch)
                seq_lens = np.array([inputs_batch[k].shape[1] for k in range(len(inputs_batch))])
                inputs_batch = [np.concatenate((inputs_batch[j],np.zeros((1,pad_until-inputs_batch[j].shape[1],gloveSize),np.float32)), axis=1) for j in range(len(inputs_batch))]
                inputs_batch = np.concatenate(inputs_batch, axis=0)
                labels_batch = np.concatenate(labels_batch, axis=0)
                y_, _, curr_loss = sess.run([logits,optimizer,loss], feed_dict={inputs: inputs_batch, labels: labels_batch, sequence_lengths: seq_lens, keep_prob: 0.75})
                preds = tf.argmax(tf.nn.softmax(y_),axis=1)
                pred = preds.eval(session=sess)+1#convert to prediction and normalize
                correct_class += np.sum(pred==np.array(correct_labels))
                attempts += min(batch_size, inputs_batch.shape[0])
                smoothed_cost_list.append(curr_loss)
            smoothed_cost = float(sum(smoothed_cost_list))/len(smoothed_cost_list)
            objectives_summary = tf.Summary()
            objectives_summary.value.add(tag='tr_smoothed_cost', simple_value=smoothed_cost)
            summary_writer.add_summary(objectives_summary)
            summary_writer.flush()
            #print str(correct_class), str(attempts)
            if (i+1) % 2 == 0:
                #print "reached"
                saver.save(sess, './experiment_logs/biRNN', global_step = (i+1))

            print "Epoch " + str(i+1) + ": Accuracy = " + str(float(correct_class)*100/attempts) + ", Smoothed Cost : "  + str(smoothed_cost)

def main():
    train_set = [("This was delicious", 5), ("They should pay you to eat the food",1), ("Overrated yet still solid", 3), ("Good price to quality ratio",4)]
    yelp = pd.read_csv("newreviews.csv")
    #yelp_reduced = yelp.loc[950:999]
    #yelp_reduced = yelp.sample(n=20000)
    yelp_reduced = yelp.loc[0:19999]
    train(yelp_reduced)
  
if __name__== "__main__":
    main()