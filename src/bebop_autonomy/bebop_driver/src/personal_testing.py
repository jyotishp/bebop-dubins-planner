
### personal testing1 for presicting biases using lstm

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
%matplotlib inline


x_biases = pd.read_csv('room2_z_biases.csv')
x_biases.head()#returns the first n(Default 5) rows of the caller object-DataFrame.head(n=5)


data_to_use = x_biases['-9.762'].values
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_use.reshape(-1, 1))


plt.figure(figsize=(12,7), frameon=False, facecolor='brown', edgecolor='blue')
plt.title('Scaled  between 1,-1 X_axis bias from room2 data')
plt.xlabel('Timestamp')
plt.ylabel('Scaled value of biases')
plt.plot(scaled_data, label='Scaled bias values')
plt.legend()
plt.show()


def window_data(data, window_size):
    '''
    This function is used to create Features and Labels datasets. By windowing the data.
    
    Input: data - dataset used in the project
           window_size - how many data points we are going to use to predict the next datapoint in the sequence 
                       [Example: if window_size = 1 we are going to use only the previous day to predict todays stock prices]
    
    Outputs: X - features splitted into windows of datapoints (if window_size = 1, X = [len(data)-1, 1])
             y - 'labels', actually this is the next number in the sequence, this number we are trying to predict
    
    **since in our case the period of bias change is approx 250 timestamp for TUM dataset we will test with 2 lambda as window size
    '''
    X = []
    y = []
    
    i = 0
    #with each sliding window of 500 in x  1 GT value is inserted in y(500 is counted as 1 tuple for len()) 
    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
        
        i += 1
    assert len(X) ==  len(y)
    return X, y


X, y = window_data(scaled_data, 500)## giving the data to the above function with the window size to be 500


X_train  = np.array(X[:7000])#it willl give 0 to 6999
y_train = np.array(y[:7000])

X_test = np.array(X[7000:])# from 7000 to end
y_test = np.array(y[7000:])

print("X_train size: {}".format(X_train.shape))
print("y_train size: {}".format(y_train.shape))
print("X_test size: {}".format(X_test.shape))
print("y_test size: {}".format(y_test.shape))


#Hyperparameters used in the network
batch_size = 3 #how many windows of data we are passing at once(i.e. using in one epoch)
window_size = 500 #how big window_size is (Or How many days do we consider to predict next point in the sequence)
# window_size = 100 #how big window_size is (Or How many days do we consider to predict next point in the sequence)
hidden_layer = 256 #How many units do we use in LSTM cell
clip_margin = 4 #To prevent exploding gradient, we use clipper to clip gradients below -margin or above this margin
learning_rate = 0.001 
epochs = 5 


inputs = tf.placeholder(tf.float32, [batch_size, window_size, 1])
targets = tf.placeholder(tf.float32, [batch_size, 1])


# with tf.device('/gpu:0'):
# LSTM weights
#Weights for the input gate(all the biases are initialized to be zero) 
weights_input_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05), name="in-gate")
weights_input_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05), name="in-hid-gt")
bias_input = tf.Variable(tf.zeros([hidden_layer]), name="in-bias")

#weights for the forgot gate
weights_forget_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05),name="forget")
weights_forget_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05),name="forget-hid")
bias_forget = tf.Variable(tf.zeros([hidden_layer]), name="forget-bias")

#weights for the output gate
weights_output_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05),name="out-gate")
weights_output_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05),name="out-hid")
bias_output = tf.Variable(tf.zeros([hidden_layer]),name="out-bias")

#weights for the memory cell
weights_memory_cell = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05),name="mem-cell")
weights_memory_cell_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05),name="mem-cell-hid")
bias_memory_cell = tf.Variable(tf.zeros([hidden_layer]),name="mem-cell-bias")



## Output layer weigts
weights_output = tf.Variable(tf.truncated_normal([hidden_layer, 1], stddev=0.05))
bias_output_layer = tf.Variable(tf.zeros([1]))



def LSTM_cell(input, output, state):
    
    input_gate = tf.sigmoid(tf.matmul(input, weights_input_gate) + tf.matmul(output, weights_input_hidden) + bias_input)
    
    forget_gate = tf.sigmoid(tf.matmul(input, weights_forget_gate) + tf.matmul(output, weights_forget_hidden) + bias_forget)
    
    output_gate = tf.sigmoid(tf.matmul(input, weights_output_gate) + tf.matmul(output, weights_output_hidden) + bias_output)
    
    memory_cell = tf.tanh(tf.matmul(input, weights_memory_cell) + tf.matmul(output, weights_memory_cell_hidden) + bias_memory_cell)
    
    state = state * forget_gate + input_gate * memory_cell
    
    output = output_gate * tf.tanh(state)
    return state, output



outputs = []
for i in range(batch_size): #Iterates through every window in the batch
    #for each batch I am creating batch_state as all zeros and output for that window which is all zeros at
    #the beginning as well.
    batch_state = np.zeros([1, hidden_layer], dtype=np.float32) 
    batch_output = np.zeros([1, hidden_layer], dtype=np.float32)
    #for each point in the window we are feeding that into LSTM to get next output
    for ii in range(window_size):
            ### batch_state and batch_output passed in the opposite order as received by LSTM cell func
#         batch_state, batch_output = LSTM_cell(tf.reshape(inputs[i][ii], (-1, 1)), batch_state, batch_output)
        batch_state, batch_output = LSTM_cell(tf.reshape(inputs[i][ii], (-1, 1)), batch_output, batch_state)
    #last output is conisdered and used to get a prediction
    outputs.append(tf.matmul(batch_output, weights_output) + bias_output_layer)



outputs



losses = []

for i in range(len(outputs)):
    losses.append(tf.losses.mean_squared_error(tf.reshape(targets[i], (-1, 1)), outputs[i]))
    
loss = tf.reduce_mean(losses)



gradients = tf.gradients(loss, tf.trainable_variables())
clipped, _ = tf.clip_by_global_norm(gradients, clip_margin)
optimizer = tf.train.AdamOptimizer(learning_rate)
trained_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))



session = tf.Session(config=tf.ConfigProto(log_device_placement=True))



session.run(tf.global_variables_initializer())



## code for graph visualization
writer = tf.summary.FileWriter("/home/rrc/gk_tensorflow/tesla-stocks-prediction")
writer.add_graph(session.graph)



for i in range(epochs):
    traind_scores = []
    ii = 0
    epoch_loss = []
    while(ii + batch_size) <= len(X_train):
        X_batch = X_train[ii:ii+batch_size]
        y_batch = y_train[ii:ii+batch_size]
        
        o, c, _ = session.run([outputs, loss, trained_optimizer], feed_dict={inputs:X_batch, targets:y_batch})
        
        epoch_loss.append(c)
        traind_scores.append(o)
        ii += batch_size
    #if (i % 30) == 0:
        print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))



saver = tf.train.Saver()
saver.save(session, "./file")



sup =[]
for i in range(len(traind_scores)):
    for j in range(len(traind_scores[i])):
        sup.append(traind_scores[i][j][0])


# test_output
o = session.run(outputs, feed_dict={inputs:X_test[0:batch_size]})


i=0
j=0
data_vec=[]
data_vec.append(scaled_data[7000:7503])
append_input=[]
for j in range(0,3):## as the batch size is 3
    append_input.append(data_vec[j:500+j])
# print (append_input[0])
print (append_input[0])
print X_test[0:batch_size]




####Write the testing code with augmenting inputs for this prediction with the outputs of the prevoius predictions
tests = []
i = 0
j = 0
#Append the prediction to the data and prepere the data for prediction
data_vec=[]
data_vec.append(scaled_data[7000:7503])
append_input=[]
for j in range(0,3):## as the batch size is 3
    append_input.append(data_vec[j:500+j])
    
batch_size = 3
while i+batch_size <= len(X_test):
    
    o = session.run(outputs, feed_dict={inputs:append_input})
    data_vec[500+i+1:500+i+3+1] = [o[0][0],o[1][0],o[2][0]]
    #     print(len(o))
    i += 1
    tests.append(o)

    append_input = []
    for j in range(i,i+3):
        append_input.append(data_vec[j:500+j])

print(len(tests))



tests_new = []

for i in tests:
    for j in i:
        tests_new.append(j[0])

print(len(tests_new))
# for i in range(len(tests)):
#     for j in range(len(tests[i][0])):
#         tests_new.append(tests[i][0][j])




test_results = []
# for i in range(12000):
for i in range(len(tests)):
#     if i >= 701:
#         test_results.append(tests_new[i-701])
#     else:
#         test_results.append(None)
    test_results.append(tests[i][0][0][0])
    # print tests[i][0][0][0]

print len(test_results)



plt.figure(figsize=(16, 7))
plt.plot(scaled_data[7500:], label='Original data')
# plt.plot(sup, label='Training data')
plt.plot(test_results, label='Testing data')
plt.plot(tests_new, label='Testing data')
plt.legend()
plt.show()



session.close()