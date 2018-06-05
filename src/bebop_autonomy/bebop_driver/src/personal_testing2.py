###personal testing 2(lstm bias prediction)

#importing all the needed libraries
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
%matplotlib inline


#read the csv files and store the data points into the numpy array
my_data = genfromtxt('room2_z_biases.csv', delimiter=',')
print len(my_data)
print my_data
print my_data[0]


#try to scale the data points between -1 and 1
scaler = StandardScaler()
scaled_data = scaler.fit_transform(my_data.reshape(-1, 1))
## Plot the scaled data
plt.figure(figsize=(12,7), frameon=False, facecolor='brown', edgecolor='blue')
plt.title('Scaled  between 1,-1 X_axis bias from room2 data')
plt.xlabel('Timestamp')
plt.ylabel('Scaled value of biases')
plt.plot(my_data, label='Scaled bias values')
plt.legend()
plt.show()
type(scaled_data)



# print scaled_data
aaa = scaled_data[1:3]
print aaa
aaa_data = scaler.fit_transform(aaa.reshape(-1, 1))



### Define a function that prepares all the data for trainig/testing
def window_data(all_data, window_size):
    '''
    This network takes all the data points and divides into windows of 500 and then stores the corresponding
    501st datapoint as the true label and it does so in sliding window manner.
    
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
    while (i + window_size) <= len(all_data) - 1:
        #this will append till i+window_size-1
        X.append(all_data[i:i+window_size])# x.append(a) adds the row vector "a" in the next column
        #this will append the  i+window_size element(which is the GT of prediction) for the current network
        y.append(all_data[i+window_size])#
        
        i += 1
    assert len(X) ==  len(y)
    return X, y



## Pass all the available data to the above function to format it in windowed form
X, y = window_data(scaled_data, 500)## giving the data to the above function with the window size to be 500



#splitting the data into training and the points
X_train  = np.array(X[:7000])#it willl give 0 to 6999
y_train = np.array(y[:7000])

X_test = np.array(X[7000:])# from 7000 to end
y_test = np.array(y[7000:])

print("X_train size: {}".format(X_train.shape))
print("y_train size: {}".format(y_train.shape))
print("X_test size: {}".format(X_test.shape))
print("y_test size: {}".format(y_test.shape))



## Defining the network parameters
batch_size = 3 #how many windows of data we are passing at once(i.e. using in one epoch)
# batch_size = 1 #how many windows of data we are passing at once(i.e. using in one epoch)
window_size = 500 #how big window_size is (Or How many days do we consider to predict next point in the sequence)
hidden_layer = 256 #How many units do we use in LSTM cell
clip_margin = 4 #To prevent exploding gradient, we use clipper to clip gradients below -margin or above this margin
learning_rate = 0.001 
epochs = 50 



## defining the placeholders for training inputs(batch_size x window_size)
inputs = tf.placeholder(tf.float32, [batch_size, window_size, 1])
targets = tf.placeholder(tf.float32, [batch_size, 1])


## Define the placeholders for testing inputs
# inputs_test = tf.placeholder(tf.float32, [1, window_size, 1])
# targets_test = tf.placeholder(tf.float32, [1, 1])



# LSTM weights sizes for the lstm cell to be declared
# Weights for the input gate(all the biases are initialized to be zero) 
# initializing all the weights from a normal distribution
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



#Define the output layer weights
weights_output = tf.Variable(tf.truncated_normal([hidden_layer, 1], stddev=0.05))
bias_output_layer = tf.Variable(tf.zeros([1]))



# Define an LSTM cell using the weights defined above
# Accepts an input(passed on to every hidden unit LSTM), prev_out(256 vec) and prev_state(256 vec) and 
# calculates current_output(256 Vector), current_state(256 Vector) 
def LSTM_cell(input, output, state):
    
    input_gate = tf.sigmoid(tf.matmul(input, weights_input_gate) + tf.matmul(output, weights_input_hidden) + bias_input)
    
    forget_gate = tf.sigmoid(tf.matmul(input, weights_forget_gate) + tf.matmul(output, weights_forget_hidden) + bias_forget)
    
    output_gate = tf.sigmoid(tf.matmul(input, weights_output_gate) + tf.matmul(output, weights_output_hidden) + bias_output)
    
    memory_cell = tf.tanh(tf.matmul(input, weights_memory_cell) + tf.matmul(output, weights_memory_cell_hidden) + bias_memory_cell)
    
    state = state * forget_gate + input_gate * memory_cell
    
    output = output_gate * tf.tanh(state)
    return state, output



#Define how the outputs is computed
#(loop through the network for each input data point and calculate the state and output of the lstm cell) 
outputs = []
for i in range(batch_size): #Iterates through every window in the batch
    #for each window of the batch I am creating batch_state as all zeros and output for that window which is all zeros at
    #the beginning as well.
    batch_state = np.zeros([1, hidden_layer], dtype=np.float32) 
    batch_output = np.zeros([1, hidden_layer], dtype=np.float32)
    #for each point in the window we are feeding that into LSTM to get next output
    for ii in range(window_size):
        # the original syntax passed the otuput and state in opposite order
        #batch_state, batch_output = LSTM_cell(tf.reshape(inputs[i][ii], (-1, 1)), batch_state, batch_output)
        batch_state, batch_output = LSTM_cell(tf.reshape(inputs[i][ii], (-1, 1)),batch_output, batch_state)
    # last output is conisdered and used to get a prediction(the output of the lstm cell is used to get the final output)
    # appending output for one window of the batch(taking the 256 vec from the hidden layer output for last input of the 
    # window and converting it to prediction)
    outputs.append(tf.matmul(batch_output, weights_output) + bias_output_layer)

#print output for the entire batch
outputs



#Define the loss function(define the equation to calculate the losses corresponding to the output)
losses = []
## calculating the loss vector corresponding to the outputs for one batch(i.e. 7)
for i in range(len(outputs)):
    ## mean squared error (targets(GT)-output) losses:batch loss(vec) ## Here there is only one term in the mse(may be multiple in many cases)
    losses.append(tf.losses.mean_squared_error(tf.reshape(targets[i], (-1, 1)), outputs[i]))
    #the loss is taken as the mean of the batch loss
loss = tf.reduce_mean(losses)



# Define the optimizer(now the gradient is computed according to the mean loss)
gradients = tf.gradients(loss, tf.trainable_variables())
clipped, _ = tf.clip_by_global_norm(gradients, clip_margin)
optimizer = tf.train.AdamOptimizer(learning_rate)
trained_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))




###session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
session = tf.Session()
# saver = tf.train.import_meta_graph('file.meta')
# saver.restore(session, './checkpoint')
# session.run(tf.global_variables_initializer())
# with tf.Session() as session:
#   new_saver = tf.train.import_meta_graph('file.meta')
saver.restore(session, tf.train.latest_checkpoint('./'))



# ## code for saving the data for graph visualization
# writer = tf.summary.FileWriter("/home/rrc/gk_tensorflow/tesla-stocks-prediction")
# writer.add_graph(session.graph)



# session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# session.run(tf.global_variables_initializer())
# ## code for graph visualization
# writer = tf.summary.FileWriter("/home/rrc/gk_tensorflow/tesla-stocks-prediction")
# writer.add_graph(session.graph)



# ## start the trainig
# for i in range(epochs):# going the epochs number of times through the whole dataset
#     traind_scores = []
#     ii = 0
#     epoch_loss = []
#     while(ii + batch_size) <= len(X_train): # going through the whole dataset to 
#         X_batch = X_train[ii:ii+batch_size]
#         y_batch = y_train[ii:ii+batch_size]
        
#         o, c, _ = session.run([outputs, loss, trained_optimizer], feed_dict={inputs:X_batch, targets:y_batch})
        
#         epoch_loss.append(c)
#         traind_scores.append(o)
#         ii += batch_size
#     #if (i % 30) == 0:
#         print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))



# # save the tarined network
# saver = tf.train.Saver()
# saver.save(session, "./file")


# #save the training history
# sup =[]
# for i in range(len(traind_scores)):
#     for j in range(len(traind_scores[i])):
#         sup.append(traind_scores[i][j][0])



#Write the testing code with augmenting inputs for this prediction with the outputs of the prevoius predictions
tests = []
i = 0
j = 0

#Append the prediction to the data and prepere the data for prediction
scaled_data_vec = scaled_data[:503]
window_data_vec = []
batch_data_vec = []
# data_vec.append(scaled_data[:window_size+3])
type(scaled_data_vec)
print scaled_data



# batch_data_vec = []
# for l in range(0,3):
#     for m in range(l,500+l):
#         window_data_vec.append(scaled_data_vec[m])
#     batch_data_vec.append(window_data_vec[:])
# #     print len(window_data_vec)
#     window_data_vec = []
    
for l in range(0,3):
    a0=np.array(scaled_data_vec[0:500])
    a1=np.array(scaled_data_vec[1:501])
    a2=np.array(scaled_data_vec[2:502])
    
    batch_data_vec = np.vstack(([a0], [a1]))
    batch_data_vec = np.vstack((batch_data_vec, [a2]))
    
#     batch_data_vec.append(window_data_vec[:])
#     print len(window_data_vec)
#     window_data_vec = []
    
# print len(batch_data_vec)
# type(batch_data_vec)
print type(batch_data_vec)
print len(batch_data_vec[0])
# a0
print scaled_data_vec



o = session.run(outputs, feed_dict={inputs:batch_data_vec})


# append the predicted data to the original vector and feed it for prediction
i=0
while i+batch_size <= len(X_test):
    print "The loop number:"+str(i)
#     batch_data_vec = []
    for l in range(0,3):
        a0=np.array(scaled_data_vec[l+i:500+l+i])
        a1=np.array(scaled_data_vec[l+i+1:501+l+i])
        a2=np.array(scaled_data_vec[l+i+2:502+l+i])

        batch_data_vec = np.vstack(([a0],[a1]))
        batch_data_vec = np.vstack((batch_data_vec, [a2]))

        
#         batch_data_vec.append(np.array(scaled_data_vec[i+l:500+i+l]))
#         print len(window_data_vec)
    
    o = session.run(outputs, feed_dict={inputs:batch_data_vec})
#     tupled_data_vec[500+i][0]= o[0][0][0]
#     tupled_data_vec[500+i+1][0]= o[1][0][0]
#     tupled_data_vec[500+i+2][0]= o[2][0]
    scaled_data_vec = np.vstack((scaled_data_vec,o[2][0]))
#     tupled_data_vec.append(o[2][0][0])
#     batch_data_vec = []
#     print len(tupled_data_vec)    
#     scaled_data_vec.append(o[0][0])
#     scaled_data_vec.append(o[1][0])
#     scaled_data_vec.append(o[2][0])
    i=i+1



plt.figure(figsize=(16, 7))
plt.plot(scaled_data[7500:], label='Original data')
# plt.plot(sup, label='Training data')
plt.plot(scaled_data_vec, label='Testing data')
# plt.plot(tests_new, label='Testing data')
plt.legend()
plt.show()


session.close()