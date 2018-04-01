# neuralnetbackprop.py
# uses Python version 2.7.8

import random
import math
from Normalizer import Normalizer as nm

# ------------------------------------

def show_data(matrix, num_first_rows):
  #for i in range(len(matrix)):
  for i in range(0, num_first_rows-1):
    print "[" + str(i).rjust(2) + "]",
    for j in range(len(matrix[i])):
      print str("%.1f" % matrix[i][j]).rjust(5),
    print "\n",
  print "........"
  last_row = len(matrix) - 1
  print "[" + str(last_row).rjust(2) + "]",
  for j in range(len(matrix[last_row])):
    print str("%.1f" % matrix[last_row][j]).rjust(5),
  print "\n"

#def show_vector(vector):
  #for i in range(len(vector)):
  #  if i % 125 == 0: # 9 columns
  #    print "\n",
  #  if vector[i] >= 0.0:
  #    print '',
 # summed = [0 for x in range(128)] 
  #for i in range(128):
   #   for j in range(84):
    #      summed[i] += vector[i][j]
          #print "%.2f " % vector[i][j] # 4 decimals
      #print "\n"
  #for i in range(128):
   #   print summed[i]
    #  print "\n"
# ------------------------------------

class NeuralNetwork:
  
  def __init__(self, num_input, num_hidden, num_output):
    self.num_input = num_input
    self.num_hidden = num_hidden
    self.num_output = num_output
    self.inputs = [0 for i in range(num_input)]
    self.ih_weights = self.make_matrix(num_input, num_hidden)
    self.h_biases = [0 for i in range(num_hidden)]
    self.h_outputs = [0 for i in range(num_hidden)]
    self.ho_weights = self.make_matrix(num_hidden, num_output)
    self.o_biases = [0 for i in range(num_output)]
    self.outputs = [0 for i in range(num_output)]
    # random.seed(0) # hidden function is 'normal' approach
    self.rnd = random.Random(0) # allows multiple instances
    self.initialize_weights()

  def make_matrix(self, rows, cols):
    result = [[0 for j in range(cols)] for i in range(rows)]
    return result

  def set_weights(self, weights):
    k = 0
    for i in range(self.num_input):
      for j in range(self.num_hidden):
        self.ih_weights[i][j] = weights[k]
        k += 1
    for i in range(self.num_hidden):
      self.h_biases[i] = weights[k]
      k += 1
    for i in range(self.num_hidden):
      for j in range(self.num_output):
        self.ho_weights[i][j] = weights[k]
        k += 1
    for i in range(self.num_output):
      self.o_biases[i] = weights[k]
      k += 1

  def get_weights(self):
    num_wts = ((self.num_input * self.num_hidden) + self.num_hidden +
      (self.num_hidden * self.num_output) + self.num_output)
    result = [0 for i in range(num_wts)]
    k = 0
    for i in range(self.num_input):
      for j in range(self.num_hidden):
        result[k] = self.ih_weights[i][j]
        k += 1
    for i in range(self.num_hidden):
      result[k] = self.h_biases[i]
      k += 1
    for i in range(self.num_hidden):
      for j in range(self.num_output):
        result[k] = self.ho_weights[i][j]
        k += 1
    for i in range(self.num_output):
      result[k] = self.o_biases[i]
      k += 1
    return result

  def initialize_weights(self):
    num_wts = ((self.num_input * self.num_hidden) + self.num_hidden +
      (self.num_hidden * self.num_output) + self.num_output)
    wts = [0 for i in range(num_wts)]
    lo = -0.01
    hi = 0.01
    for i in range(len(wts)):
      wts[i] = (hi - lo) * self.rnd.random() + lo
    self.set_weights(wts)

  def show_vector(self, vector):
    summed = [0 for x in range(128)]
    text_file = open("Output.txt", "w")
    for i in range(126):
      for j in range(84):
        summed[i] += self.ih_weights[i][j]
    for i in range(128):
      text_file.write("Weight: %s %s \n" % summed[i], i) 
    text_file.close()


  def compute_outputs(self, x_values):
    h_sums = [0 for i in range(self.num_hidden)]
    o_sums = [0 for i in range(self.num_output)]

    for i in range(len(x_values)):
      self.inputs[i] = x_values[i]

    for j in range(self.num_hidden):
      for i in range(self.num_input):
        h_sums[j] += (self.inputs[i] * self.ih_weights[i][j])

    for i in range(self.num_hidden):
      h_sums[i] += self.h_biases[i]

    for i in range(self.num_hidden):
      self.h_outputs[i] = self.hypertan(h_sums[i])

    for j in range(self.num_output):
      for i in range(self.num_hidden):
        o_sums[j] += (self.h_outputs[i] * self.ho_weights[i][j])

    for i in range(self.num_output):
      o_sums[i] += self.o_biases[i]

    soft_out = self.softmax(o_sums)
    for i in range(self.num_output):
      self.outputs[i] = soft_out[i]

    result = [0 for i in range(self.num_output)]
    for i in range(self.num_output):
      result[i] = self.outputs[i]
    return result
    
  def hypertan(self, x):
    if x < -20.0:
      return -1.0
    elif x > 20.0:
      return 1.0
    else:
      return math.tanh(x)

  def softmaxnaive(self, o_sums):
    div = 0
    for i in range(len(o_sums)):
      div = div + math.exp(o_sums[i])
    result = [0 for i in range(len(o_sums))]
    for i in range(len(o_sums)):
      result[i] = math.exp(o_sums[i]) / div
    return result

  def softmax(self, o_sums):
    m = max(o_sums)
    scale = 0
    for i in range(len(o_sums)):
      scale = scale + (math.exp(o_sums[i] - m))
    result = [0 for i in range(len(o_sums))]
    for i in range(len(o_sums)):
      result[i] = math.exp(o_sums[i] - m) / scale
    return result

  def train(self, train_data, max_epochs, learn_rate, momentum):#, val_data):
    o_grads = [0 for i in range(self.num_output)] # gradients
    h_grads = [0 for i in range(self.num_hidden)]
  
    ih_prev_weights_delta = self.make_matrix(num_input, num_hidden) # momentum
    h_prev_biases_delta = [0 for i in range(self.num_hidden)]
    ho_prev_weights_delta = self.make_matrix(num_hidden, num_output)
    o_prev_biases_delta = [0 for i in range(self.num_output)]

    epoch = 0
    x_values = [0 for i in range(self.num_input)]
    t_values = [0 for i in range(self.num_output)]
    sequence = [i for i in range(len(train_data))]

    while epoch < max_epochs:
      self.rnd.shuffle(sequence)
      for ii in range(len(train_data)):
        idx = sequence[ii]
        for j in range(self.num_input): # peel off x_values 
          x_values[j] = train_data[idx][j]
        for j in range(self.num_output): # peel off t_values
          t_values[j] = train_data[idx][j + self.num_input]
        self.compute_outputs(x_values) # outputs stored internally
               
        # --- update-weights (back-prop) section

        for i in range(self.num_output): # 1. compute output gradients
          derivative = (1 - self.outputs[i]) * self.outputs[i]
          o_grads[i] = derivative * (t_values[i] - self.outputs[i])
     
        for i in range(self.num_hidden): # 2. compute hidden gradients
          derivative = (1 - self.h_outputs[i]) * (1 + self.h_outputs[i])
          sum = 0
          for j in range(self.num_output):
            x = o_grads[j] * self.ho_weights[i][j]
            sum += x
          h_grads[i] = derivative * sum

        for i in range(self.num_input): # 3a. update input-hidden weights
          for j in range(self.num_hidden):
           delta = learn_rate * h_grads[j] * self.inputs[i]
           self.ih_weights[i][j] += delta
           self.ih_weights[i][j] += momentum * ih_prev_weights_delta[i][j] # momentum
           ih_prev_weights_delta[i][j] = delta # save the delta for momentum 

        for i in range(self.num_hidden): # 3b. update hidden biases
          delta = learn_rate * h_grads[i]
          self.h_biases[i] += delta
          self.h_biases[i] += momentum * h_prev_biases_delta[i]; # momentum
          h_prev_biases_delta[i] = delta # save the delta

        for i in range(self.num_hidden): # 4a. update hidden-output weights
          for j in range(self.num_output):
            delta = learn_rate * o_grads[j] * self.h_outputs[i]
            self.ho_weights[i][j] += delta
            self.ho_weights[i][j] += momentum * ho_prev_weights_delta[i][j]; # momentum
            ho_prev_weights_delta[i][j] = delta # save

        for i in range(self.num_output): # 4b. update output biases
          delta = learn_rate * o_grads[i]
          self.o_biases[i] += delta
          self.o_biases[i] += momentum * o_prev_biases_delta[i] # momentum
          o_prev_biases_delta[i] = delta # save

        # --- end update-weights
      epoch += 1
      print "Model accuracy on validation data =",
      #acc_train = self.accuracy(val_data)
      # print "%.4f" % acc_train

    result = self.get_weights()
    return result

  def accuracy(self, data):
    num_correct = 0
    num_wrong = 0
    x_values = [0 for i in range(self.num_input)]
    t_values = [0 for i in range(self.num_output)]

    for i in range(len(data)):
      for j in range(self.num_input): # peel off x_values 
        x_values[j] = data[i][j]
      for j in range(self.num_output): # peel off t_values
        t_values[j] = data[i][j + self.num_input]

      y_values = self.compute_outputs(x_values)
      max_index = y_values.index(max(y_values))

      if t_values[max_index] == 1.0:
        num_correct += 1;
      else:
        num_wrong += 1;

    return (num_correct * 1.0) / (num_correct + num_wrong)

def randomize_data(data, data_length):
    rnd = random.Random(0)
    sequence = [i for i in range(len(normalized_data))]
    rnd.shuffle(sequence)
    train_data = []
    val_data = []
    test_data = []
    train_size = (2*data_length)/3              #80% training
    test_size = (data_length - train_size)     #10% validation
    #test = data_length - train_size - val_size  #10% test
    for i in range(data_length):            #pick 80% random data to build neural network
        if i < train_size:
            train_data.append(data[sequence[i]])
        elif i > train_size:
            test_data.append(data[sequence[i]])
    return (train_data, test_data)

# ------------------------------------
print "\nBegin neural network using Python"
print "\nGoal is to predict poisinous species from 22 different characteristics \n"
print "The 8124-item raw data looks like: \n"
print "[0]  p,x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u"
print "[1]  e,x,s,y,t,a,f,c,b,k,e,c,s,s,w,w,p,w,o,p,n,n,g"
print "[2]  e,b,s,w,t,l,f,c,b,n,e,c,s,s,w,w,p,w,o,p,n,n,m"
print ". . ."
print "[8124] e,x,s,n,f,n,a,c,b,y,e,?,s,s,o,o,p,o,o,p,o,c,l"

def load_sections(filename):
    lines = open(filename).read().split('\n')
    data = []
    for line in lines:
        data.append(line.split(','))
    return data


data = load_sections("agaricus-lepiota.data") #load data
normalizer = nm(data)                         #initialize class
normalized_data = normalizer.encode()         #encode data into binary

data_size = len(normalized_data)              #sub-sampling
for i in range(4):
    train_data, test_data = randomize_data(normalized_data, data_size)
    

test = len(train_data)

print "\nCreating a 125-input, 9-hidden, 2-output neural network"
print "Using tanh and softmax activations \n"
num_input = 22
num_hidden = 15
num_output = 2
nn = NeuralNetwork(num_input, num_hidden, num_output)

max_epochs = 100    # 
learn_rate = 0.1  # 
momentum = 0.01
print "Setting max_epochs = " + str(max_epochs)
print "Setting learn_rate = " + str(learn_rate)
print "Setting momentum = " + str(momentum)

print "\nBeginning training using back-propagation"
weights = nn.train(train_data, max_epochs, learn_rate, momentum)#, val_data)
print "Training complete \n"
print "Final neural network weights and bias values:"
#nn.show_vector(weights)

print "Model accuracy on training data =",
acc_train = nn.accuracy(train_data)
print "%.4f" % acc_train

print "Model accuracy on test data     =",
acc_test = nn.accuracy(test_data)
print "%.4f" % acc_test

print "\nEnd back-prop\n"

