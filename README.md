# Name: Chan Hiu Ki (Ki Ki)
# UTLN: hchan03
# Class: Comp131 AI
# Problem Set 4

----------------------------------------------------------------------
How to run code:
$ python3 ann.py


----------------------------------------------------------------------
Output:
          ***            Program:            ***              

Hello! I am currently training the ANN to help classify all the Iris that got mixed up. Please give me a moment.
 
          ***        Training Complete       ***              
 
Please ENTER your attributes in this format: 5.1,3.5,1.4,0.2


5.1,3.5,1.4,0.2
 
          ***       Result: Iris-virginica         ***      


----------------------------------------------------------------------
Architecture of Code:

Data manipulation: data is transformed to JSON in data.json, then pushed to data array in ann.py

Neuron Class:
Stores the error, activation, potential, and weights of each neuron.

Input Neuron Class:
Stores the input attribute values into the neuron.

ANN Class:
Populates the network using the "Input Neuron class" and the "Neuron class". ANN has three layers, outer, hidden, input layer. Input layer has Input Neurons, Hidden and Outer Layer has Neurons, 

The hidden layer: 5 Neurons. Each hidden Neuron is associated with 4 input neurons, so there are 4 weighted path from the input neuron layer -> hidden neuron. The weights would be [1,2,3,4].

The outer layer: 3 Neurons. Each outer Neuron is associated with 5 hidden neurons, so there are 5 weighted path from the hidden neuron layer -> outer neuron. The weights associated would be [1,2,3,4,5] 

ANN is divided into three main phases, initialize(), forward_propagation(), backward_propagation(). Initialize creates all the 3 layers, sets randomized weights (all normalized). forward_propagation() calculates the potential, hidden of the neurons, and backward_propagation() helps update the weights and change them. Once initialize is called, the data set then runs, creating new input values and updating the weights by calling forward and backward propagation. 

Each step is modular, divided into hidden Layer and outer layer. Hence, calculating the potential for hidden differs from the output, so you will find the function find_potential_hidden() and find_potential_outer(). 

How I calculated Potential:
Summation of (all weights * attributes)

How I calculated Activation:
Activation = Sigmoid(Potential)

How I calculated Error for outer:
x = self.outerLayer[i].output
t = self.target[i] => target
error = x * (1-x) * (t-x)

How I updated weight for hidden i -> outer j:
new_weight = prev_weight(i,j) - (output of i * error of j)

How I updated weight from input i  -> outer j:
new_weight = (pre_weight(i,j) - output i) * learning rate
Learning rate is set as 0.05

----------------------------------------------------------------------
Other information on training: 
All the values are normalized.
The training set stops when the validation error is lower than 3%. 





