# ./README.md for instructions, information to run 

import random
import math
import json



# Define Constants
CONST_FLOWERS = {"Iris-setosa": [1,0,0],
            "Iris-versicolor": [0,1,0],
            "Iris-virginica":[0,0,1]}
CONST_ANSWER = {0:"Iris-setosa", 1:"Iris-setosa",2:"Iris-virginica"}
data = []


# Function: Open Json File with data
def openfile():
    with open('data.JSON') as json_data:
        d = json.load(json_data)

    for i in range(len(d)):
        data.append(list(d[i].values()))

# Function: Convert Potential to Activation 
def sigmoid(x):
  return 1 / (1 + math.exp(-x))



########################################################################
# Class: Neuron. Used for creating hidden and outer layer neurons.
# Stores weights associated with Neuron, target, error, potential and 
# output(activation) value.
########################################################################
class Neuron():
    def __init__(self, name=None):
        self.potential = None
        self.weights = []
        self.pre_nodes = []
        self.output = None
        self.name = name
        self.error = None

    def insert_error(self,error):
        self.error = error

    def insert_potential(self,potential):
        self.potential = potential

    def insert_output(self,output):
        self.output = output

    def insert_weight(self,weights):
        self.weights.append(weights)

    def insert_pre_nodes(self,pre_nodes):
        self.pre_nodes.append(pre_nodes)

    def initialize_weights(self,size):  
        for i in range(size):
            self.weights.append(random.uniform(0, 1))

    def _print(self):
        print (self.weights)

    def print_potential(self):
        print( self.potential)

    def print_name(self):
        print (self.name)



########################################################################
# Class: Input Layer Neuron. Stores values of attributes
########################################################################
class InputNeuron():

    def __init__(self, attribute, value):
        self.value = value
        self.attribute = attribute

    def _print(self):
        print( self.attribute, self.value )

    def update_value(self,value):
        self.value = value



########################################################################
# Class: Ann. Trains ANN according to attributes. Main function
# include initialize(), forward_propagation(), backward_propagation()
########################################################################
class ANN():

    def __init__(self):

        # THREE MAIN LAYERS
        self.inputLayer=[]
        self.hiddenLayer= []
        self.outerLayer=[]

        # target = the target classification
        self.target = None
        self.learningrate = 0.05

    # Function: Insert the target answer for input
    def insert_target(self,target):
        self.target = CONST_FLOWERS[target]

    # Function: to normalize the values to 0-1 by dividing by the average
    def normalize_input(self,input_values):
        total = 0
        for i in range(len(input_values)):
            total+=input_values[i]
        normalized_array = []
        for i in range(len(input_values)):
            normalized_array.append(input_values[i]/total)

        return normalized_array

    # Function: initialize input layer with Neurons on first call
    def init_input_layer(self, input_values):
        # call to normalize_input
        input_values = self.normalize_input(input_values);

        A1 = InputNeuron("A1", input_values[0])
        A2 = InputNeuron("A2", input_values[1])
        A3 = InputNeuron("A3", input_values[2])
        A4 = InputNeuron("A4", input_values[3])

        # initialize input + layer
        self.inputLayer.append(A1)
        self.inputLayer.append(A2)
        self.inputLayer.append(A3)
        self.inputLayer.append(A4)  

    # Function: initialize hidden layer with Neurons on first call
    def init_hidden_layer(self):

        # create hidden layer
        H1 = Neuron("H1")
        H2 = Neuron("H2")
        H3 = Neuron("H3")
        H4 = Neuron("H4")
        H5 = Neuron("H5")

        self.hiddenLayer.append(H1)
        self.hiddenLayer.append(H2)
        self.hiddenLayer.append(H3)
        self.hiddenLayer.append(H4)
        self.hiddenLayer.append(H5)

        # initialize weight with random values
        for i in range(5):
            self.hiddenLayer[i].initialize_weights(4)

    # Function: initialize output layer with neurons on first call
    def init_output_layer(self):

        # create neurons
        O1 = Neuron("H1") 
        O2 = Neuron("H2")
        O3 = Neuron("H3")

        self.outerLayer.append(O1)
        self.outerLayer.append(O2)
        self.outerLayer.append(O3)
    
        # initialize weight
        for i in range(len(self.outerLayer)):
            self.outerLayer[i].initialize_weights(5)


    # Function: add values to input layer
    def create_input_layer(self,input_values):
        # call to normalize_input
        input_values = self.normalize_input(input_values);
        for i in range(len(self.inputLayer)):
            self.inputLayer[i].update_value(input_values[i])


    # Function: find potential for hidden layer neurons
    def find_potential_hidden(self):
        for neuron in range(len(self.hiddenLayer)):
            potential = 0
            for i in range(len(self.hiddenLayer[neuron].weights)):
                potential = self.hiddenLayer[neuron].weights[i] * self.inputLayer[i].value

            self.hiddenLayer[neuron].insert_potential(potential)


    # Function: find output(activation) for hidden layer neurons
    # Potential -> sigmoid(potential) -> Activation
    def find_output_hidden(self):
        for neuron in range(len(self.hiddenLayer)):
            self.hiddenLayer[neuron].output = sigmoid(self.hiddenLayer[neuron].potential)   


    # Function: find potential for outer layer neurons
    def find_potential_outer(self):
        for neuron in range(len(self.outerLayer)):
            potential = 0
            for i in range(len(self.outerLayer[neuron].weights)): #
                
                potential = self.outerLayer[neuron].weights[i] * self.hiddenLayer[i].output

            self.outerLayer[neuron].insert_potential(potential)


    # Function: find output(activation) for outer layer neurons
    def find_output_outer(self):
        for neuron in range(len(self.outerLayer)):
            self.outerLayer[neuron].output = sigmoid(self.outerLayer[neuron].potential)


    # Function: for back propagation, find error for outer layer neurons
    def find_error_outer(self):
        for i in range(len(self.outerLayer)):
            x = self.outerLayer[i].output
            t = self.target[i]
            error = x * (1-x) * (t-x)
            self.outerLayer[i].error = error


    # Function: to update the weight of hidden layer to outer layer in back propagation
    def update_weight_outer(self):
        for neuron in range(len(self.outerLayer)):
            for weight in range(len(self.outerLayer[neuron].weights)):
                error = self.outerLayer[neuron].error 
                pre_weight = self.outerLayer[neuron].weights[weight]
                output = self.hiddenLayer[weight].output
                new_weight = pre_weight - (output * error)
                self.outerLayer[neuron].weights[weight] = new_weight


    # Function: to update the weight of input layer to hidden layer in back propagation
    def update_weight_hidden(self):
        learning = self.learningrate

        for neuron in range(len(self.hiddenLayer)):
            for i in range(len(self.hiddenLayer[neuron].weights)):
                pre_weight = self.hiddenLayer[neuron].weights[i]
                output = self.inputLayer[i].value
                new_weight =( pre_weight - output) * learning 
                self.hiddenLayer[neuron].weights[i] = new_weight


    # Function: Manages the initialization of the first data input
    def initialize(self, data):
        self.init_input_layer(data[:4])
        self.insert_target(data[4])
        self.init_hidden_layer()
        self.init_output_layer()
        self.forward_propagation()
        self.backward_propagation()


    # Function: calls to forward_propagation functions
    def forward_propagation(self):
        self.find_potential_hidden()
        self.find_output_hidden() #activation function
        self.find_potential_outer()
        self.find_output_outer()


    # Function: calls to backward propagation functions
    def backward_propagation(self):
        self.find_error_outer()
        self.update_weight_outer()
        self.update_weight_hidden()

    # Function: Wanted to use this see if total error is < validation error set to 0.03
    def calculate_total_error(self):
        total_error  = 0
        for i in range(len(self.outerLayer)):
            x = self.outerLayer[i].output
            t = self.target[i]
            error = x * (1-x) * (t-x)
            total_error += (error * error)

        total_error = total_error / 2

        return total_error

    # Function: Call to return the output for attributes by the gardener
    def classification(self):

        result = []
        for i in range(len(self.outerLayer)):
            out = self.outerLayer[i].output
            result.append(out)
        return CONST_ANSWER[result.index(max(result))]




########################################################################
# Main function: User output, runs ANN, and asks for input from gardener
########################################################################
if __name__ == "__main__":

    print(' \n          ***            Program:            ***              ')

    print ("\nHello! I am currently training the ANN to help classify all the Iris that got mixed up. Please give me a moment.")


    # Opens file to populate data
    openfile()
    # Creates ANN
    ann = ANN()
    # initialize the first data 
    ann.initialize(data[0])

    # pick 80% for training, 20% for validation
    training_num = int(len(data) * .8)

    # stop when validation error is < 0.03
    validation_error = float(0.03)
    error = float(100)

    while( error > validation_error):
        # training set
        for i in range(1,training_num):
            ann.create_input_layer(data[i][:4])
            ann.insert_target(data[i][4])
            ann.forward_propagation()
            ann.backward_propagation()

        # validation set
        for i in range(training_num+1, len(data)):
            ann.create_input_layer(data[i][:4])
            ann.insert_target(data[i][4])
            ann.forward_propagation()
            ann.backward_propagation()

        error = float(ann.calculate_total_error())


    print(' \n          ***        Training Complete       ***              ')

    print(' \nPlease ENTER your attributes in this format: 5.1,3.5,1.4,0.2')

    # Gets user data
    s = input()
    attributes = list(map(float, s.split(",")))
    
    ann.create_input_layer(attributes)
    ann.forward_propagation()
    classified = ann.classification()

    print(' \n          ***       Result: ' + classified + '         ***              ')
    print('\n')

















