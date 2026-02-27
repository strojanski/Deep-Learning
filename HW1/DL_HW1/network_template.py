import pandas as pd
import numpy as np
import pickle

class Network(object):
    def __init__(self, sizes, optimizer="sgd", lambda_ = .01, decay=False, decay_rate=.9):
        # weights connect two layers, each neuron in layer L is connected to every neuron in layer L+1,
        # the weights for that layer of dimensions size(L+1) X size(L)
        # the bias in each layer L is connected to each neuron in L+1, the number of weights necessary for the bias
        # in layer L is therefore size(L+1).
        # The weights are initialized with a He initializer: https://arxiv.org/pdf/1502.01852v1.pdf
        self.weights = [((2/sizes[i-1])**0.5)*np.random.randn(sizes[i], sizes[i-1]) for i in range(1, len(sizes))]
        self.biases = [np.zeros((x, 1)) for x in sizes[1:]]
        self.optimizer = optimizer
        if self.optimizer == "adam":
            # First moment 
            self.mw = [np.zeros_like(w) for w in self.weights]
            self.mb = [np.zeros_like(b) for b in self.biases]
            # Second moment 
            self.vw = [np.zeros_like(w) for w in self.weights]
            self.vb = [np.zeros_like(b) for b in self.biases]
            self.t = 0

            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8

        self.lambda_ = lambda_           
        self.decay = decay
        self.decay_rate = decay_rate 
        
    def train(self, training_data,training_class, val_data, val_class, epochs, mini_batch_size, eta):
        # training data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # training_class - numpy array of dimensions [c x m], where c is the number of classes
        # epochs - number of passes over the dataset
        # mini_batch_size - number of examples the network uses to compute the gradient estimation

        iteration_index = 0
        eta_current = eta

        n = training_data.shape[1]
        for j in range(epochs):
            print("Epoch"+str(j))
            loss_avg = 0.0
            mini_batches = [
                (training_data[:,k:k + mini_batch_size], training_class[:,k:k+mini_batch_size])
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                output, Zs, As = self.forward_pass(mini_batch[0])
                gw, gb = self.backward_pass(output, mini_batch[1], Zs, As)

                self.update_network(gw, gb, eta_current)

                # Implement the learning rate schedule for Task 5
                eta_current = eta
                iteration_index += 1

                loss = cross_entropy(mini_batch[1], output)
                loss_avg += loss
                
            if self.decay:
                eta_current = eta_current * np.exp(-j * self.decay_rate)
                

            print("Epoch {} complete".format(j))
            print("Loss:" + str(loss_avg / len(mini_batches)))
            if j % 10 == 0:
                self.eval_network(val_data, val_class)



    def eval_network(self, validation_data,validation_class):
        # validation data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # validation_class - numpy array of dimensions [c x m], where c is the number of classes
        n = validation_data.shape[1]
        loss_avg = 0.0
        tp = 0.0
        for i in range(validation_data.shape[1]):
            example = np.expand_dims(validation_data[:,i],-1)
            example_class = np.expand_dims(validation_class[:,i],-1)
            example_class_num = np.argmax(validation_class[:,i], axis=0)
            output, Zs, activations = self.forward_pass(example)
            output_num = np.argmax(output, axis=0)[0]
            tp += int(example_class_num == output_num)

            loss = cross_entropy(example_class, output)
            loss_avg += loss
        print("Validation Loss:" + str(loss_avg / n))
        print("Classification accuracy: "+ str(tp/n))

    def update_network(self, gw, gb, eta):
        # gw - weight gradients - list with elements of the same shape as elements in self.weights
        # gb - bias gradients - list with elements of the same shape as elements in self.biases
        # eta - learning rate
        # SGD
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                self.weights[i] -= eta * gw[i]
                self.biases[i] -= eta * gb[i]
        elif self.optimizer == "adam":
            ########### Implement the update function for Adam:
            self.t += 1 
            for i in range(len(self.weights)):

                self.mw[i] = self.beta1 * self.mw[i] + (1 - self.beta1) * gw[i]
                self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * gb[i]

                self.vw[i] = self.beta2 * self.vw[i] + (1 - self.beta2) * gw[i]**2
                self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * gb[i]**2

                mw_hat = self.mw[i] / (1 - self.beta1**self.t)
                mb_hat = self.mb[i] / (1 - self.beta1**self.t)
                vw_hat = self.vw[i] / (1 - self.beta2**self.t)
                vb_hat = self.vb[i] / (1 - self.beta2**self.t)

                self.weights[i] -= eta * mw_hat / (np.sqrt(vw_hat) + self.epsilon)
                self.biases[i]  -= eta * mb_hat / (np.sqrt(vb_hat) + self.epsilon)

        else:
            raise ValueError('Unknown optimizer:'+self.optimizer)



    def forward_pass(self, inputs):
        Zs = []
        As = [inputs] 

        A = inputs
        for layer, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            Z = weights @ A + bias 
            Zs.append(Z)

            if layer != len(self.weights) - 1:
                A = sigmoid(Z)
            else:
                A = softmax(Z)
            As.append(A)

        output = As[-1]
        return output, Zs, As
        # input - numpy array of dimensions [n0 x m], where m is the number of examples in the mini batch and
        # n0 is the number of input attributes
        ########## Implement the forward pass
    
    def backward_pass(self, output, target, Zs, activations):
        ########## Implement the backward pass
        n_layers = len(self.weights)
        
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        
        dLdZ = softmax_dLdZ(output, target)
        
        dw[-1] = dLdZ @ activations[-2].T
        db[-1] = np.sum(dLdZ, axis=1, keepdims=True)
        
        dLdA = self.weights[-1].T @ dLdZ
        
        for l in range(n_layers - 2, -1, -1):
            dLdZ = dLdA * sigmoid_prime(Zs[l])
            
            dw[l] = dLdZ @ activations[l].T
            db[l] = np.sum(dLdZ, axis=1, keepdims=True)
            
            if l > 0:
                dLdA = self.weights[l].T @ dLdZ 
                
        # regularization
        for i in range(len(dw)):
            dw[i] += self.lambda_ * self.weights[i]
        
        return dw, db

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

def softmax_dLdZ(output, target):
    # partial derivative of the cross entropy loss w.r.t Z at the last layer
    return output - target

def cross_entropy(y_true, y_pred, epsilon=1e-12):
    targets = y_true.transpose()
    predictions = y_pred.transpose()
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def load_data_cifar(train_file, test_file):
    train_dict = unpickle(train_file)
    test_dict = unpickle(test_file)
    train_data = np.array(train_dict['data']) / 255.0
    train_class = np.array(train_dict['labels'])
    train_class_one_hot = np.zeros((train_data.shape[0], 10))
    train_class_one_hot[np.arange(train_class.shape[0]), train_class] = 1.0
    test_data = np.array(test_dict['data']) / 255.0
    test_class = np.array(test_dict['labels'])
    test_class_one_hot = np.zeros((test_class.shape[0], 10))
    test_class_one_hot[np.arange(test_class.shape[0]), test_class] = 1.0
    return train_data.transpose(), train_class_one_hot.transpose(), test_data.transpose(), test_class_one_hot.transpose()

if __name__ == "__main__":
    train_file = "./data/train_data.pckl"
    test_file = "./data/test_data.pckl"
    train_data, train_class, test_data, test_class = load_data_cifar(train_file, test_file)
    val_pct = 0.1
    val_size = int(train_data.shape[1] * val_pct)
    val_data = train_data[..., :val_size]
    val_class = train_class[..., :val_size]
    train_data = train_data[..., val_size:]
    train_class = train_class[..., val_size:]
    # The Network takes as input a list of the numbers of neurons at each layer. The first layer has to match the
    # number of input attributes from the data, and the last layer has to match the number of output classes
    # The initial settings are not even close to the optimal network architecture, try increasing the number of layers
    # and neurons and see what happens.
    net = Network([train_data.shape[0],100, 100,10], optimizer="sgd")
    net.train(train_data,train_class, val_data, val_class, 20, 64, 0.01)
    net.eval_network(test_data, test_class)


