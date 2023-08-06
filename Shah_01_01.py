# Shah, Jay Bijal
# 1002_070_971
# 2023_02_27
# Assignment_01_01

import numpy as np
import copy

def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    weights = []
    n_dim = len(X_train)
    m = layers[0]
    for i in range(len(layers)):
        np.random.seed(seed)
        if i == 0:
            weight1 = np.random.randn(m, n_dim+1)
            weights.append(weight1)
        else:
            w = np.random.randn(layers[i], layers[i-1]+1)
            weights.append(w)

    def feedforward(weightm, input):
        
        input = np.insert(input, 0, 1, axis=0)
        layers_output = []
        layers_output.append(sigmoid(np.dot(weightm[0], input)))

        for l in range(1,len(layers)): #feedforawrd
            layers_input = layers_output[l-1]
            layers_input = np.insert(layers_input, 0, 1, axis=0)
            layers_output.append(sigmoid(np.dot(weightm[l], layers_input)))
        
        return(weightm, layers_output)

    def imse(weightm, input, train):
        input = np.insert(input, 0, 1, axis=0)
        layers_output = []
        layers_output.append(sigmoid(np.dot(weightm[0], input)))

        for l in range(1,len(layers)): #feedforawrd
            layers_input = layers_output[l-1]
            layers_input = np.insert(layers_input, 0, 1, axis=0)
            layers_output.append(sigmoid(np.dot(weightm[l], layers_input)))
        
        return(np.mean(np.square(train - layers_output[-1])))

    def pderi(weights):
        tempwp = []
        tempnp = []
        for i in range(len(weights)):
            tempwp.append(weights[i]+ h)
            tempnp.append(weights[i]- h)
            
        der3 = []
        for wi in range(len(weights)):
            der2 = []
            for i in range(0,weights[wi].shape[0]):
                der1 = []
                for j in range(0, weights[wi].shape[1]):
                    wp = copy.deepcopy(weights)
                    wn = copy.deepcopy(weights)
                    wp[wi][i][j] = copy.deepcopy(tempwp[wi][i][j])
                    wn[wi][i][j] = copy.deepcopy(tempnp[wi][i][j])
                    a = imse(wp, X_train, Y_train)
                    b = imse(wn, X_train, Y_train)
                    der = ((a - b)/(2*h))*alpha
                    w_new = weights[wi][i][j] - der
                    der1.append(w_new)
                der2.append(der1)
            der2 = np.asarray(der2)
            der3.append(der2)
        return(der3)
                    
    answer = []
    mse=[]
    if epochs == 0:
        weight , output = feedforward(weights, X_test)
        answer.append(weight)
        answer.append(mse)
        answer.append(output[-1])
    else:
        for e in range(1,epochs+1):
            w_new = pderi(weights)
            mse.append(imse(w_new, X_test, Y_test))
            weight , output = feedforward(w_new, X_test) 
            weights = copy.deepcopy(w_new)
        answer.append(w_new)
        answer.append(mse)
        answer.append(output[-1])
        
    return(answer)