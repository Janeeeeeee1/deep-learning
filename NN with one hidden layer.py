import numpy as np

def sigmoid(z):
    """
    Compute sigmoid function
    :param z: a scalar or numpy array of any size
    :return: s
    """
    s=1/(1+np.exp(-z))
    return s

def layer_sizes(X,Y):
    """
    :param X: input dataset of shape (input size,number of size)
    :param Y: labels of shape (output size, number of size)
    :return: n_x the size of input layer
             n_h the size of hidden layer
             n_y the size of the output layer
    """
    n_x=X.shape[0]
    n_h=4
    n_y=Y.shape[0]
    return (n_x,n_h,n_y)

def initialize_parameters(n_x,n_h,n_y):
    """
    :param n_x: size of input layer
    :param n_h: size of the hidden layer
    :param n_y: size of the output layer
    :return: parameters python dictionary containing your parameters
                    W1 weight matrix of shape (n_h,n_x)
                    b1 bais of shape (n_h,1)
                    W2 weight matrix of shape (n_y,n_h)
                    b2 bais of shape (n_y,1)
    """
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))

    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    return parameters

def forward_propagation(X,parameters):
    """
    Forward propagation with one hidden layer
    :param X: data of shape (n_x,number of examples)
    :param parameters: dictionary containing W1,b1,W2,b2
    :return: A2 the sigmoid output of the second layer
             cache the dictionary containing Z1,A1,Z2,A2
    """
    m=X.shape[1]
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    Z1=np.dot(W1,X)+b1
    A1=sigmoid(Z1)
    Z2=np.dot(W2,X)+b2
    A2=sigmoid(Z2)

    assert(A2.shape==(1,m))
    cache={
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2
    }
    return A2,cache

def compute_cost(A2,Y):
    """
    Compute cross-entropy cost
    :param A2: the sigmoid output of the second activation (1,number of examples)
    :param Y: the true "label"
    :return: cost
    """
    m=Y.shape[1]
    logreg=np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),1-Y)
    cost=-np.sum(logreg)/m
    cost=np.squeeze(cost)
    assert(isinstance(cost,float))
    return cost

def backward_propagation(parameters,cache,X,Y):
    """
    Implement the backward propagtion
    :param parameters: python dictionary containing W1,b1,W2,b2
    :param cache: python dictionary containing Z1,A1,Z2,A2
    :param X: input data of shape
    :param Y: true label vector of shape (1, number of examples)
    :return: grads python dictionary containing dW1,db1,dW2,db2
    """
    m=X.shape[1]
    W1=parameters["W1"]
    W2=parameters["W2"]
    A1=cache["A1"]
    A2=cache["A2"]

    dZ2=A2-Y
    dW2=np.dot(dZ2,A1.T)/m
    db2=np.sum(dZ2,axis=1,keepdims=True)/m
    dZ1=np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1=np.dot(dZ1,X.T)/m
    db1=np.sum(dZ1,axis=1,keepdims=True)/m

    grads={
        "dW1":dW1,
        "db1":db1,
        "dW2":dW2,
        "db2":db2
    }
    return grads

def update_parameters(parameters,grads,learning_rate=1.2):
    """
    Update parameters using gradient descent
    :param parameters: dictionary containing W1,b1,W2,b2
    :param grads: dictionary containing Z1,A1,Z2,A2
    :param learning_rate: learning rate of gradient descent
    :return: dictionary containing updated parameters
    """
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    dW1=grads["dW1"]
    db1=grads["db1"]
    dW2=grads["dW2"]
    db2=grads["db2"]

    W1=W1-learning_rate*dW1
    b1=b1-learning_rate*db1
    W2=W2-learning_rate*dW2
    b2=b2-learning_rate*db2

    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    return parameters

def nn_model(X,Y,n_h,num_iterations=10000,print_cost=False):
    """
    :param X: dataset
    :param Y: labels
    :param n_h: size of the hidden layer
    :param num_iterations: number of iterations
    :param print_cost: if True, print the cost every 1000 iterations
    :return: parameters learnt by the model. They can then be used to predict
    """
    n_x=layer_sizes(X,Y)[0]
    n_y=layer_sizes(X,Y)[2]
    parameters=initialize_parameters(n_x,n_h,n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
        A2,cache=forward_propagation(X,parameters)
        cost=compute_cost(A2,Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=1.2)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" %(i,cost))

    return parameters

def predict(parameters,X):
    """
    Using learnt parameters to predict a class in X
    :param parameters: dictionary containing parameters of the model
    :param X: dataset
    :return: predictions vector of predictions of the model
    """
    A2,cache=forward_propagation(X,parameters)
    predictions=A2>0.5
    return predictions

