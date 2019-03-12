import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    Compute the sigmoid of z
    :param z: A scalar or numpy array of any size
    :return: sigmoid(z)
    """
    s=1/(1+np.exp(-z))
    return s

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape(dim,1) for w and initialize b to 0
    :param dim: size of the w vector we want
    :return: w initialized vector of shape (dim,1)
             b initialized scalar (corresponds to the bias)
    """
    w=np.zeros((dim,1))
    b=0

    assert(w.shape==(dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    return w,b

def propagate(w,b,X,Y):
    """
    Implement the cost function and its gradient for propagation
    :param w: weights, a numpy array of size (num_px*num_px*3,1)
    :param b: bias, a scalar
    :param X: data of size (num_px*num_px*3,number of example)
    :param Y: true "label" vector (contain 0 or 1)
    :return: cost negative log-likelihood cost for logistic regression
             dw gradient of the loss respect to w
             db gradient of the loss respect to b
    """
    m=X.shape[1]
    A=sigmoid(np.dot(w.T,X)+b)
    cost=-np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))/m
    dw=np.dot(X,(A-Y).T)/m
    db=np.sum(A-Y)/m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)

    grads={
        "dw":dw,
        "db":db
    }
    return grads,cost

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    :param w: weights, a numpy array of size (num_px*num_px*3,1)
    :param b: bais, a scalar
    :param X: data of shape (num_px*num_px*3,number of example)
    :param Y: true "label" vector
    :param num_iterations: number of iterations of optimization loops
    :param learning_rate: learning rate of optimization algorithm
    :param print_cost: True to print the loss every 100 steps
    :return: params dictionary containing w and b
             grads dictionnary containing the gradients of w and b
             costs list of all the cost computed during the optimization
    """
    costs=[]
    for i in range(num_iterations):
        grads,cost = propagate(w,b,X,Y)
        dw=grads["dw"]
        db=grads["db"]
        w=w-learning_rate*dw
        b=b-learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i : %f" %(i,cost))

    params={
        "w":w,
        "b":b
    }
    grads={
        "dw":dw,
        "db":db
    }
    return params,grads,costs

def predict(w,b,X):
    """
    Predict whether the label is 0 or 1 using LR
    :param w: weights, a numpy array of size (num_px*num_px*3,1)
    :param b: bais, a scalar
    :param X: data of shape (num_px*num_px*3, number of example)
    :return: Y_prediction a numpy array vector containing of the predictions
    """
    m=X.shape[1]
    Y_prediction = np.zeros((1,m))
    w=w.reshape(X.shape[0],1)
    A=sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0][i]>0.5:
            Y_prediction[0][i]=1
        else:
            Y_prediction[0][i]=0

    assert(Y_prediction.shape==(1,m))
    return Y_prediction

def LR(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.5,print_cost=False):
    """
    Build LR model
    :param X_train: (num_px*num_px*3,m_train)
    :param Y_train: (1,m_train)
    :param X_test:  (num_px*num_px*3,m_train)
    :param Y_test:  (1,m_test)
    :param num_iterations: number of iterations
    :param learning_rate: learning rate of optimization algorithm
    :param print_cost: set to True to print the cost every iterations
    :return: d dictionary containing information about the model
    """
    w,b=initialize_with_zeros(X_train.shape[0])
    parameters,grads,costs=optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost=False)
    w=parameters["w"]
    b=parameters["b"]
    Y_prediction_test=predict(w,b,X_test)
    Y_prediction_train=predict(w,b,X_train)

    print("train accuracy:{}%".format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100))
    print("test accuracy:{}%".format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
    d={
        "costs":costs,
        "Y_prediction_test":Y_prediction_test,
        "Y_prediction_train":Y_prediction_train,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "num_iterations":num_iterations
    }
    return d

def LR_plot_curve(costs,learning_rate):
    """
    Plot learning curve using costs
    :param costs: costs, a numpy array
    :return: None
    """
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("learning rate="+str(learning_rate))
    plt.show()
