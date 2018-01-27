import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1 / (1 + np.exp(-z));
    return s;


def initializeWithZeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    w = np.zeros((dim, 1));
    b = 0;
    assert (w.shape == (dim, 1));
    assert (isinstance(b, float) or isinstance(b, int));
    return w, b;


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient

    Arguments:
    w -- weights, a numpy array of size (n, 1)
    b -- bias, a scalar
    X -- data of size (n, m)
    Y -- true "label" vector

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """

    m = X.shape[1];

    # FP
    A = sigmoid(np.dot(w.T, X) + b);  # compute activation
    cost = (-1) / m * (np.dot(np.log(A), Y.T) + np.dot(np.log(1 - A), (1 - Y).T));  # compute cost

    # BP
    dw = 1 / m * np.dot(X, (A - Y).T);
    db = 1 / m * np.sum(A - Y);

    assert (dw.shape == w.shape);
    assert (db.dtype == float);
    cost = np.squeeze(cost);
    assert (cost.shape == ());
    grads = {"dw": dw,
             "db": db};

    return grads, cost;


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (n, 1)
    b -- bias, a scalar
    X -- data of shape (n, m)
    Y -- true "label" vector
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """

    costs = [];

    for i in range(num_iterations):

        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y);

        # Retrieve derivatives from grads
        dw = grads["dw"];
        db = grads["db"];

        # update rule
        w -= learning_rate * dw;
        b -= learning_rate * db;

        # Record the costs
        if i % 100 == 0:
            costs.append(cost);

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost));

    params = {"w": w,
              "b": b};

    grads = {"dw": dw,
             "db": db};

    return params, grads, costs;


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (n, 1)
    b -- bias, a scalar
    X -- data of size (n, m)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1];
    Y_prediction = np.zeros((1, m));
    w = w.reshape(X.shape[0], 1);
    A = sigmoid(np.dot(w.T, X) + b);  # activation
    Y_prediction[A > 0.5] = 1;
    Y_prediction[A <= 0.5] = 0;
    assert (Y_prediction.shape == (1, m));

    return Y_prediction;


def simpleModel(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost):
    """
    Builds a logistic regression model with no cv set

    Arguments:
    X_train -- training set represented by a numpy array of shape (n, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (n, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    res -- dictionary containing information about the trained model.
    """

    n = X_train.shape[0];
    # initialize parameters with zeros
    w, b = initializeWithZeros(n);

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost);

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"];
    b = parameters["b"];

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test);
    Y_prediction_train = predict(w, b, X_train);

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100));
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100));

    res = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations};

    return res;