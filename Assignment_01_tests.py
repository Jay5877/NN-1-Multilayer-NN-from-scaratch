import numpy as np
# Modify the line below based on your last name
# for example:
from Shah_01_01 import multi_layer_nn
# from Your_last_name_01_01 import multi_layer_nn

def sigmoid(x):
    # This function calculates the sigmoid function
    # x: input
    # return: sigmoid(x)
    # Your code goes here
    return 1/(1+np.exp(-x))

def create_toy_data_nonlinear(n_samples=1000):
    X = np.zeros((n_samples, 4))
    X[:, 0] = np.linspace(-1, 1, n_samples)
    X[:, 1] = np.linspace(-1, 1, n_samples)
    X[:, 2] = np.linspace(-1, 1, n_samples)
    X[:, 3] = np.linspace(-1, 1, n_samples)

    y = X[:, 0]**2 + 2*X[:, 1]  - 0.5*X[:, 2] + X[:, 3]**3 + 0.3

    # shuffle X and y
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    return X, y

def create_toy_data_nonlinear_2d(n_samples=1000):
    X = np.zeros((n_samples, 4))
    X[:, 0] = np.linspace(-1, 1, n_samples)
    X[:, 1] = np.linspace(-1, 1, n_samples)
    X[:, 2] = np.linspace(-1, 1, n_samples)
    X[:, 3] = np.linspace(-1, 1, n_samples)
    y = np.zeros((n_samples, 2))
    y[:, 0] = 0.5*X[:, 0] -0.2 * X[:, 1]**2 - 0.2*X[:, 2] + X[:, 3]*X[:,1] - 0.1
    y[:, 1] = 1.5 * X[:, 0] + 1.25 * X[:, 1]*X[:, 0] + 0.4 * X[:, 2] * X[:, 0]

    # shuffle X and y
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    return X, y

def test_can_fit_data_test():
    np.random.seed(12345)
    from sklearn.model_selection import train_test_split
    X, y = create_toy_data_nonlinear(n_samples=110)
    y = sigmoid(y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    Y_train = Y_train.reshape(-1,1)
    Y_train = Y_train
    Y_test = Y_test.reshape(-1,1)
    Y_test = Y_test

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,1],alpha=0.35,epochs=1000,h=1e-8,seed=1234)
    assert err[1] < err[0]
    assert err[2] < err[1]
    assert err[3] < err[2]
    assert err[10] < 0.15
    assert err[999] < 0.0024
    assert abs(err[9] - 0.11626994890207282) < 1e-5



def test_can_fit_data_test_2d():
    np.random.seed(1234)
    from sklearn.model_selection import train_test_split
    X, y = create_toy_data_nonlinear_2d(110)
    y = sigmoid(y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    Y_train = Y_train.reshape(-1,2)
    Y_train = Y_train
    Y_test = Y_test.reshape(-1,2)
    Y_test = Y_test

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,2],alpha=0.35,epochs=1000,h=1e-8,seed=1234)
    assert err[1] < err[0]
    assert err[2] < err[1]
    assert err[3] < err[2]
    assert err[10] < 0.5
    assert err[999] < 0.0068
    assert abs(err[9] - 0.11573813086247146) < 1e-5


def test_check_weight_init():
    np.random.seed(1234)
    from sklearn.model_selection import train_test_split
    X, y = create_toy_data_nonlinear(n_samples=110)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=1234)
    Y_train = Y_train.reshape(-1,1)
    Y_train = Y_train
    Y_test = Y_test.reshape(-1,1)
    Y_test = Y_test

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,1],alpha=0.35,epochs=0,h=1e-8,seed=1234)
    assert np.allclose(W[0], np.array([[-0.41675785, -0.05626683],
       [-2.1361961 ,  1.64027081],
       [-1.79343559, -0.84174737],
       [ 0.50288142, -1.24528809],
       [-1.05795222, -0.90900761]]))
    assert np.allclose(W[1], np.array([[-0.41675785],
       [-0.05626683],
       [-2.1361961 ]]))

def test_large_h_test():
    from sklearn.model_selection import train_test_split
    X, y = create_toy_data_nonlinear(n_samples=110)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    Y_train = Y_train.reshape(-1,1)
    Y_train = Y_train
    Y_test = Y_test.reshape(-1,1)
    Y_test = Y_test

    [W, err_1, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,1],alpha=0.35,epochs=100,h=1,seed=2)
    [W, err_2, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [2, 1], alpha=0.35,
                                   epochs=100, h=1e-8, seed=2)
    assert abs(err_1[-1] - err_1[0]) < 1e-3 or err_1[-1] > err_1[0] # with large h the error should either stay the same or may increase
    assert abs(err_2[-1] - err_2[0]) > 0.1 # with small h the error should decrease


def test_large_alpha_test():
    # if alpha is too large, the weights will change too much with each update, and the error will either increase or not improve much

    from sklearn.model_selection import train_test_split
    X, y = create_toy_data_nonlinear(n_samples=110)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    Y_train = Y_train.reshape(-1,1)
    Y_train = Y_train
    Y_test = Y_test.reshape(-1,1)
    Y_test = Y_test

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,1],alpha=10,epochs=100,h=1,seed=2)
    assert err[-1] > 0.4


def test_small_alpha_test():
    # if the alpha value is very small (e.g. 1e-9), the weights should not change much with each update, and the error should not decrease
    from sklearn.model_selection import train_test_split
    X, y = create_toy_data_nonlinear(n_samples=110)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    Y_train = Y_train.reshape(-1,1)
    Y_train = Y_train
    Y_test = Y_test.reshape(-1,1)
    Y_test = Y_test

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,1],alpha=1e-9,epochs=1000,h=1e-8,seed=2)
    assert abs(err[-1] - err[-2]) < 1e-5
    assert abs(err[1] - err[0]) < 1e-5

def test_number_of_nodes_test():
    # check if the number of nodes is being used in creating the weight matrices
    from sklearn.model_selection import train_test_split
    X, y = create_toy_data_nonlinear(n_samples=110)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    Y_train = Y_train.reshape(-1,1)
    Y_train = Y_train
    Y_test = Y_test.reshape(-1,1)
    Y_test = Y_test

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[100,1],alpha=1e-9,epochs=0,h=1e-8,seed=2)

    assert W[0].shape == (5, 100)
    assert W[1].shape == (101, 1)

    [W, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [42, 1], alpha=1e-9,
                                   epochs=0, h=1e-8, seed=2)
    assert W[0].shape == (5, 42)
    assert W[1].shape == (43, 1)

    [W, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [42, 2], alpha=1e-9,
                                   epochs=0, h=1e-8, seed=2)
    assert W[0].shape == (5, 42)
    assert W[1].shape == (43, 2)

def test_check_output_shape():
    # check if the number of nodes is being used in creating the weight matrices
    from sklearn.model_selection import train_test_split
    X, y = create_toy_data_nonlinear(n_samples=110)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    Y_train = Y_train.reshape(-1, 1)
    Y_train = Y_train
    Y_test = Y_test.reshape(-1, 1)
    Y_test = Y_test

    [W, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [100, 1], alpha=1e-9, epochs=0, h=1e-8, seed=2)
    assert Out.shape == Y_test.shape

def test_check_output_shape_2d():
    np.random.seed(1234)
    from sklearn.model_selection import train_test_split
    X, y = create_toy_data_nonlinear_2d(110)
    y = sigmoid(y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    Y_train = Y_train.reshape(-1,2)
    Y_train = Y_train
    Y_test = Y_test.reshape(-1,2)
    Y_test = Y_test

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,2],alpha=0.35,epochs=1000,h=1e-8,seed=1234)
    assert Out.shape == Y_test.shape

def test_check_output_values():
    np.random.seed(1234)
    from sklearn.model_selection import train_test_split
    X, y = create_toy_data_nonlinear_2d(110)
    y = sigmoid(y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    Y_train = Y_train.reshape(-1,2)
    Y_train = Y_train
    Y_test = Y_test.reshape(-1,2)
    Y_test = Y_test

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,2],alpha=0.35,epochs=0,h=1e-8,seed=1234)
    expected_Out = np.array([[0.01974967, 0.71039717], [0.05156214, 0.64169265], [0.15171837, 0.49588593], [0.17392495, 0.47804094],
                             [0.05403315, 0.63605476], [0.26409141, 0.44140354], [0.24360667, 0.44452713], [0.02096201, 0.71095784],
                             [0.29556927, 0.44331848], [0.05667479, 0.63019167], [0.01997007, 0.71058803]])
    assert np.allclose(Out, expected_Out, atol=1e-5)

def test_check_weight_update():
    np.random.seed(1234)
    from sklearn.model_selection import train_test_split
    X, y = create_toy_data_nonlinear_2d(110)
    y = sigmoid(y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    Y_train = Y_train.reshape(-1,2)
    Y_train = Y_train
    Y_test = Y_test.reshape(-1,2)
    Y_test = Y_test
    np.random.seed(1234)
    [W_before, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,2],alpha=0.2,epochs=0,h=1e-8,seed=1234)
    np.random.seed(1234)
    [W_after, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [2, 2], alpha=0.2, epochs=1, h=1e-8, seed=1234)
    delta1 = (W_after[0] - W_before[0])
    delta2 = (W_after[1] - W_before[1])
    correct_delta1 = np.array([[-0.00188851, -0.00329594],[ 0.00045242, -0.0021995 ],[ 0.00045242, -0.0021995 ], [ 0.00045242, -0.0021995 ], [ 0.00045242, -0.0021995 ]])
    correct_delta2 = np.array([[ 0.00783472,  0.00257216], [ 0.00174925, -0.00355675], [ 0.00291452, -0.00106518]])
    assert np.allclose(delta1, correct_delta1, atol=1e-5)
    assert np.allclose(delta2, correct_delta2, atol=1e-5)
