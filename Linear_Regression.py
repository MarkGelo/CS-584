# load required library
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import numpy as np

# Assignment 1 - Linear Regression
# Package Prereqs: scikit-learn, matplotlib, numpy
# Implement what is asked at the TODO section.
# !! Please notice no library versions of linear regression are allowed.


def load_dataset():
    '''
    ** Do not modify this function. **
    Load diabetes dataset. We only use one feature and 60 instances.
    '''

    X, y = load_diabetes(return_X_y=True)
    return X[:60, 2], y[:60]


def train_test_split(X, y):
    '''
    Randomly split data into train and test set.
    Two thirds of the raw data will be the train set and one third of the raw data will be the test set.
    '''

    ######################################################################################
    # Full Mark: 10                                                                      #
    # TODO:                                                                              #
    # 1. shuffle the indices of data first.                                              #
    # (Hint: use numpy.arange and numpy.random.shuffle)                                  #
    # 2. select two thirds of the data as train set, the rest of data as test set.       #
    ######################################################################################

    instances = X.shape[0]

    rng = np.arange(X.shape[0])
    np.random.shuffle(rng) # basically shuffled indicies for random splits

    X_train = X[rng][:int((instances * 2) / 3), ] # 0 to 2/3 of X for X_train
    X_test = X[rng][int((instances * 2) / 3):, ] # 2/3 to end of X for X_test
    y_train = y[rng][:int((instances * 2) / 3), ] # 0 to 2/3 of y for y_train
    y_test = y[rng][int((instances * 2) / 3):, ] # 2/3 to end of y for y_test

    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    
    return X_train, X_test, y_train, y_test


def plot_data(X, y):
    '''
    Draw scatter plot using raw data.
    '''

    #########################################################################
    # Full Mark: 10                                                         #
    # TODO:                                                                 #
    # 1. make a scatter plot of the raw data                                #
    # 2. set title for the plot                                             #
    # 3. set label for X,y axis                                             #
    # e.g.,                                                                 #
    #https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.pyplot.scatter.html#
    #########################################################################

    plt.scatter(X, y)
    plt.xlabel("Body Mass Index (BMI)")
    plt.ylabel("Measure of disease progression one year after baseline")
    plt.title("Diabetes and BMI")

    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # return the plt object
    return plt


def cost_function(weights, X, y):
    '''
    Define the cost function.
    '''

    #########################################################################
    # Full Mark: 25                                                         #
    # TODO:                                                                 #
    # Implement the Mean Square Error function:                             #
    # https://en.wikipedia.org/wiki/Mean_squared_error#Mean                 #
    #                                                                       #
    # (Hint: Use numpy functions)                                           #
    #########################################################################

    # y = w[0] * X + w[1]

    y_pred = weights[0] * X[:, 1] + weights[1]

    # MSE(X bar) = E[(X bar - u)^2] -- using https://en.wikipedia.org/wiki/Mean_squared_error#Mean
    x_bar = np.mean(y_pred)
    u = np.mean(y)
    mse = (x_bar - u) ** 2

    # or MSE = 1/n sum (y - y_pred)^ 2
    mse_1 = np.mean((y - y_pred) ** 2)

    cost = mse_1
    
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # return cost
    return cost


def gradient_descent(weights, X, y):
    '''
    Update weights using gradient descent algorithm.
    '''

    # define your learning_rate and epoch
    lr = 0.1
    epoch = 20000
    
    # define cost
    cost_list = []
    
    # for loop
    for i in range(epoch):
        #########################################################################
        # Full Mark: 25                                                         #
        # TODO:                                                                 #
        # 1. update weights with learning rate lr                               #
        # 2. append the updated cost to cost list                               #
        # (Hint: Use numpy functions)                                           #
        #########################################################################

        # y_pred = w[0] * X + w[1]
        # E = (1/n) * SUM((y - ypred) ** 2) = (1/n) * SUM((yi - (w[0] * xi + w[1])) ** 2)
        # D_0 = -(2/n) * SUM(xi(yi - (w[0] * xi + w[1])))
        # D_1 = -(2/n) * SUM(yi - (w[0] * xi + w[1]))

        y_pred = weights[0] * X[:, 1] + weights[1]
        cost_list.append(cost_function(weights, X, y))

        n = X.shape[0]
        D_0 = -(2/n) * np.sum(X[:, 1] * (y - y_pred))
        D_1 = -(2/n) * np.sum(y - y_pred)
        weights[0] = weights[0] - (lr * D_0)
        weights[1] = weights[1] - (lr * D_1)

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        
    # return updated weights and cost list
    return weights, cost_list


def plot_iteration(cost, epoch=20000):
    '''
    Plot the cost for each iteration.
    '''

    #########################################################################
    # Full Mark: 10                                                         #
    # TODO:                                                                 #
    # 1. plot the cost for each iteration                                   #
    # 2. set title and labels for the plot                                  #
    # (Hint: Use plt.plot function to plot and range(n))                    #
    #########################################################################

    plt.plot(range(epoch), cost)
    plt.title(f"Cost vs Iterations ({epoch})")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    
    # show plot
    plt.show()


def plot_final(weights, X, y):
    '''
    Draw the simple linear regression model.
    '''

    # draw the raw data first
    model_plot = plot_data(X, y)

    #########################################################################
    # Full Mark: 10                                                         #
    # TODO:                                                                 #
    # 1. create a series of x coordinates in proper range.                  #
    # (Hint: use numpy.arange)                                              #
    # 2. calculate y coordinates:                                           #
    #                         y = w * X + b                                 #
    # 3. plot the curve and set title                                       #
    #########################################################################

    x = np.linspace(min(X), max(X))
    y_eq = weights[0] * x + weights[1]
    model_plot.plot(x, y_eq, color = 'red')
    model_plot.title("Final Diabetes Linear Regression Plot")
    model_plot.xlabel("Body Mass Index (BMI)")
    model_plot.ylabel("Measure of disease progression one year after baseline")

    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # show plot
    model_plot.show()


def print_test_error(weights, X, y_true):
    '''
    Use optimized weights to predict y, and print test error.
    '''

    #########################################################################
    # Full Mark: 10                                                         #
    # TODO:                                                                 #
    # 1. predict the target value y of X:                                   #
    #                            y = w * X + b                              #
    # 2. calculate the Mean Square Error using true y and predicted y       #
    #########################################################################

    y_pred = weights[0] * X + weights[1]
    
    # MSE(X bar) = E[(X bar - u)^2] -- using https://en.wikipedia.org/wiki/Mean_squared_error#Mean
    x_bar = np.mean(y_pred)
    u = np.mean(y_true)
    mse = (x_bar - u) ** 2

    # or MSE = 1/n sum (y - y_pred)^ 2
    mse_1 = np.mean((y_true - y_pred) ** 2)

    error = mse_1

    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # print test error
    print("Test error: %.4f" % error)
    return error


def main():
    '''
    ** Do not modify this function. **
    '''

    # Plot raw data points
    X, y = load_dataset()
    plot = plot_data(X, y)
    plot.show()

    # Split train and test set
    X = np.c_[np.ones(X.size), X]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # initialize weight
    weights = np.ones(X_train.shape[1])

    # calculate training cost
    init_cost = cost_function(weights, X_train, y_train)
    print("Initial cost: %.4f" % init_cost)

    # gradient descent to find the optimal fit
    weights, cost_list = gradient_descent(weights, X_train, y_train)

    # draw the cost change for iterations
    plot_iteration(cost_list)

    # draw the final linear model
    # it is shown as a red line, you can change the color anyway
    plot_final(weights, X_train[:, 1], y_train)

    # Print test error
    print_test_error(weights, X_test[:, 1], y_test)


if __name__ == '__main__':
    main()
