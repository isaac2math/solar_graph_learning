from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


###############################
# define the class 'costscom' #
###############################

'''
this class is used to compute the L2 prediction error of a regression model
reg is the model you use for prediction, which could be any model (even classification)
X and Y are the data used for estimation

This class follows the paradigm of Scikit-learn. Any model in Scikit-learn has a function called '.predict()'.
'''


class costs_com:

    def __init__(self, X, Y, reg):
        # for convinience, we define the common variable (variables we need to use for each of the following functions) in the class as follows (the common variable is defined as self.xxxx)

        self.reg = reg
        self.X = X
        self.Y = Y

    # compute the mean square error of the estimator #
    def L2(self):

        Y_pred = self.reg.predict(self.X)
        Y_pred.shape = self.Y.shape
        loss = mean_squared_error(self.Y, Y_pred)
        res = self.Y - Y_pred

        return loss, res


##################################
# test if this module works fine #
##################################

# this part is set up to test the functionability of the class above
# you can run all the codes in this file to test if the class works
# when you call the class from this file
# the codes after " if __name__ == '__main__': " will be ingored

if __name__ == '__main__':

    from simulator import simul
    
    sample_size = 200
    n_dim = 100
    n_info = 5
    cov_noise = 1
    n_iter = 20
    reg = LinearRegression()

    trial1 = simul(sample_size, n_dim, n_info)
    X, Y = trial1.data_gen()

    reg.fit(X, Y)

    trial2 = costs_com(X, Y, reg)

    loss, res = trial2.L2()

    print(loss)
    print(res)
