import numpy as np

################################
# define the class 'simulator' #
################################
'''
this class is used to define the data generating process (Y = Xb + e) for the Demonstration simulation in Section 3; Simulation 1 and 2 in Section 4.

Check this before you run the code:
Plz check if you have 'sci-kit learn', 'numpy', 'matplotlib' and 'tqdm' installed. If not,
    1. run 'pip install scikit-learn numpy matplotlib tqdm' if you use pure Python3
    2. run 'conda install scikit-learn numpy matplotlib tqdm' if you use Anaconda3

Inputs:
    1. sample_size : the total sample size we generate for cv-lars-lasso, cv-cd and solar
    2. n_dim       : the number of total variables in X
    3. n_info      : the number of informative variables

Outputs:
    1. X, Y : the inputs and output of regression
'''


class simul:

    def __init__(self, sample_size, n_dim, n_info):
        # for convinience, we define the common variable (variables we need to use for each of the following functions) in the class as follows (the common variable is defined as self.xxxx)

        self.sample_size = sample_size
        self.n_dim = n_dim
        self.n_info = n_info

    # data-generating process
    def data_gen(self):

        # 1. generating the covariance matrix for X,
        # we add a matrix full of 1/2 with an identity matrix multiplied with 1/2
        a = np.ones((self.n_dim, self.n_dim)) * 0.5
        A = np.eye(self.n_dim) * 0.5

        cov_x = a + A

        # 2. generating the mean of each column in X (which is 0)
        mean_x = np.zeros(self.n_dim)

        # 3. generating X as a multivariate Gaussian
        X = np.random.multivariate_normal(mean_x, cov_x, self.sample_size)

        # 4. generate regression coefficients in DGP as an increasing sequence (2,3,4,5,6 in our paper)
        beta_info = np.arange(2, self.n_info + 2)

        # in DGP, generate regression coefficients of redundant variables as 0
        # concatenate the regression coefficients of informative variables and redundant variables
        beta = np.concatenate(
            (beta_info, np.zeros(self.n_dim - self. n_info)), axis=0)

        # 5. generate the Gaussian random noise
        noise = np.random.normal(0, 1, self.sample_size)

        # transform Gaussian random noise into a column
        # transform regression coefficients in DGP into a row (based on the requirement of np.inner )
        noise.shape = (self.sample_size, 1)
        beta.shape = (1, self.n_dim)

        # 6. generate Y by adding random noise with the inner product of X and beta
        Y = np.inner(X, beta) + noise

        return X, Y

##################################
# test if this module works fine #
##################################


'''
this part is set up to test the functionability of the class above;
you can run all the codes in this file to test if the class works;
when you call the class from this file, the codes (function or class) after " if __name__ == '__main__': " will be ingored
'''


if __name__ == '__main__':

    sample_size = 200
    n_dim = 100
    n_info = 5

    trial = simul(sample_size, n_dim, n_info)
    X, Y = trial.data_gen()

    print(X)
    print(Y)

    # to verfiy if we generate X and Y correctly, we compute and output the OLS coefficients of Y onto all informative variables
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression(); reg.fit(X[:, 0:5], Y)
    print(reg.coef_)
