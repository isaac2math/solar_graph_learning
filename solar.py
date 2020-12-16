import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

from sklearn.linear_model import Lars, LinearRegression, LassoLarsCV, LassoCV
from costcom import costs_com
from sklearn.exceptions import ConvergenceWarning

# For recent version of Scikit-learn: since the class 'Lars' may rely on the Cholesky decomposition and hence may have potential convergence problem in high dimensional data (p is much larger than n), we input the following commmand to skip the convergence warning.
warnings.filterwarnings(
    "ignore", category=ConvergenceWarning, module="sklearn")

#############################
# define the class of solar #
#############################

'''
this class is used to demonstrate the performance comparison among solar, cv-lars-lasso and cv-cd

if you want the result of solar only, plz define 'lasso = False'

Check this before you run the code:
Plz check if you have 'sci-kit learn', 'numpy', 'matplotlib' and 'tqdm' installed. If not,
    1. run 'pip install scikit-learn numpy matplotlib tqdm' if you use pure Python3
    2. run 'conda install sci-kit learn numpy matplotlib tqdm' if you use Anaconda3

Modules:
    1. from scikit-learn, we call 'LassoLarsCV' and 'LassoCV' for cv-lars-lasso and cv-cd respectively;
    2. from scikit-learn, we call 'Lars' to compute solar.
    3. we use 'numpy' for matrix computation and random variable generation;
    4. for simulator and costcom, plz see 'simulator.py' and 'costcom.py' for detail;
    5. we use class 'time' to time the computation of solar

Inputs:
    1. X and y     : the inputs and output of regression
    2. sample_size : the total sample size we generate for cv-lars-lasso, cv-cd and solar
    3. n_dim       : the number of total variables in X
    4. n_info      : number of informative variables in data-generating process
    5. n_repeat    : the number of subsamples in solar
    6. step_size   : step size to tune the value of c for solar;
    7. rnd_seed    : the random seed for reproduction

Outputs:
    1. solar_coef   : the solar regression coefficients (defined at the end of Algorithm 3);
    2. opt_c        : value of c* in solar;
    3. test_error   : the list of test errors for tuning the value of c;
    4. Qc_list      : the nest sets of Q(c), for all value of c from 1 to 0;
    5. la_list      : number of variables selected by CV-lars-lasso;
    6. la_vari_list : the indices of variables selected by CV-lars-lasso;
    7. cd_list      : number of variables selected by CV-cd;
    8. cd_vari_list : the indices of variables selected by CV-cd;

Remarks:
    1. In each round of subsampling, we randomly take out 10% points out of the sample and make the rest as the subsample in this round
    2. In solar, we first take 20% points from the sample as test set, the remainder of which is called X_pretrain and Y_pretrian. We use X_pretrain and Y_pretrian to estimate the average solution path (Algorithm 2 in paper); we implement validation (not CV) on test set to finalize variable selection (Algorithm 3). As competitors, we use X and Y for LassoLarsCV (called CV-lars-lasso in paper) and LassoCV (called CV-cd in paper) estimation, which relies on 10-fold CV. The rest follows algo 2 and 3 in paper.
'''


class solar:

    def __init__(self, X, Y, n_repeat, step_size, lasso=True):
        # for convinience, we define the common variable (variables we need to use for each of the following functions) in the class as follows (the common variable is defined as self.xxxx)

        self.sample_size = X.shape[0]  # sample size
        # the number of subsamples generated in solar (10 in paper)
        self.n_repeat = n_repeat
        # (grid search) step size for tuning the value of c for solar
        self.step_size = step_size
        # the size of the data reserved for average soluation path estimation (Algorithm 2)
        self.train_size = int(self.sample_size * 0.8)
        # the size of each subsample
        self.subsample_size = int(self.train_size * 0.9)
        self.n_dim = X.shape[1]  # the number of total variables in X
        self.lasso = lasso  # if lasso = true, this package will compute cv-lars-lasso and cv-cd alongside with solar for comparison

        # the maximum value of c in its grid search (for plotting)
        self.q_start = 1
        # the minimum value of c in its grid search (for plotting)
        self.q_end = 0.1
        self.q_step = -0.02  # step size of c in its grid search (for plotting)

        # the observations of X in test set
        self.X_test = X[int(-1 * self.sample_size * 0.2):, :]
        # the observations of Y in test set
        self.y_test = Y[int(-1 * self.sample_size * 0.2):, :]
        # the observations of X for average soluation path estimation
        self.X_so = X[:int(-1 * self.sample_size * 0.2), :]
        # the observations of Y for average soluation path estimation
        self.y_so = Y[:int(-1 * self.sample_size * 0.2), :]

        self.X = X
        self.y = Y  # the sample we generate via data-generating process

    def fit(self):

        # 1. construct a placeholder called 'qhat_k_container' for the list of all q_hat^k (defined in Algorithm 2) of each subsample
        qhat_k_container = list()

        # 2. estimate q_hat^k (for the solution path) on each subsample and save them as elements of the placeholder
        for j in range(self.n_repeat):

            # a. randomly choose a subset of sample points (whose index is 'index_subsample') that is used to generate a subsample in each repeat
            index_subsample = np.random.choice(
                self.train_size, self.subsample_size, replace=False)
            # b. based on 'index_subsample', take the corresponding observations of X out and save them as the subample
            X_subsample = self.X_so[index_subsample]
            # c. based on 'index_subsample', take the corresponding observations of Y out and save them as the subample
            y_subsample = self.y_so[index_subsample]

            # d. scikit-learn requires 'y_subsample' to be an one-dimension array
            y_subsample.shape = (y_subsample.shape[0],)

            # e. given a subsample, compute q_hat^k (the solution path) using lars

            # e(1). call the class 'Lars'
            trial_1 = Lars(n_nonzero_coefs=min(
                X_subsample.shape[1] + 1, X_subsample.shape[0] + 1))
            # e(2). fit lars on the subsample
            trial_1.fit(X_subsample, y_subsample)
            # e(3). save the active set of lars (indices of variables select by lars) as 'active'.
            active = trial_1.active_

            # f. The active set of lars is ranked based on the chronology of variable inclusion at different stages of lars. For example [2,1,3] means x_2 is included at stage 1, x_1 is included at stage 2 and x_3 is included at stage 3. Based on the active set of lars, we compute q_hat^k (defined as 'qhat_k' in code) as defined in Algorithm 2

            # f(1). we generate 'qhat_k' as an array of zeros;
            qhat_k = np.zeros((1, self.n_dim))
            # f(2). we compute the i-th value of q_hat^k for the corresponding variable based on Algorithm 2; replace i-th term in 'qhat_k' with the value we just compute
            for i in active:

                qhat_k[0, i] = 1 - \
                    (np.where(np.array(active) == i)[0][0]) / (self.n_dim)

            # f(3). we append the result into 'qhat_k_container' as one element of the list
            qhat_k_container.append(qhat_k)

        # 3. if self.lasso == True, we compute CV-lars-lasso and CV-cd on the original sample X and Y (not on the subsample)
        if (self.lasso == True):

            # a(1). call the class for CV-lars-lasso (called LassoLarsCV in Scikit-learn)
            # a(2). we set the number of folds in CV as 10
            trial_2 = LassoLarsCV(cv=10)
            # b. change y into one-dimensional array (required by Scikit-learn)
            yy = self.y
            yy.shape = (self.sample_size,)
            # c.  fit CV-lars-lasso on X and Y
            trial_2.fit(self.X, yy)

            # d. save 'la_list' as the number of variables in the active set of CV-lars-lasso
            la_list = len(trial_2.active_)
            # e. save 'la_vari_list' as the active set of CV-lars-lasso
            la_vari_list = trial_2.active_

            # f. call the class for CV-cd (called LassoCV in Scikit-learn)
            # f(1). we set the number of folds in CV as 10
            # f(2). for reproduction, we fix the random seed of training-validation split in CV (random_state=0)
            trial_3 = LassoCV(cv=10, random_state=0)

            # g.  fit cv-cd on X and Y
            trial_3.fit(self.X, yy)

            # h. save 'cd_list' as the number of variables in the active set of CV-cd
            cd_list = np.count_nonzero(trial_3.coef_)
            # i. save 'cd_vari_list' as the active set of CV-cd
            cd_vari_list = np.nonzero(trial_3.coef_)[0]

        # 4. compute q_hat and Q(c) (defined in Algorithm 2)
        # a(1). we transform the list of all q_hat^k ('qhat_k_container') into a matrix ('qhat_k_container_matrix')
        # a(2). row of the matrix: the q_hat^k on a given subsample for all variables
        # a(3). colum of the matrix: the corresponding value of q_hat^k for a given variable on all subsamples
        qhat_k_container_matrix = np.concatenate(qhat_k_container, axis=0)
        # b.  compute the the value of qhat for each variable (qhat defined in Algorithm 2 of the paper)
        qhat_value = np.mean(qhat_k_container_matrix, axis=0)

        # c. set 'Qc_list' as the container of Q(c) for all value of c
        Qc_list = list()
        # d. set 'c_seq' as the sequence of c for the grid search of c* in solar
        c_seq = np.arange(max(qhat_value), 0.1, self.step_size)

        # e. generate Q(c) for each value of c
        for j in c_seq:
            # e(1). define 'container' as the placeholder of Q(c) when c == j;
            container = list()

            for i in range(self.X.shape[1]):
                # e(2). include all variables into 'container' if their corresponding values in q-hat are larger or equal to j;
                if (qhat_value[i] >= j):

                    container.append(i)
            # e(3). append 'container' (Q(c) when c == j) into 'Qc_list' (the container of Q(c) for all value of c);
            Qc_list.append(container)

        # 5. compute the test error of each value of c
        # we use grid search on test set to choose c*;
        # for each value of c in the grid search, train a OLS of Y_so on the variables of Q(c) in X_so (Y_so and X_so defined at the begining);

        # a. container for test errors
        test_error = list()

        # b. compute the test error of each Q(c) on test set
        # b(0). set i as the indices of all variables in Q(c) for a given value of c;
        for i in Qc_list:
            # b(1). call the LinearRegression class;
            OLS_1 = LinearRegression()
            # b(2). compute OLS of Y_so on the variables in Q(c) in X_so;
            OLS_1.fit(self.X_so[:, i], self.y_so)
            # b(3). compute the L2 prediction error of OLS on test set (X_test, y_test);
            s1 = costs_com(self.X_test[:, i], self.y_test, OLS_1)
            loss_test_1, _ = s1.L2()
            # b(4). save the L2 error as the test error of Q(c) for each value of c; append it into the container of test errors;
            test_error.append(loss_test_1)

        # 6. tuning c via grid search
        # 6(a). transform 'test_error' from a list into an array;
        test_error = np.asarray(test_error)
        # 6(b). save the location of minimum of 'test_error' as 'min_loc_val';
        min_loc_val = np.where(test_error == min(test_error))[0]
        # 6(c). save the correpsonding value of c (c*) as 'opt_c';
        opt_c = c_seq[min_loc_val]
        # 6(d). find Q(c*) and save it as 'Q_opt_c';
        Q_opt_c = Qc_list[max(min_loc_val)]

        # 7. Regression of Y onto the selected variables ( Q(c*) ) in X
        # 7(a). call the LinearRegression class;
        OLS_2 = LinearRegression()
        # 7(b). fit OLS of Y on the variables of Q(c*) in X;
        OLS_2.fit(self.X[:, Qc_list[max(min_loc_val)]], self.y)
        # 7(c). set 'solar_coef' (an array of zeros) as the placeholder of solar regression coefficents
        solar_coef = np.zeros([self.n_dim, 1])
        # 7(d). put the estimated regression coefficents into their corresponding place of 'solar_coef'
        solar_coef[Q_opt_c, 0] = OLS_2.coef_

        # 8. define la_list, la_vari_list as empty list if self.lasso != True (if we don't want to compute cv-lars-lasso and cv-cd)
        if (self.lasso != True):

            la_list = []
            la_vari_list = []
            cd_list = []
            cd_vari_list = []

        return solar_coef, opt_c, test_error, Qc_list, la_list, la_vari_list, cd_list, cd_vari_list

    # return Q(c), for all c from (start from q_start and end at q_end)
    def q_list(self, Qc_list):

        # 1. concatenate Qc_list into a matrix
        var_mark_plot = np.concatenate(Qc_list)
        # 2. compute the value of c for each Q(c) and the corresponding variables in each Q(c)
        var_index, counts = np.unique(var_mark_plot, return_counts=True)

        var_index_ordered = [x for _, x in sorted(zip(counts, var_index))]

        var_plot = var_index_ordered[::-1]

        cou_plot = np.sort(counts)[::-1] / \
            ((self.q_end - self.q_start) / self.q_step)

        var_plot = ['X' + str(i) for i in var_plot]

        # 3. print the list of variables with different value of c

        var_loc_list = list()
        var_q_list = list()

        q_value_list = np.unique(cou_plot)[::-1]

        i = 1

        for j in q_value_list:

            ans_ind = np.where([cou_plot == j])[1]
            ans_var = [var_plot[i] for i in ans_ind]

            var_loc_list.append(ans_ind)
            var_q_list.append(ans_var)

            print('q_hat value >= ', j)
            print(var_q_list[:i])

            i += 1


##################################
# test if this module works fine #
##################################

'''
this part is set up to test the functionability of the class above;
you can run all the codes in this file to test if the class works;
when you call the class from this file, the codes (even functions or classes) after " if __name__ == '__main__': " will be ingored
'''

if __name__ == '__main__':

    from simulator import simul

    sample_size = 50
    n_dim = 100
    n_info = 5
    n_repeat = 10
    step_size = -0.02

    np.random.seed(0)

    # generate X and Y
    trial1 = simul(sample_size, n_dim, n_info)
    X, Y = trial1.data_gen()

    # start timing
    start = time.time()

    # train solar
    trial2 = solar(X, Y, n_repeat, step_size, lasso=True)

    solar_coef, opt_c, test_error, Qc_list, la_list, la_vari_list, cd_list, cd_vari_list = trial2.fit()

    # end timing
    end = time.time()

    # count how many non-zero elements in solar regression coefficents (or how many variables solar selects eventually)
    print('number of variables that solar selects: ',
          len(np.where(solar_coef != 0)[0]))

    # count ow many variables cv-lars-lasso selects
    print('number of variables that cv-lars-lasso selects: ', len(la_vari_list))

    # count ow many variables cv-cd selects
    print('number of variables that cv-cd  selects: ', len(cd_vari_list))

    # report the computation time of solar, CV-cd and CV-lars-lasso
    print('time of solar computation: ', end - start)

    trial2.q_list(Qc_list)
