################################################################################
# lib.py contained all the function needed to run Analytic_filtering,
# Filtering_function, Histogram_computation, Test_distribution & Test_parralel
#
# Contact : Sullivan MARAFICO                sullivan.marafico@u-psud.fr
################################################################################
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import special, interpolate, integrate, misc, stats
from threading import Thread
import multiprocessing as mp


special.seterr(all="ignore")
def TestPDF(total, precision):
    '''Test if total(sum of PDF) is smaller or bigger than 1 +- 2*precision
    If yes, it shows a warning with the values of total'''

    if total < 1-2*precision :
        print("/!\ WARNING : Sum of probabilities < 1-2*precision (cf. TestPDF)")
        print("/!\ 1-precision = " +str(1-precision)+ "   | Sum = " +str(total))
        return False

    if total > 1+2*precision :
        print("/!\ WARNING : Sum of probabilities > 1 (cf. TestPDF)")
        print("/!\ 1-precision = " +str(1-precision)+ "   | Sum = " +str(total))
        return False

    return True


def ComputePDF(llVV):
    '''Give the lambda(python) function of the PDF of the poissonian
    distribution for a given lambda(value, llVV)'''

    return  lambda xx : pow(llVV,xx) * np.exp(-llVV) / special.gamma(xx+1)


def Log_parameters(mu, sig):
    '''Return the parameters used in stats.lognorm (scipy)'''

    sigma = sig
    loc   = 0
    scale = np.exp(mu)

    return sigma, loc, scale


def boundaries_log(precision, mu, sig):
    '''Return the boundaries for the integral to be equal to 1-precision'''

    sigma, loc, scale = Log_parameters(mu, sig)

    return (stats.lognorm.ppf(precision, sigma, loc, scale)).astype(int), (stats.lognorm.ppf(1-precision, sigma, loc, scale) + 1).astype(int)


def Log_norm(x_value, lambdaV, dLambda):
    '''return the value of log-normale for x = x_value
    and  sigma = dLambda & mean-value = exp(lambdaV)'''

    sigma, loc, scale = Log_parameters(lambdaV, dLambda)

    return stats.lognorm.pdf(x_value, sigma, loc, scale)


def plot_log_norm(precision, lambdaV, dLambda):
    '''Plot Log_norm of parameters  (lambdaV, dlambda) for values between
    precision --> 1-precision'''

    sigma, loc, scale = Log_parameters(lambdaV, dLambda)

    x_min, x_max = boundaries_log(precision, lambdaV, sigma)
    x = np.linspace(x_min, x_max, 100)
    pdf_result = Log_norm(x, lambdaV, dLambda)
    f = interpolate.interp1d(x,pdf_result)
    res  = integrate.quad(f, x[0], x[-1])[0]
    plt.plot(x, pdf_result, label="Log-normal distribution")
    plt.ylabel('PDF', fontsize=20)
    plt.xlabel("$\lambda$", fontsize=20)
    plt.legend(prop={'size': 10})
    print("Integral of log_norm = " + str(res))


def ComputePDF_times_logN(xx, mu, dLambda):
    '''Give the lambda(python) function of the PDF of the poissonian
    distribution times the log-normal function for a given k'''

    return  lambda llVV : pow(llVV,xx) * np.exp(-llVV) / special.gamma(xx+1) * Log_norm(llVV, mu, dLambda)


def ComputeCDF_1D(result):
    '''Compute the CDF of a discrete PDF'''

    mysum = np.cumsum(result)

    return mysum


def FindMostLikely_1D(x, result):
    '''Find the most likely value for each PDF in
    an array of PDF (numpy 2D array)'''

    jmax = np.argmax(result)
    x_max = x[jmax]

    return x_max


def GetMedian_1D(x,cdf) :
    '''Get the median from a CDF and x values (2 numpy 1D array)'''
    if len(cdf) > 1 :

        f = interpolate.interp1d(cdf, x)

        if 0.5 < cdf[0] :
            x_result = 0
        else:
            x_result = f(0.5)

    else:
        x_result = x[0]


    return x_result


def GetAverageDiscrete_1D(x, pdf, precision = 0.01):
    '''Get the mean value of an array with discrete PDF & x values (2 numpy 1D array)'''

    TestPDF(sum(pdf), precision)

    return sum(x*pdf)


def Sum_weighted_random_poisson(lambdaV, omega, nombre_rand = 10000):
    '''Get nombre_rand random value for each lambdaV from a poisson distribution
    wich are weighted by omega and sum the result '''

    result  = [np.random.poisson(lambdaV[i], nombre_rand)*omega[i] for i in range(len(lambdaV))]
    res_tot = np.sum(result, axis=0)

    return res_tot


def Concatenate_weighted_random_poisson(lambdaV, omega, nombre_rand = 10000):
    '''Get nombre_rand random value for each lambdaV from a poisson distribution
    wich are weighted by omega and concatenate the result '''

    result = [np.random.poisson(lambdaV[i], nombre_rand) for i in range(len(lambdaV)) ]
    res_tot = np.concatenate(result)

    return res_tot


def log_to_ln(log_lambda, dlog_lambda):
    '''Transform 2 values of log into 2 values of ln'''

    lambdaV  = np.log(10)*log_lambda   #:
    dlambdaV = np.log(10)*dlog_lambda  #:

    return lambdaV, dlambdaV


def argument_ln_big_k(k, lambdaV, mu, sigma):
    '''Argument of the exponentiel of the log-n distribution for k big enough
    (stirling approximation)'''

    result = k*np.log(lambdaV/k) - lambdaV - (np.log(lambdaV) - mu)**2/(2*sigma**2) + k
    return result


def argument_ln(k, lambdaV, mu, sigma):
    '''Argument of the exponentiel of the log-n distribution for k small enough'''

    result = k*np.log(lambdaV) - lambdaV - (np.log(lambdaV) - mu)**2/(2*sigma**2)
    return result


def func_logN_times_Poisson(k, mu, sigma):
    '''Function logNormal times poisson with a lambda argument dlambdaV
    if k<150 it is the classical computation, if k>150 k factorial is computed
    using stirling approximation'''

    if k<150:
        result =  lambda lambdaV : np.exp(argument_ln(k, lambdaV, mu, sigma))/(lambdaV*sigma*np.sqrt(2*np.pi)*misc.factorial(k))
    else:
        result =  lambda lambdaV : np.exp(argument_ln_big_k(k, lambdaV, mu, sigma))/(lambdaV*sigma*2*np.pi*np.sqrt(k))
    return result


class Random_distrib(Thread):
    '''Equivalent of "random_from_distrib" for multi-threading'''

    def __init__(self, log_lambda, dlog_lambda, precision, N_points=10000):
        Thread.__init__(self)
        self.N_points = N_points
        self.precision = precision
        self.mu, self.sigma = log_to_ln(log_lambda, dlog_lambda)
        self.lambda_min, self.lambda_max = boundaries_log(precision, self.mu, self.sigma)
        self.max_x = (self.lambda_max+20).astype(int)

    def run(self):
        """Run the thread"""

        # Integral computation
        distribution = Custom_pdf(0, self.max_x)
        distribution.calcul(self.precision, self.mu, self.sigma, self.lambda_min, self.lambda_max)

        # Get random values
        self.result = distribution.rvs(size=self.N_points)


def random_from_distrib(log_lambda, dlog_lambda, precision, N_points=10000):
    '''Return N_points random values from a Custom_pdf of Log_parameters
    log_lambda, dlog_lambda'''

    if(log_lambda) > -5:
        mu, sigma = log_to_ln(log_lambda, dlog_lambda)
        lambda_min, lambda_max = boundaries_log(precision, mu, sigma)

        max_x = (lambda_max+20).astype(int)

        # Integral computation
        distribution = Custom_pdf(0, max_x)
        distribution.calcul(precision, mu, sigma, lambda_min, lambda_max)

        # Get random values
        result = distribution.rvs(size=N_points)
    else:
        result = np.zeros(N_points)
    return result


def random_from_distrib_process(q, log_lambda, dlog_lambda, precision, N_points=10000):
    '''Return N_points random values from a Custom_pdf of Log_parameters
    log_lambda, dlog_lambda for multiprocessing'''

    if(log_lambda) > -4:
        mu, sigma = log_to_ln(log_lambda, dlog_lambda)
        lambda_min, lambda_max = boundaries_log(precision, mu, sigma)

        max_x = (lambda_max+20).astype(int)

        # Integral computation
        start_time = time.time()
        distribution = Custom_pdf(0, max_x)
        distribution.calcul(precision, mu, sigma, lambda_min, lambda_max)

        # Get random values
        start_time = time.time()
        np.random.seed()
        result = distribution.rvs(size=N_points)

    else:
        result = np.zeros(N_points)

    q.put(result)


def Compute_histogram_multi_process(log_lambdaV, dlog_lambda, omega, bin_fd = True, precision = 0.01, N_points = 10000):
    '''Compute the histogram of a cluster of galaxies of parameters log_lambdaV,
    dlog_lambda, flux omega, precision of the Integral for multi-processing mode
    If bin_fd is true use the FD method for binning, if false : uses min(omega)
    as value between two bins.
    return : the histogram value & the edges (numpy histogram)  '''

    q = [mp.Queue() for i in range(len(log_lambdaV))]
    p = [mp.Process(target=random_from_distrib_process, args=(q[i], log_lambdaV[i], dlog_lambda[i], precision, N_points)) for i in range(len(q))]
    [p[i].start() for i in range(len(p))]
    result = [q[i].get()*omega[i] for i in range(len(p))]
    [p[i].join() for i in range(len(p))]

    res_tot = np.sum(result, axis=0)

    if not bin_fd:
        min_omega = np.argmin(omega)
        dx        = min(omega)
        max_bin   = max(res_tot+10*dx)

        bin       = np.arange(0,max_bin, dx)
    else:
        bin       ='fd'


    # Create the histogram and normalize it
    histogram_value, edges = np.histogram(res_tot, bins=bin)
    histogram_value = histogram_value/histogram_value.sum()
    return histogram_value, edges


def Return_histogram_multi_process(log_lambdaV, dlog_lambda, omega, bin_fd = True, precision = 0.01, N_points = 10000):
    '''Return the histogram of a cluster of galaxies of parameters log_lambdaV,
    dlog_lambda, flux omega, precision of the Integral for multi-processing mode
    If bin_fd is true use the FD method for binning, if false : uses min(omega)
    as value between two bins.
    return : the histogram value & the edges (numpy histogram)  '''

    q = [mp.Queue() for i in range(len(log_lambdaV))]
    p = [mp.Process(target=random_from_distrib_process, args=(q[i], log_lambdaV[i], dlog_lambda[i], precision, N_points)) for i in range(len(q))]
    [p[i].start() for i in range(len(p))]
    result = [q[i].get()*omega[i] for i in range(len(p))]
    [p[i].join() for i in range(len(p))]

    res_tot = np.sum(result, axis=0)

    if not bin_fd:
        min_omega = np.argmin(omega)
        dx        = min(omega)
        max_bin   = max(res_tot+10*dx)

        bin       = np.arange(0,max_bin, dx)
    else:
        bin       ='fd'
    return res_tot, bin


def Compute_histogram_multi_thread(log_lambdaV, dlog_lambda, omega, bin_fd = True, precision = 0.01, N_points = 10000):
    '''Compute the histogram of a cluster of galaxies of parameters log_lambdaV,
    dlog_lambda, flux omega, precision of the Integral
    If bin_fd is true use the FD method for binning, if false : uses min(omega)
    as value between two bins.
    return : the histogram value & the edges (numpy histogram)'''

    thread = [Random_distrib(log_lambdaV[i], dlog_lambda[i], precision, N_points) for i in range(len(log_lambdaV))]

    [thread[i].start() for i in range(len(thread))]
    [thread[i].join() for i in range(len(thread))]
    result = [thread[i].result*omega[i] for i in range(len(thread))]

    res_tot = np.sum(result, axis=0)

    if not bin_fd:
        min_omega = np.argmin(omega)
        dx        = min(omega)
        max_bin   = max(res_tot+10*dx)

        bin       = np.arange(0,max_bin, dx)
    else:
        bin       ='fd'

    # Create the histogram and normalize it
    histogram_value, edges = np.histogram(res_tot, bins=bin)
    histogram_value = histogram_value/histogram_value.sum()

    return histogram_value, edges


def Compute_histogram(log_lambdaV, dlog_lambda, omega, bin_fd = True, precision = 0.01, N_points = 10000):
    '''Compute the histogram of a cluster of galaxies of parameters log_lambdaV,
    dlog_lambda, flux omega, precision of the Integralself.
    If bin_fd is true use the FD method for binning, if false : uses min(omega)
    as value between two bins.
    return : the histogram value & the edges (numpy histogram)'''

    result  = [random_from_distrib(log_lambdaV[i], dlog_lambda[i], precision, N_points)*omega[i] for i in range(len(log_lambdaV))]

    res_tot = np.sum(result, axis=0)

    if not bin_fd:
        min_omega = np.argmin(omega)
        dx        = min(omega)
        max_bin   = max(res_tot+10*dx)

        bin       = np.arange(0,max_bin, dx)
    else:
        bin       ='fd'

    # Create the histogram and normalize it
    histogram_value, edges = np.histogram(res_tot, bins=bin)
    histogram_value = histogram_value/histogram_value.sum()

    return histogram_value, edges


def Get_expected_values(mu, sigma, omega):
    '''Get and return expected median, mode, mean values for a mu & sigma np.array'''
    expected_mean =  np.sum(omega*np.exp(mu+ sigma**2 / 2), axis = 1)

    exp_sigma_tot_carre = 1 + np.sum(omega**2 * np.exp(2*mu+sigma**2) * (np.exp(sigma**2) - 1),  axis = 1) / expected_mean**2

    expected_median  =  expected_mean / np.sqrt(exp_sigma_tot_carre)
    expected_mode    =  expected_median / exp_sigma_tot_carre

    return expected_mean, expected_median, expected_mode


class Custom_pdf(stats.rv_discrete):
    '''Custom Compound Probability Distribution (Log-Normal/Poisson)'''

    def calcul(self, precision, mu, sigma, lambda_min, lambda_max) :
        '''Calculate the pmf of the discrete distribution for a given mu & sigma'''

        # Compute k from 0 to lambda_max + 20 ()
        max_bin = (lambda_max+20).astype(int)

        # Compute the integral & calculate the time ###############
        self.resultat = [integrate.quad(func_logN_times_Poisson(i, mu, sigma), lambda_min, lambda_max)[0] for i in range(max_bin)]


        normalisation = np.sum(self.resultat)
        self.resultat = self.resultat/normalisation

        # Return an error message if sum proba bellow a threshold
        if not TestPDF(normalisation, precision):
            print("mu = " + str(mu) + "  | sigma = " +str(sigma))

        return max_bin



    def _pmf(self, k):
       '''pmf = pdf for a discrete distribution (Scipy method)'''
       k = np.array(k).astype(int)

       return self.resultat[k]
