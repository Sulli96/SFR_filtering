import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table, Column
import astropy.coordinates as Coord
import astropy.units as u

import coordinates_LS as LS

import lib
from tqdm import tqdm
import csv
import argparse
import pandas



### Utilities #####################################################
#global variable to avoid recomputing ln(10) each time
ln10 = np.log(10)

def is_number(s):
    '''Check if s is a number'''
    try:
        float(s)
        return True
    except ValueError:
        return False

def find_nearest(value, array):
    '''Find the id of the nearest value'''
    idx = (np.abs(array - value)).argmin()
    return idx


def  Gaus(x, mu, sig):
    '''Gaussian'''
    t = (x-mu)/sig
    return np.exp(-0.5*t*t)/(sig*np.sqrt(2*np.pi))

def mean_lognormal(mu_ln10, sigma_ln10):
    '''Average of log-normal variable'''
    mu = mu_ln10*ln10
    sigma = sigma_ln10*ln10
    return np.exp((mu+0.5*sigma*sigma))


def variance_lognormal_div_mean2(sigma_ln10):
    '''V/mu^2 of log-normal variable'''
    sigma = sigma_ln10*ln10

    return (np.exp(sigma*sigma)-1)

def variance_cpd(mu, sigma):

    return (np.exp(sigma*sigma)-1)  *np.exp(2*mu+sigma*sigma) + np.exp(mu+sigma*sigma/2)

def Compute_median_multi_process(q, log_lambdaV, dlog_lambdaV, precision):

    if log_lambdaV < -4:
        median = 0

    else:
        # Convert log_lambda & dlog to mu & sigma
        mu, sigma = lib.log_to_ln(log_lambdaV, dlog_lambdaV)

        # Compute boundaries
        lambda_min, lambda_max = lib.boundaries_log(precision, mu, sigma)
        k_max = (lambda_max+10).astype(int)
        k = np.linspace(0,k_max,k_max+1)

        # Create distribution
        distribution = lib.Custom_pdf(0, k_max)

        # Compute the distribution
        distribution.calcul(precision, mu, sigma, lambda_min, lambda_max)

        # Get CDF
        cdf = distribution.cdf(k)

        # Compute the estimator
        median = lib.GetMedian_1D(k, cdf)
    q.put(median)

def WriteSFR(D, SFR, filename='SFR.csv'):
    '''Save the computation to a file'''

    print("\n-->Writing mode on, creating a file " + filename)

    with open(filename, mode='w') as flux_file:
        flux_writer = csv.writer(flux_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        [flux_writer.writerow([ D[i], SFR[i] ]) for i in range(len(SFR))]


### CSFH ##########################################################
#Dmax < 100 Mpc -> only percent correction to z-D relation
def MadauDickinson_SFR(D_Mpc):
    '''Madau and Dickinson Cosmic Star Formation history'''
    c = 3E8#m/s
    H0 = 70E3#m/s/Mpc
    OnePz = 1+D_Mpc*H0/c
    #Eq. 15 from Madau & Dickinson 2014 - arXiv:1403.0007
    CSFH = 1.5E-2*np.power(OnePz,2.7)/(1+np.power(OnePz/2.9,5.6))#M_sun/yr/Mpc^3

    return D_Mpc, CSFH

### Class to handle catalog of galaxies ###########################
class TableGalaxies:
    '''Table of Galaxies Class'''

    def __init__(self, filename, fd_method=False, precision=0.01, N_points=1000, estimator="median", verbose=False):
        '''Initialize an instance of the class'''
        self.file = filename
        self.t = Table.read(filename,format="ascii")
        self.clean_table()

        # Parameters
        self.R, self.eta, self.B, self.lc, self.dmax = 7, 1e-5, 100, 1, 1.5
        self.fd_method        = fd_method
        self.precision        = precision
        self.N_points         = N_points
        self.estimator_method = estimator


    def clean_table(self):
        '''Remove useless rows and columns'''
        #remove the column that are not used
        valid_col = ['Name','J2000.0','D','logSFR','elogSFR','logM*']
        invalid_colum = []
        for col_name in self.t.colnames:
            if col_name not in valid_col:
                invalid_colum.append(col_name)
        self.t.remove_columns(invalid_colum)

        #check if the valid columns are in the file
        for col_name in valid_col:
            try:
                self.t[col_name]
            except ValueError:
                print(self.file,": missing column ->",col_name)

        #remove lines with no D  or SFR estimate
        n_gal = self.t['Name'].size
        invalid_rows = self.t['D'].mask + self.t['logSFR'].mask + self.t['elogSFR'].mask + self.t['logM*'].mask
        self.t.remove_rows(invalid_rows)
        n_del = invalid_rows.sum()

        #add a "J" in front of each coordinate
        coordinate = []
        for c in self.t['J2000.0']:
            coordinate.append('J'+c)
        self.t.add_column(Column(coordinate),name="Coord")
        #self.t.remove_column('J2000.0')

        #(print) statement
        print('\n',self.file,": table is clean,", n_del,"/",n_gal,"removed")

    def SetEstimatorMethod(self, estimator="median"):

        if(estimator != "median" and estimator != "mean"):
            print("Bad choice of self.estimator_method, it should be equal to mean or median")
            print("Current : self.estimator_method = " + str(estimator))
            exit()
        else:
            self.estimator_method = estimator

    def SetParameters(self, R, eta, B=100, lc=1, dmax=1.5):
        self.R, self.eta, self.B, self.lc, self.dmax = R, eta, B, lc, dmax

    def Delta_theta(self, D):
        '''R in EV, B in nG, Lc in kPc, D & Dmax in Mpc'''

        if D <= self.dmax:
            result = 25*(10/self.R)*(self.B/100)*(self.lc/10)**0.5*(D/1)**0.5 * np.pi/180
        else:
            result = 25*(10/self.R)*(self.B/100)*(self.lc/10)**0.5*(self.dmax/1)**0.5 * np.pi/180
        return result

    def Delta_tau(self, D, delta_theta):
        '''Compute delta_tau from J. Biteau 22/03/2019'''

        c    = 3E8#m/s
        D    = D * 3.085 * 10**22  #Convert Mpc to m
        dmax = self.dmax * 3.085 * 10**22 #Convert Mpc to m

        if D <= dmax:
            result = delta_theta**2 / (4*c) * D
        else:
            result = delta_theta**2 / (4*c) * dmax
        result = result / (3600*24*365.25)
        return result

    def Select(self, Dmin=0, Dmax=np.infty):
        '''Saved the galaxies that are between Dmin & Dmax'''

        id_range   = np.where( (self.t['D']>Dmin) & (self.t['D']<=Dmax) )
        self.log_sfr, self.dlog_sfr = self.t['logSFR'][id_range], self.t['elogSFR'][id_range]
        self.D     = self.t['D'][id_range]
        self.name  = self.t['Name'][id_range]
        self.coord = self.t['Coord'][id_range]
        self.j2000 = self.t['J2000.0'][id_range]


    def lambdaV(self):
        '''Compute the sum of the average SFR'''

        self.delta_tau = [ self.Delta_tau(self.D[i], self.Delta_theta(self.D[i])) for i in range(len(self.D))]
        self.log_lambda  = [self.log_sfr[i]   + np.log10(self.delta_tau[i] * self.eta) for i in range(len(self.D))]
        self.dlog_lambda = [self.dlog_sfr[i]  for i in range(len(self.D))]
        self.omega       = 1/(4*np.pi*self.D*self.D)


    def Nburst(self):
        '''Compute the number of expected burst for all galaxies'''

        self.estimator = np.zeros_like(self.log_lambda)
        self.flux   = np.zeros_like(self.log_lambda)

        if self.estimator_method == "median":
            for i in tqdm(range(len(self.log_lambda))):

                # If log_lambda < -5, take 0 as the median
                if self.log_lambda[i] < -4.5:
                    self.estimator[i] = 0

                else:
                    # Convert log_lambda & dlog to mu & sigma
                    mu, sigma = lib.log_to_ln(self.log_lambda[i], self.dlog_lambda[i])

                    # Compute boundaries
                    lambda_min, lambda_max = lib.boundaries_log(self.precision, mu, sigma)
                    k_max = (lambda_max+10).astype(int)
                    k = np.linspace(0,k_max,k_max+1)

                    # Create distribution
                    distribution = lib.Custom_pdf(0, k_max)

                    # Compute the distribution
                    distribution.calcul(self.precision, mu, sigma, lambda_min, lambda_max)

                    # Get CDF
                    cdf = distribution.cdf(k)

                    # Compute the estimator
                    self.estimator[i]   = lib.GetMedian_1D(k, cdf)
                pass
        elif self.estimator_method == "mean":
            self.estimator = self.log_lambda

        else:
            self.BadEstimatorMethod()

        self.flux = self.estimator*self.omega

        N_galaxies = len(self.flux)
        count = N_galaxies - np.count_nonzero(self.flux)
        print (str(count) + "/" + str(N_galaxies) + " galaxies have flux = 0")

    def Nburst_multi_process(self):
        '''Do the same as N_burst but using multi processing'''

        start_time = lib.time.time()
        q = [lib.mp.Queue() for i in range(len(self.log_lambda))]
        p = [lib.mp.Process(target=Compute_median_multi_process, args=(q[i], self.log_lambda[i], self.dlog_lambda[i], self.precision)) for i in range(len(q))]
        [p[i].start() for i in range(len(p))]
        self.estimator = [np.float(q[i].get()) for i in range(len(p))]
        [p[i].join() for i in range(len(p))]

        np.asarray(self.estimator)
        self.flux = self.estimator * self.omega

        N_galaxies = len(self.flux)
        count = N_galaxies - np.count_nonzero(self.flux)
        print (str(count) + "/" + str(N_galaxies) + " galaxies have flux = 0")

        print("---Nburst_multi_process took %s seconds ---" % (lib.time.time() - start_time))

    def Nburst_grouping(self, Dmin, Dmax):
        '''Throw a Monte-Carlo simulation with the galaxies between Dmin & Dmax_plot
        Return the value of the expected flux & the error on this flux'''

        # Select the event between Dmin & Dmax
        id_range   = np.where( (self.D>Dmin) & (self.D<=Dmax) )[0]

        log_lambdaV  = [self.log_lambda[id_range[i]] for i in range(len(id_range))]
        dlog_lambdaV = [self.dlog_lambda[id_range[i]] for i in range(len(id_range))]
        delta_tau    = [self.delta_tau[id_range[i]] for i in range(len(id_range))]
        #omega        = [self.omega[id_range[i]] for i in range(len(id_range))]


        log_lambdaV = np.array(log_lambdaV)
        dlog_lambdaV = np.array(dlog_lambdaV)
        omega  = np.ones_like(log_lambdaV)
        delta_tau_moy = np.sum(delta_tau)/np.size(delta_tau)

        if self.estimator_method =="median":
            # Compute the variance of each distribution
            mu, sigma = lib.log_to_ln(log_lambdaV, dlog_lambdaV)
            variance = variance_cpd(mu,sigma)


            # Compute histogram using MC simulation
            pdf, edges = lib.Compute_histogram_multi_process(log_lambdaV, dlog_lambdaV, omega, bin_fd=self.fd_method, precision=self.precision, N_points=self.N_points)
            x_bins = edges[:-1]

            # Get the median/flux
            cdf    = lib.ComputeCDF_1D(pdf)
            estimator = lib.GetMedian_1D(x_bins, cdf) / (delta_tau_moy * self.eta)
            variance_tot = np.sum(variance / (np.asarray(delta_tau)**2 * self.eta**2))


        elif self.estimator_method =="mean":
            mean_value = np.asarray(mean_lognormal(log_lambdaV, dlog_lambdaV))

            if len(mean_value)>0:
                estimator    = np.sum(mean_value / delta_tau) / self.eta
                variance_tot = variance_lognormal_div_mean2(dlog_lambdaV) * mean_value**2
                variance_tot = np.sum(variance_tot / (np.asarray(delta_tau)**2 * self.eta**2))
            else:
                estimator = 0
                variance_tot = 0
        else:
            self.BadEstimatorMethod()


        return estimator, variance_tot

    def Smeared_Nburst_grouping(self, Dmin, Dmax):
        '''The same as Nburst_grouping but return the montecarlo run & the variance
        instead of the median and the error'''

        # Select the event between Dmin & Dmax
        id_range   = np.where( (self.D>Dmin) & (self.D<=Dmax) )[0]

        log_lambdaV  = [self.log_lambda[id_range[i]] for i in range(len(id_range))]
        dlog_lambdaV = [self.dlog_lambda[id_range[i]] for i in range(len(id_range))]
        delta_tau    = [self.delta_tau[id_range[i]] for i in range(len(id_range))]

        log_lambdaV = np.array(log_lambdaV)
        dlog_lambdaV = np.array(dlog_lambdaV)
        omega  = np.ones_like(log_lambdaV)

        # Compute the variance of each distribution
        mu, sigma = lib.log_to_ln(log_lambdaV, dlog_lambdaV)
        variance = variance_cpd(mu,sigma)

        # Compute histogram using MC simulation
        if len(omega)>0:
            random_sampling, binning = lib.Return_histogram_multi_process(log_lambdaV, dlog_lambdaV, omega, bin_fd=self.fd_method, precision=self.precision, N_points=self.N_points)
        else:
            random_sampling = np.zeros(1)

        variance_tot = np.sum(variance / (np.asarray(delta_tau)**2 * self.eta**2))

        return random_sampling, variance_tot

    def Binning_galaxies(self, Dmin, Dmax, dsmear=0.25, step_hist=0.01, step=0.01):
        '''Compute the histogram of bin of step_hist width, then do a gaussian
        smearing every step with sigma=dsmear'''

        # Create array for points & binning
        x_points = np.arange(Dmin, Dmax, step)
        binning_hist = np.arange(Dmin, Dmax, step_hist)

        N_points = len(x_points)-(2*int(dsmear/step)+1)
        x = np.zeros(N_points)
        sfr_DividedByV = np.zeros(N_points)
        variance_tot_DividedByVSquared = np.zeros(N_points)

        if self.estimator_method =="median":
            result  = [self.Smeared_Nburst_grouping(binning_hist[i], binning_hist[i+1]) for i in tqdm(range(len(binning_hist)-1))]

        elif self.estimator_method =="mean":

            mean = np.zeros(len(binning_hist)-1)
            var = np.zeros(len(binning_hist)-1)

            for i in range(len(binning_hist)-1):
                mean[i], var[i] = self.Nburst_grouping(binning_hist[i], binning_hist[i+1])


        # Compute the volume of each shell
        Volume = ((binning_hist[:-1]+step_hist)**3-(binning_hist[:-1])**3)*4*np.pi/3

        for i in range(N_points):

            # Compute the mean of the gaussian & the associated delta_tau
            mu        = (x_points[i]+x_points[i+1])/2
            delta_tau = self.Delta_tau(mu, self.Delta_theta(mu))

            # Normalize omega
            omega = Gaus(binning_hist[:-1]+step_hist/2, mu, dsmear)
            norm = np.sum(omega)*step_hist/(np.sqrt(2*np.pi)*dsmear)
            omega /= norm

            #Compute volume and position
            V = np.sum(Volume*omega)
            x[i] = mu

            #Compute median & variance
            if self.estimator_method =="median":
                result_tot = [result[j][0]*omega[j] for j in range(len(result))]
                result_tot = np.sum(result_tot, axis=0)
                variance   = [result[j][1]*omega[j] for j in range(len(result))]
                variance   = np.sum(variance)

                # Do the histogram & get the median
                max_bin   = max(result_tot+1)
                bin       = np.arange(0,max_bin, 0.01)

                pdf, edges = np.histogram(result_tot, bins=bin)
                pdf = pdf/pdf.sum()
                x_bins = edges[:-1]

                cdf    = lib.ComputeCDF_1D(pdf)
                estimator = lib.GetMedian_1D(x_bins, cdf)

                # Store median & variance
                sfr_dividedByV[i] = estimator / self.eta / delta_tau / V
                variance_tot_DividedByVSquared[i] =  variance / (V**2)

            elif self.estimator_method =="mean":
                sfr_DividedByV[i] = np.sum(mean*omega) / V
                variance_tot_DividedByVSquared[i]  = np.sum(var*omega)  / (V**2)

        return x, sfr_DividedByV, np.sqrt(variance_tot_DividedByVSquared)


    def Nburst_density(self, Dmin=0, Dmax=np.infty):
        '''Compute the flux between Dmin & Dmax'''

        #default result
        dres, flux_res, e_flux_res = 0, 0, 0

        #identify galaxies in the right distance range
        id_range = np.where( (self.t['D']>Dmin) & (self.t['D']<=Dmax) )
        d = self.t['D'][id_range]

        #ensure that we are not counting outside of the range of interest
        dmin_all = np.min(self.t['D'])
        Dmin = max(Dmin,dmin_all)
        dmax_all = np.max(self.t['D'])
        Dmax = min(Dmax,dmax_all)

        if d.size>0:
            #looks at the minimum and maximum distance for the volume estimation

            if Dmax>Dmin:
                V = (Dmax**3-Dmin**3)*4*np.pi/3
                flux_tot, var_flux_tot = self.Nburst_grouping(Dmin, Dmax)
                e_flux_tot = np.sqrt(var_flux_tot)
                dres = (3/4)*(Dmax**4-Dmin**4)/(Dmax**3-Dmin**3)#assume uniform distrib in the shell
                flux_res, e_flux_res = flux_tot/V, e_flux_tot/V

        return dres, flux_res, e_flux_res


    def sfr_tot(self, Dmin=0, Dmax=np.infty):
        '''Compute the sum of the average SFR'''

        #identify galaxies in the right distance range
        id_range = np.where( (self.t['D']>Dmin) & (self.t['D']<=Dmax))
        sfr, e_sfr = self.t['logSFR'][id_range], self.t['elogSFR'][id_range]

        #mean and variance for each galaxy
        mean_sfr = mean_lognormal(sfr, e_sfr)
        variance_div_mean2_sfr = variance_lognormal_div_mean2(e_sfr)
        variance_sfr = variance_div_mean2_sfr *mean_sfr**2

        #total mean and variance as sum
        tot_sfr = mean_sfr.sum()
        tot_var = variance_sfr.sum()

        return tot_sfr, np.sqrt(tot_var)


    def sfr_density(self, dmin=0, dmax=np.infty):
        '''Compute the average SFR density'''

        #default result
        dres, sfr_res, esfr_res = 0, 0, 0

        #identify galaxies in the right distance range
        id_range = np.where( (self.t['D']>dmin) & (self.t['D']<=dmax) )
        d = self.t['D'][id_range]

        #ensure that we are not counting outside of the range of interest
        dmin_all = np.min(self.t['D'])
        dmin = max(dmin,dmin_all)
        dmax_all = np.max(self.t['D'])
        dmax = min(dmax,dmax_all)

        if d.size>0:
            #looks at the minimum and maximum distance for the volume estimation
            if dmax>dmin:
                V = (dmax**3-dmin**3)*4*np.pi/3
                sfr_tot, e_sfr_tot = self.sfr_tot(dmin, dmax)
                dres = (3/4)*(dmax**4-dmin**4)/(dmax**3-dmin**3)#assume uniform distrib in the shell
                sfr_res, esfr_res = sfr_tot/V, e_sfr_tot/V

        return dres, sfr_res, esfr_res


    def smeared_sfr_density(self, dmin=0, dmax=100, dsmear = 0.25):
        '''Gaussian smoothed SFR density - dsmear in Mpc'''

        #Load the data
        #id_range = np.where( (self.t['D']>dmin) & (self.t['D']<=dmax) )
        id_range   = np.where( (self.D>dmin) & (self.D<=dmax) )[0]
        dist_src = self.D
        sfr_src  = mean_lognormal(self.t['logSFR'][id_range], self.t['elogSFR'][id_range])

        variance_div_mean2_sfr = variance_lognormal_div_mean2(self.t['elogSFR'][id_range])
        variance_sfr = variance_div_mean2_sfr * sfr_src**2

        #Range of interest
        dmin, dmax = np.min(dist_src), np.max(dist_src)
        #d_range = np.linspace(dmin, dmax, np.int(10*(dmax-dmin)/dsmear))
        d_range = np.arange(dmin, dmax, 0.01)
        delta_d = d_range[1]-d_range[0]

        volume = 4.*np.pi/3.*((d_range+delta_d/2)**3-(d_range-delta_d/2)**3)

        #Fill a function as a sum of weigthed Dirac
        m = np.zeros_like(d_range)
        var_m = np.zeros_like(d_range)
        var_sfr = np.zeros_like(d_range)


        for i, d in enumerate(dist_src):
            dist = d
            idx = find_nearest(dist,d_range)
            m[idx] += sfr_src[i]
            var_m[idx] += variance_sfr[i]

        #Convolve SFR with Gaussian centered on mid range
        dmid = 0.5*(d_range[0]+d_range[-1])
        window = Gaus(d_range, dmid, dsmear)
        norm = np.sum(window)*delta_d/(np.sqrt(2*np.pi)*dsmear)


        sfr = np.convolve(m, window/norm, 'same')
        var_sfr = np.convolve(var_m, window/norm, 'same')
        conv_volume = np.convolve(volume, window/norm, 'same')

        e_sfr = np.sqrt(var_sfr)

        '''#Compute volume
        r1 = d_range+0.5*delta_d
        r2 = d_range-0.5*delta_d
        V = 4.*np.pi/3.*(r1**3-r2**3)'''

        d_range = d_range + delta_d*0.5

        return d_range, sfr/conv_volume, e_sfr/conv_volume


    ### Plot the SFR and SFR density ###############################################
    def plotSFR(self, dmin = 0, dmax=100, step=0.5, noDisplay=False):
        '''Two panel plot, showing individual objects and their global SFR density'''

        if noDisplay:
            plt.ioff()

        plt.rcParams.update({'font.size': 16,'legend.fontsize': 12})

        #Plot range
        Dmin_plot, Dmax_plot = 0.005, 20

        #load the source SFR and distances
        id_range = np.where( (self.t['D']>dmin) & (self.t['D']<=dmax) )
        d_src      = self.D
        sfr_src    = mean_lognormal(self.t['logSFR'][id_range], self.t['elogSFR'][id_range])

        #Distance bins for computation
        dist = np.arange(dmin, dmax, step)

        print("############################")
        print("Values of SFR : (x, sfr, e_sfr)")
        #Compute the star formation density
        d_dens, sfr_dens, e_sfr_dens = [], [], []
        for i in range(0,dist.size-1):
            ds, s, es = table_LocalVolume.sfr_density(dist[i],dist[i+1])
            if s>0:
                d_dens.append(ds)
                sfr_dens.append(s)
                e_sfr_dens.append(es)
                print(ds,s,es)

        #Compute a smeared SFR density from the data
        d_range_smear, sfr_d_smear, e_sfr_d_smear = self.smeared_sfr_density(dmin, dmax)


        #Load the Cosmic Star Formation History
        dCSFH, CSFH = MadauDickinson_SFR(np.logspace(np.log(Dmin_plot), np.log(Dmax_plot)))

        #Plot########################################################
        fig, (ax1, ax2) = plt.subplots(figsize=(5, 8), nrows=2)
        plt.tight_layout()
        plt.subplots_adjust(left = 0.22, bottom = 0.08)#, right=0.95, bottom = 0.1, top = 0.95

        #Panel 1: sources
        plt.subplot(211)
        plt.xlabel(r"Distance, $D$  [Mpc]")
        plt.ylabel(r"SFR  [M$_\odot$ yr$^{-1}$]")
        plt.xlim(0, Dmax_plot)
        if(dmin > 1):
            plt.xlim(0,12)
            plt.ylim(0,1.8)
        else:
            plt.xlim(0,11)
            plt.yscale('log')
            plt.ylim(1e-8,1e1)

        plt.plot(d_src, sfr_src,alpha=0.2, marker='o', linestyle='')

        #Panel 2: SFR density
        plt.subplot(212)
        plt.xlabel(r"Distance, $D$  [Mpc]")
        plt.ylabel(r"SFR density  [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]")
        plt.xlim(0, Dmax_plot)
        if(dmin > 1):
            plt.xlim(0,12)
            plt.ylim(0,0.15)
        else:
            plt.xlim(0,11)
            plt.yscale('log')
            plt.ylim(1e-4,1e2)

        e_sfr_dens = np.asarray(e_sfr_dens)
        sfr_dens = np.asarray(sfr_dens)

        e_lowerSFR  = sfr_dens*(1-np.exp(-e_sfr_dens/sfr_dens))
        e_higherSFR = sfr_dens*(np.exp(+e_sfr_dens/sfr_dens)-1)

        plt.errorbar(d_dens, sfr_dens, yerr = (e_lowerSFR,e_higherSFR), linestyle = '', marker = 'o', label=r"Local Volume")
        plt.plot(dCSFH, CSFH,alpha=0.5, linestyle='-.', label="Cosmic SFR")

        lowerY  = sfr_d_smear*np.exp(-e_sfr_d_smear/sfr_d_smear)
        higherY = sfr_d_smear*np.exp(+e_sfr_d_smear/sfr_d_smear)
        plt.fill_between(d_range_smear, lowerY, higherY, alpha=0.5, color='tab:blue', label=r'Smeared')
        plt.plot(d_range_smear, sfr_d_smear, label = "Running average", color = "tab:blue")
        plt.legend()

        if noDisplay:
            plt.close(fig)

    ### Plot the flux and flux density ###############################################
    def plotLambda(self, dmin = 0, dmax=100, noDisplay=False):
        plt.errorbar(self.D, self.log_lambda, yerr=self.dlog_lambda, alpha=0.2, marker='o', linestyle='')
        #plt.xscale("log")
        plt.xlim(dmin,dmax)
        plt.xlabel(r"Distance, $D$  [Mpc]")
        plt.ylabel(r"lambda")
        plt.show()

    def plotFilteredSFR(self, dmin = 0, dmax=100, step = 0.5, filename_pic = "test.png", filename_sfr="test.csv", noDisplay=False):
        '''Two panel plot, showing individual objects and their global flux density'''

        if noDisplay:
            plt.ioff()

        plt.rcParams.update({'font.size': 16,'legend.fontsize': 12})

        #Plot range
        Dmin_plot, Dmax_plot = 0.005, 20

        #load the source SFR and distances
        d_src      = self.D
        flux_src   = np.asarray(self.estimator) / (np.asarray(self.eta) * np.asarray(self.delta_tau))

        #Distance bins for computation
        dist = np.arange(dmin, dmax, step)

        x_Smeared, result_Smeared, variance_Smeared = self.Binning_galaxies(dmin,dmax)



        #Compute the star formation density
        print("############################")
        print("Values of filtered SFR : (x, sfr, e_sfr)")
        d_dens, flux_dens, e_flux_dens = [], [], []
        for i in tqdm(range(0,dist.size-1)):
            ds, s, es = table_LocalVolume.Nburst_density(dist[i],dist[i+1])
            if s>0:
                d_dens.append(ds)
                flux_dens.append(s)
                e_flux_dens.append(es)
        [print(d_dens[i],flux_dens[i],e_flux_dens[i]) for i in range(len(d_dens))]


        #Load the Cosmic Star Formation History
        dCSFH, CSFH = MadauDickinson_SFR(np.logspace(np.log(Dmin_plot), np.log(Dmax_plot)))

        #Plot########################################################
        fig, (ax1, ax2) = plt.subplots(figsize=(5, 8), nrows=2)
        plt.tight_layout()
        plt.subplots_adjust(left = 0.22, bottom = 0.08)#, right=0.95, bottom = 0.1, top = 0.95

        #Panel 1: sources
        plt.subplot(211)
        plt.xlabel(r"Distance, $D$  [Mpc]")
        plt.ylabel(r"$\rm SFR_{\rm UHECR}$ [M$_\odot$ yr$^{-1}$]")
        plt.xlim(0, Dmax_plot)
        if(dmin > 1):
            plt.xlim(0,12)
            plt.ylim(0,1.8)
        else:
            plt.xlim(0,11)
            plt.yscale('log')
            plt.ylim(1e-8,1e1)

        plt.plot(d_src, flux_src,alpha=0.2, marker='o', linestyle='')

        #Panel 1: density
        plt.subplot(212)
        plt.xlabel(r"Distance, $D$  [Mpc]")
        plt.ylabel(r"$\rm SFR_{\rm UHECR}$ density [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]")
        plt.xlim(0, Dmax_plot)

        if(dmin > 1):
            plt.xlim(0,12)
            plt.ylim(0,0.15)
        else:
            plt.xlim(0,11)
            plt.yscale('log')
            plt.ylim(1e-4,1e2)

        e_flux_dens = np.asarray(e_flux_dens)
        flux_dens = np.asarray(flux_dens)



        e_lowerFlux  = flux_dens*(1-np.exp(-e_flux_dens/flux_dens))
        e_higherFlux = flux_dens*(np.exp(+e_flux_dens/flux_dens)-1)


        plt.errorbar(d_dens, flux_dens, yerr = (e_lowerFlux,e_higherFlux), linestyle = '', marker = 'o', label=r"Local Volume")
        plt.plot(dCSFH, CSFH,alpha=0.5, linestyle='-.', label="Cosmic SFR")

        lowerY  = result_Smeared*np.exp(-variance_Smeared/result_Smeared)
        higherY = result_Smeared*np.exp(+variance_Smeared/result_Smeared)

        plt.fill_between(x_Smeared, lowerY, higherY, alpha=0.5, color='orange', label=r'Smeared')
        plt.plot(x_Smeared, result_Smeared, color='orange', linestyle = '-', label=r"Running value")
        plt.legend()
        plt.savefig(filename_pic)
        WriteSFR(x_Smeared, result_Smeared, filename_sfr)

        if noDisplay:
            plt.close(fig)


    ### Plot a skymap of average SFR/4piD^2 ###########################
    def mapSFR(self, dmin = 0, dmax = np.infty, norm=1000, noDisplay=False):
        '''Map of SFR/4piD^2 vs distance'''

        if noDisplay:
            plt.ioff()

        plt.rcParams.update({'font.size': 16})
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)

        id_range = np.where( (self.t['D']>dmin) & (self.t['D']<dmax) )

        dist_src = self.t['D'][id_range]
        sfr_src = mean_lognormal(self.t['logSFR'][id_range], self.t['elogSFR'][id_range])
        coord_src = self.t['Coord'][id_range]

        flux = sfr_src/(4*np.pi*dist_src**2)
        sc = Coord.SkyCoord(coord_src, unit=(u.hour, u.deg))

        size_scale = norm*np.sqrt(flux)
        color_scale = dist_src

        #format -> left to right, centered on 180 deg
        dec_rad = sc.dec.radian
        ra_rad = sc.ra.radian
        angle_center = Coord.Angle(180,u.deg)
        ra_plot = angle_center.rad - ra_rad

        #supergalactic plane
        npts_plane = 4096
        sgp = Coord.SkyCoord(sgl = np.linspace(0, 360, npts_plane)*u.degree, sgb=np.zeros(npts_plane)*u.degree, frame="supergalactic")
        sgp_EQ = sgp.icrs
        sgl_rad = angle_center.rad - sgp_EQ.ra.radian
        sgb_rad = sgp_EQ.dec.radian

        #local sheet plane
        ls = LS.LocalSheet(lsl = np.linspace(0, 360, npts_plane)*u.degree, lsb=np.zeros(npts_plane)*u.degree)
        ls_EQ = ls.transform_to(Coord.ICRS)
        lsl_rad = angle_center.rad - ls_EQ.ra.radian
        lsb_rad = ls_EQ.dec.radian

        #plot
        fig = plt.figure(figsize=(8,4))
        title_plot = str(dmin)+r"$\,$Mpc$ < D < $"+str(dmax)+"$\,$Mpc"
        plt.suptitle(title_plot,fontsize=16)
        ax = fig.add_subplot(111, projection="mollweide")
        ax.set_xticklabels([r"330$\degree$", r"300$\degree$", r"270$\degree$", r"240$\degree$", r"210$\degree$", r"180$\degree$", r"150$\degree$", r"120$\degree$", r"90$\degree$", r"60$\degree$", r"30$\degree$"])

        plt.xlabel('R.A.')
        plt.ylabel('Dec.')
        plt.grid(True)
        #plt.scatter(sgl_rad,sgb_rad, marker=".", c = 'k', s = 0.5)
        plt.scatter(lsl_rad,lsb_rad, marker=".", c = 'grey', s = 0.5)


        p = plt.scatter(ra_plot, dec_rad, c = color_scale, vmin=0, vmax = np.max(self.t['D']), s = size_scale, marker = 'o', alpha = 0.5)
        cbar = plt.colorbar(p, fraction = 0.023, pad = 0.04)
        cbar.set_label("Distance  [Mpc]", rotation=270, labelpad=+20)

        plt.subplots_adjust(top=0.95,bottom=0.0)
        if noDisplay:
            plt.close(fig)


    def mapFilteredSFR(self, dmin = 0, dmax = np.infty, filename="test.png", norm=1000, noDisplay=False):
        '''Map of the flux vs distance'''

        if noDisplay:
            plt.ioff()

        plt.rcParams.update({'font.size': 16})
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)


        dist_src = self.D
        coord_src = self.coord

        flux = np.asarray(self.flux) / np.asarray(self.eta) / np.asarray(self.delta_tau)
        sc = Coord.SkyCoord(coord_src, unit=(u.hour, u.deg))

        size_scale = norm*np.sqrt(flux)
        color_scale = dist_src

        #format -> left to right, centered on 180 deg
        dec_rad = sc.dec.radian
        ra_rad = sc.ra.radian
        angle_center = Coord.Angle(180,u.deg)
        ra_plot = angle_center.rad - ra_rad

        #supergalactic plane
        npts_plane = 4096
        sgp = Coord.SkyCoord(sgl = np.linspace(0, 360, npts_plane)*u.degree, sgb=np.zeros(npts_plane)*u.degree, frame="supergalactic")
        sgp_EQ = sgp.icrs
        sgl_rad = angle_center.rad - sgp_EQ.ra.radian
        sgb_rad = sgp_EQ.dec.radian

        #local sheet plane
        ls = LS.LocalSheet(lsl = np.linspace(0, 360, npts_plane)*u.degree, lsb=np.zeros(npts_plane)*u.degree)
        ls_EQ = ls.transform_to(Coord.ICRS)
        lsl_rad = angle_center.rad - ls_EQ.ra.radian
        lsb_rad = ls_EQ.dec.radian

        #plot
        fig = plt.figure(figsize=(8,4))
        title_plot = str(dmin)+r"$\,$Mpc$ < D < $"+str(dmax)+"$\,$Mpc"
        plt.suptitle(title_plot,fontsize=16)
        ax = fig.add_subplot(111, projection="mollweide")
        ax.set_xticklabels([r"330$\degree$", r"300$\degree$", r"270$\degree$", r"240$\degree$", r"210$\degree$", r"180$\degree$", r"150$\degree$", r"120$\degree$", r"90$\degree$", r"60$\degree$", r"30$\degree$"])

        plt.xlabel('R.A.')
        plt.ylabel('Dec.')
        plt.grid(True)
        #plt.scatter(sgl_rad,sgb_rad, marker=".", c = 'k', s = 0.5)
        plt.scatter(lsl_rad,lsb_rad, marker=".", c = 'grey', s = 0.5)


        p = plt.scatter(ra_plot, dec_rad, c = color_scale, vmin=0, vmax = np.max(self.t['D']), s = size_scale, marker = 'o', alpha = 0.5)
        cbar = plt.colorbar(p, fraction = 0.023, pad = 0.04)
        cbar.set_label("Distance  [Mpc]", rotation=270, labelpad=+20)

        plt.subplots_adjust(top=0.95,bottom=0.0)
        plt.savefig(filename)

        if noDisplay:
            plt.close(fig)

    def WriteSimple(self):

        filename = "src_data_update.dat"
        f = open(filename,"w")
        f.write("#Name    J2000.0    D[Mpc]    log10(SFRHa[M.yr-1])\n")
        for i, n in enumerate(self.name):
            f.write(n.replace(" ", "")+"\t"+self.j2000[i]+"\t"+str(self.D[i])+"\t"+str(self.sfr[i])+"\n")


    def WriteCSV(self, filename='flux.csv'):
        '''Save the computation to a file'''

        print("\n-->Writing mode on, creating a file " + filename)

        with open(filename, mode='w') as flux_file:
            flux_writer = csv.writer(flux_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            flux_writer.writerow(['Name', 'log_lambda', 'dlog_lambda', 'delta_tau', 'omega','D', 'flux'])
            [flux_writer.writerow([self.name[i], self.log_lambda[i], self.dlog_lambda[i], self.delta_tau[i], self.omega[i], self.D[i], self.flux[i]]) for i in range(len(self.log_lambda))]

    def ReadCSV(self, filename='flux.csv'):
        '''Read the saved computed data from a file'''

        print("-->  Loading computed data from : " + filename + " ...", end='')

        try:
            df = pandas.read_csv(filename)
        except:
            print(" Failed !")
            print("\n/!\ File " + filename + " does not exist ! (Use -w to create it)\n")

        self.log_lambda  = np.array(df['log_lambda'])
        self.dlog_lambda = np.array(df['dlog_lambda'])
        self.omega       = np.array(df['omega'])
        self.flux        = np.array(df['flux'])
        self.estimator      = self.flux / self.omega
        self.D           = np.array(df['D'])
        self.delta_tau   = np.array(df['delta_tau'])
        print(" Succeed !")

    def Set_dist(self, dmin, dmax):
        '''Check if dmin & dmax are in range, if not change it to match the data'''

        min_d = min(self.t['D'])
        max_d = max(self.t['D'])

        if dmin < 0:
            print("\ndmin < 0, dmin as been set to " +str(min_d) + " instead of " + str(dmin) +"\n")
            dmin = min_d
        if dmax > max_d:
            print("\ndmax > maximum distance, dmax as been set to " +str(max_d) + " instead of " + str(dmax)+"\n")
            dmax = max_d
        return dmin, dmax


### Main ##########################################################
if __name__ == "__main__":

    # Parser to select writing/reading mode
    parser = argparse.ArgumentParser()
    parser.add_argument("R", type=float, help="Value of R_cut")
    parser.add_argument("eta",type=float, help="Value of eta")
    parser.add_argument("-w","--write", help="If used, compute the result and print it into a file flux.csv. If off, load the result from a file flux.csv", action="store_true")
    parser.add_argument("-n","--noDisplay", help="If used, doesn't display the result on the screen", action="store_true")
    parser.add_argument("-m","--method", help="Choose the estimator you want to use. Can be median or mean", type=str)
    args = parser.parse_args()


    #Load the Local Volume table
    filename_LocalVolume = "LocalVolume/Karachentsev18_1029Gal_LV_Augmented.dat"
    table_LocalVolume    = TableGalaxies(filename_LocalVolume, N_points=1000)

    # Range & step for computing SFR & Flux density
    step_density = 1. #Mpc
    dmin, dmax = table_LocalVolume.Set_dist(0, 11) # Put dmin & dmax in MPC

    filename_flux        = "flux_" + str(dmin)+"_"+str(dmax) + "/flux_" +str(args.R)+"_" +str(args.eta)+".csv"
    filename_graph       = "result_"+ str(dmin)+"_"+str(dmax) +"/graph_" +str(args.R)+"_" +str(args.eta)+".png"
    filename_map         = "result_"+ str(dmin)+"_"+str(dmax) +"/map_" +str(args.R)+"_" +str(args.eta)+".png"
    filename_sfr         = "result_"+ str(dmin)+"_"+str(dmax) +"/sfr_" +str(args.R)+"_" +str(args.eta)+".csv"

    #Set Parameters for computing lambda
    table_LocalVolume.SetParameters(args.R, args.eta)

    #Select the events between dmin&dmax
    table_LocalVolume.Select(dmin, dmax)

    if args.method:
        table_LocalVolume.SetEstimatorMethod(args.method)

    # If write mode is on computed Nburst a write it into a CSV file
    if args.write:

        #Compute lambda, Nburst and write it to a csv file
        table_LocalVolume.lambdaV()
        if args.method == "median":
            table_LocalVolume.Nburst_multi_process()
            table_LocalVolume.WriteCSV(filename_flux)
        else:
            table_LocalVolume.Nburst()
    else:
        #Load lambda & Nburst from a file
        table_LocalVolume.ReadCSV(filename_flux)

    #Plot lambda
    #table_LocalVolume.plotLambda(dmin, dmax, noDisplay=args.noDisplay)

    #Plot the SFR
    table_LocalVolume.plotSFR(dmin, dmax, step=step_density, noDisplay=args.noDisplay)
    #table_LocalVolume.mapSFR(dmin, dmax, noDisplay=args.noDisplay)

    #Plot the flux vs distance & map the flux
    table_LocalVolume.plotFlux(dmin, dmax, step=step_density, filename_pic=filename_graph, filename_sfr=filename_sfr, noDisplay=args.noDisplay)
    #table_LocalVolume.mapFlux(dmin, dmax, filename=filename_map, noDisplay=args.noDisplay)

    #table_LocalVolume.WriteSimple()

    plt.show()
