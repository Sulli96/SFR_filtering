# SFR_filtering
## What is it ?
SFR_filtering gives an estimator of SFR when it follows a Compound Probability Distribution (CPD) 
wich is a mixture of a Log-normal distribution and a Poisson distribution.

## What kind of estimator ?
It can computes the median or the mean.

# What does it need ?
It loads a file named "Karachentsev18_1029Gal_LV_Augmented" in the "LocalVolume" directory. 
This file need to have at least three columns : D (distance), logSFR, elogSFR for each galaxies.

# What kind of parameter can I tune ?

* The step density(Mpc, ie. the distance between 2 points) can be change in the main, so as the dmin & dmax. 
* R & ETA are choosen when running the program.
* The parameters for computing delta_tau : B, lc, and dmax
Can be choosed by calling the method SetParameters(R, ETA, B=B_VALUE, lc=lc_value, dmax=dmax_value)
* The estimator (median or mean) can be choosed when runing the program

# How do i run it ?

To run it, type : python3 -W ignore catalog_handler.py VALUE_OF_R VALUE_OF_ETA -w.
* "-w" is used to compute the values of Filtered_SFR for individual galaxy.
When it is used, it stores the filtered values of each galaxy in a file flux_R_ETA.csv in a directory "flux_DMIN_DMAX".
This has been done to avoid computing everytime each galaxy.
When it has been compute for on set of parameter (R,ETA), the program can be run without the -w option and it will load the data instead
of computing it.
* "-W ignore" hides the warning message of Scipy when integrating functions. Those warning messages seems to have no impact on
the computations.

# Important things to know

The program has its own warning message. if the integral of a computed PDF is not close enough to 1.
