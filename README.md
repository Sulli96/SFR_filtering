# SFR_filtering
## What is it ?
SFR_filtering gives an estimator of SFR when it follows a Compound Probability Distribution (CPD)
which is a mixture of a Log-normal distribution and a Poisson distribution.
It will create three files : The map of the sky after filtering, the graph of the filtered SFR vs distance, and a file csv which contains the distances and the values of the filtered SFR.

## What kind of estimator ?
It can computes the median or the mean.
The mean is computed analyticaly while the median needs a Monte-Carlo simulation

## What does it need ?
It loads a file named "Karachentsev18_1029Gal_LV_Augmented" in the "LocalVolume" directory.
This file need to have at least three columns : D(distance), logSFR, elogSFR for each galaxy.

## What kind of parameter can I tune ?

* The step density(Mpc, ie. the distance between 2 points) can be change in the main, so as the dmin & dmax.
* R & ETA are choosen when running the program.
* The parameters for computing delta_tau : B, lc, and dmax
Can be choosed by calling the method SetParameters(R, ETA, B=B_value, lc=lc_value, dmax=dmax_value)
* The estimator (median or mean) can be choosen when runing the program
* The number of points used in the Monte-Carlo Simulation, so as the precision, can be choosen when creating the object "TableGalaxies" by putting the option N_points = NUMBER or precision = NUMBER (ie. table_LocalVolume    = TableGalaxies(filename_LocalVolume, N_points=1000, precision = 0.01))

## How do I run it ?

To run it, type : python3 -W ignore catalog_handler.py VALUE_OF_R VALUE_OF_ETA -w.
* "-w" is used to compute the values of Filtered_SFR for individual galaxy.
When it is used, it stores the filtered values of each galaxy in a file flux_R_ETA.csv in a directory "flux_DMIN_DMAX".
This has been done to avoid computing everytime each galaxies.
When it has been compute for one set of parameter (R,ETA), the program can be run without the -w option and it will load the data instead of computing it again.
* "-W ignore" hides the warning message of Scipy when integrating functions. Those warning messages seems to have no impact on
the computations.

## What are the options ?

* "-w" to compute the values and store it into a file.
* "-n" to not show the figures on the screen (usefull when using a batch script to create all the figures/files).
* "-m" to choose the method for the estimator, can be mean or median(default).

# Important things to know

The program does its own test on each galaxy. If the integral of a computed PDF is bigger of less than twice the precision ask. It will come up with a warning message. This may happen in two cases :

The parameter lambda is too big or to small.
 To avoid that, when lambda is too small, the distribution is not computed and the galaxie does not contribute to the computations. This parameter can be tuned in the file lib.py, it is a global variable called Nlim. For now Nlim = -4, it means every lambda < -4 is not taken into account. Nothing has been done for big lambda since it hasn't be needed for now.
