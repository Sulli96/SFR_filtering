from astropy.table import Table, vstack, Column
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import minimize
from scipy import stats, interpolate
import copy 

#Utilities####################################################################
def GausNorm(nentries, bin_size):
	norm = nentries*bin_size/(np.sqrt(2*np.pi))
	f = lambda x: np.exp(-x*x/2)*norm
	return f

#Table loading, put the relevant info in the Ha column #######################
#Load the tables and return SFR(Ha)
def LoadHa(filename):
	#columns = ["Name","J2000.0","T","D","logM*","logMHI","logSFRFUV","logSFRHa","SB","Th1","Thj"]#to remember what you've access to
	t = Table.read(filename, format='ascii')
	return t

def LoadFUV(filename):
	#columns = ["Name","J2000.0","T","D","logM*","logMHI","logSFRFUV","logSFRHa","SB","Th1","Thj"]#to remember what you've access to
	t = Table.read(filename, format='ascii')
	t["logSFRHa"] = t["logSFRFUV"]
	return t

def LoadM(filename):
	#columns = ["Name","J2000.0","T","D","logM*","logMHI","logSFRFUV","logSFRHa","SB","Th1","Thj"]#to remember what you've access to
	t = Table.read(filename, format='ascii')
	t["logSFRHa"] = t["logM*"]
	return t

###### Fit 1D ######################################################################
def fun_res_1D(x, sig, sig_slope):
	return sig-sig_slope*x

def fun_bias_1D(x, a, b):
	return a+b*x

def chi_square_1D_type(data,param):
	x, y = data
	a, b, sig, sig_slope = param

	f_x = fun_bias_1D(x,a, b)
	sig_y = fun_res_1D(x, sig, sig_slope)

	res2 = (y-f_x)**2/sig_y**2
	ln_sig = np.log(sig_y)
	tot = res2 + 2*ln_sig
	
	return tot.sum()

def fit_1D_type(t1,t2,pstart,bnds,x_ref):

	#Best linear model centered on log SFR = -3
	data = (t1["logSFRHa"]-x_ref, t2["logSFRHa"])

	f = lambda param: chi_square_1D_type(data, param)
	res = minimize(f, pstart, bounds = bnds)
	a, b, sig, sig_slope = res.x

	#account for the shift by x_ref for both y and x
	a += -b*x_ref
	sig += x_ref*sig_slope

	par = (a, b)
	epar = sig
	db = sig/np.sqrt(t1["logSFRHa"].size*np.var(t1["logSFRHa"]))

	print("Best-fit result:")
	print("\t a = ",'{0:.2E}'.format(a),"with xref = ",x_ref)
	print("\t b = ",'{0:.2f}'.format(b)," +/- ",'{0:.2f}'.format(db))
	print("\t sigma = ",'{0:.2f}'.format(sig))

	#Modify the SFR and adds the uncertainty
	t = copy.deepcopy(t1)  
	t["logSFRHa"] = fun_bias_1D(t1["logSFRHa"], a, b)
	dSFR = t2["logSFRHa"]-t["logSFRHa"]
	eSFR = fun_res_1D(t1["logSFRHa"], sig, sig_slope)
	t.add_column(Column(eSFR),name="e_logSFRHa")

	return t, t2, par, db, sig


#Return the result for each galaxy type - 1D
def load_type_results_1D(t1, t2, Tmin, Tmax, x_ref):
		#select only galaxies of the right type
		t1_type = t1[ (t1["T"]>=Tmin) & (t1["T"]<Tmax) ]
		t2_type = t2[ (t2["T"]>=Tmin) & (t2["T"]<Tmax) ]

		#fit galaxies of the right type
		pstart = [-2, 1, 0.7, 0.1]
		bnds = [(-5,5), (0,2), (0.01,2), (0.,0.)]
		t1_type, t2_type, par, db, epar = fit_1D_type(t1_type, t2_type,pstart,bnds,x_ref)
		text_b_val =r"$b$="+str('{0:.2f}'.format(par[1]))+r"$\pm$"+str('{0:.2f}'.format(db))

		return t1_type, t2_type, par, epar, text_b_val

###### Plot ########################################################################
def plot(t1, t2, title1, title2, eSFR_par, typeGal_bnds, text_b_val, plot_no_type=False, verbose=False):
	plt.rcParams.update({'font.size': 16})

	#Computes the statistics
	dSFR = t2["logSFRHa"] - t1["logSFRHa"]
	eSFR = t1["e_logSFRHa"]

	res = dSFR/eSFR
	chi2 = (res**2).sum()
	ndf = int(round(res.size - 4))
	pchi2 = 1 - stats.chi2.cdf(chi2, ndf)
	text_chi2 = r"$\chi^2$/ndf = "+str(round(chi2*10)/10)+" / "+str(ndf)+"\n"+"P($\chi^2$,ndf) = "+str(round(pchi2*100))+"%"
	if(verbose):
		print(text_chi2)
	ks = stats.kstest(res, 'norm', args = (0, 1))
	TS, TScrit, pcrit  = stats.anderson(res, 'norm')
	pcrit *= 1E-2
	f_interp = interpolate.interp1d(np.log(TScrit),np.log(pcrit), bounds_error=False, fill_value="extrapolate")
	ad_p = np.exp(f_interp(np.log(TS)))
	text_KS = r"K.-S. test, $p=$"+'{0:.1E}'.format(ks[1])
	text_AD = r"A.-D. test, $p=$"+'{0:.1E}'.format(ad_p)


	#Plot range and y=x line
	xmin, xmax = np.min(t1["logSFRHa"]), np.max(t1["logSFRHa"])
	x = np.linspace(xmin, xmax, 100)
	xmin, xmax = xmin-0.5, xmax+0.5

	fig = plt.subplots(figsize=(10, 8), nrows=2, ncols = 2)
	plt.tight_layout()
	plt.subplots_adjust(left = 0.08, bottom = 0.08)#, right=0.95, bottom = 0.1, top = 0.95

	plt.subplot(221)
	plt.xlim(xmin, xmax)
	plt.xlabel(title1)
	plt.ylabel(title2)
	sc = plt.scatter(t1["logSFRHa"],t2["logSFRHa"], c=t1["T"], vmin=np.min(t1["T"]), vmax=np.max(t1["T"]), alpha = 1, s=10)
	plt.plot(x, x, color="tab:blue", alpha = 0.8)
	cbar = plt.colorbar(sc)
	cbar.set_label(r'Morphology Type, $T$', rotation=270, labelpad=+20)

	plt.subplot(223)
	title3 = r"$\Delta$ log$_{10}$ SFR [M$_\odot$ /yr]"
	ymin, ymax = -2.5, 2.5
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	plt.xlabel(title1)
	plt.ylabel(title3)
	plt.plot(x, np.zeros_like(x), color="tab:blue", alpha = 0.8)
	plt.errorbar(t1["logSFRHa"],dSFR, t1["e_logSFRHa"],linestyle='',color="k",elinewidth=0.2)
	sc = plt.scatter(t1["logSFRHa"],dSFR, c=t1["T"], vmin=np.min(t1["T"]), vmax=np.max(t1["T"]), alpha = 1, s=10)
	cbar = plt.colorbar(sc)
	cbar.set_label(r'Morphology Type, $T$', rotation=270, labelpad=+20)

	plt.subplot(222)
	plt.axis('off')
	ytext, dytext = 0.95, 0.1
	plt.text(0,ytext,"Number of galaxies: "+str(dSFR.size))
	ytext-=2*dytext
	X = r"M$_*^b$"
	if("FUV" in title1):
		X = "FUV$^b$"
	plt.text(0,ytext,r"Results: SFR(H$_\alpha$) $\propto$ "+X)
	ytext-=dytext
	if not plot_no_type:
		plt.text(0,ytext,r"- $T$<"+str(typeGal_bnds[0][1])+r": $\sigma_{\rm SFR}$="+'{0:.2f}'.format(eSFR_par[0])+r"$\,$dex, "+text_b_val[0])
	else:
		plt.text(0,ytext,r"- all $T$: $\sigma_{\rm SFR}$="+'{0:.2f}'.format(eSFR_par)+r"$\,$dex, "+text_b_val)	
	ytext-=dytext
	if not plot_no_type:
		plt.text(0,ytext,r"- $T$>"+str(typeGal_bnds[2][0]-1)+r": $\sigma_{\rm SFR}$="+'{0:.2f}'.format(eSFR_par[2])+r"$\,$dex, "+text_b_val[2])
	ytext-=dytext
	if not plot_no_type:
		plt.text(0,ytext,r"- else: $\sigma_{\rm SFR}$="+'{0:.2f}'.format(eSFR_par[1])+r"$\,$dex, "+text_b_val[1])
	ytext-=2*dytext
	plt.text(0,ytext,"Normal residual distribution")
	ytext-=dytext
	plt.text(0,ytext," "+text_KS)
	ytext-=dytext
	plt.text(0,ytext," "+text_AD)


	plt.subplot(224)
	title4 = "$\Delta$ log$_{10}$ SFR / $\sigma$(log$_{10}$ SFR)"
	title5 = "" #"# entries"
	plt.xlabel(title4)
	plt.ylabel(title5)
	n, bins, patches = plt.hist(dSFR/eSFR, bins=round(dSFR.size/10))
	bin_center = np.array([0.5*(bins[i]+bins[i+1]) for i in range(bins.size-1)])
	gaus_norm = GausNorm(np.sum(n),bins[1]-bins[0])
	plt.errorbar(bin_center,n, np.sqrt(n),linestyle='',color="k",elinewidth=0.5)
	x_hist = np.linspace(bin_center[0], bin_center[-1])
	plt.plot(x_hist, gaus_norm(x_hist),linestyle='-',color="k",linewidth=0.5)



###### Specific to inclusion of Galaxy type ########################################
#Return the boundaries between various class [tmin,tmax[
def load_bounds(typeGal_bnds):
	list_bnds = []
	for i in range(typeGal_bnds.size+1):
		tmin, tmax = -np.infty, np.infty
		if i==0: 
			tmax = typeGal_bnds[i]
		elif i == typeGal_bnds.size:
			tmin = typeGal_bnds[i-1]
		else:
			tmin = typeGal_bnds[i-1]
			tmax = typeGal_bnds[i]
		list_bnds.append((tmin,tmax))
	return list_bnds


#Return the result for each galaxy type - 2D
def load_type_results_2D(t1, t2, t3, Tmin, Tmax, x_ref1, x_ref2):
		#select only galaxies of the right type
		t1_type = t1[ (t1["T"]>=Tmin) & (t1["T"]<Tmax) ]
		t2_type = t2[ (t2["T"]>=Tmin) & (t2["T"]<Tmax) ]
		t3_type = t3[ (t3["T"]>=Tmin) & (t3["T"]<Tmax) ]

		#fit galaxies of the right type
		pstart = [-2, 1, 0.2, 0.5, 0, 0]
		bnds = [(-3,-1), (0,2), (-0.5,0.5), (0.01,1), (0,0), (0,0)]
		t_type, t3_type, par, epar = fit_2D_type(t1_type,t2_type,t3_type,pstart,bnds,x_ref1,x_ref2)

		return t_type, t3_type, par, epar

###### Result without accounting for the type ######################################
def show_no_type(limit10Mpc = True):
	t_bnds_min, t_bnds_max = -np.infty, np.infty
	typeGal_bnds = [t_bnds_min, t_bnds_max]
	###### Computation: Ha vs FUV, including type ####################################
	#Load the tables containing Ha and FUV info
	t1, title1 = LoadFUV('M_Ha_FUV.dat'), r"log$_{10}$ SFR$_{\rm fit}$(FUV) [M$_\odot$ /yr]"
	t2, title2 = LoadHa('M_Ha_FUV.dat'), r"log$_{10}$ SFR(H$_\alpha$) [M$_\odot$ /yr]"

	if limit10Mpc:
		id_range = np.where(t1["D"]<=10)
		t1 = t1[id_range]
		t2 = t2[id_range]

	#Perform the fit
	print('1D FUV results')
	t1_xref_FUV = -2
	t1_FUV, t2_FUV, parFUV, eparFUV, text_b_val = load_type_results_1D(t1, t2, t_bnds_min, t_bnds_max,t1_xref_FUV)
	plot(t1_FUV, t2_FUV, title1, title2,eparFUV, typeGal_bnds, text_b_val, True)

	###### Computation: Ha vs M*, including type #####################################
	#Load the tables containing Ha and FUV info
	t1, title1 = LoadM('M_Ha_FUV.dat'), r"log$_{10}$ SFR$_{\rm fit}$(M$_*$) [M$_\odot$ /yr]"#vstack([LoadM('M_Ha_FUV.dat'),LoadM('M_Ha.dat')])
	t2, title2 = LoadHa('M_Ha_FUV.dat'), r"log$_{10}$ SFR(H$_\alpha$) [M$_\odot$ /yr]"#vstack([LoadHa('M_Ha_FUV.dat'),LoadHa('M_Ha.dat')])

	if limit10Mpc:
		id_range = np.where(t1["D"]<=10)
		t1 = t1[id_range]
		t2 = t2[id_range]

	#Perform the fit
	print('\n')
	print('1D M results')
	t1_xref_M = 9 
	t1_M, t2_M, parM, eSFRM, text_b_val = load_type_results_1D(t1, t2, t_bnds_min, t_bnds_max,t1_xref_M)
	plot(t1_M, t2_M, title1, title2, eSFRM, typeGal_bnds, text_b_val, True)


###### Result accounting for the type, possibilty to print output ##################
def show_type(output_table = False, limit10Mpc = True):

	###### Boundaries for galaxy types ###############################################
	#<0: early type
	#>=0 and <8: late type - spiral
	#>=8: late type - irregular
	Tbound0,Tbound1 = 0,8#1, 9 (worse than 0,8)#1, 8 (worse than 0,8)#0, 9 (marginally worse than 0,8)
	typeGal_bnds = load_bounds(np.array([Tbound0,Tbound1]))

	###### Computation: Ha vs FUV, including type ####################################
	#Load the tables containing Ha and FUV info
	t1, title1 = LoadFUV('M_Ha_FUV.dat'), r"log$_{10}$ SFR$_{\rm fit}$(FUV,T) [M$_\odot$ /yr]"
	t2, title2 = LoadHa('M_Ha_FUV.dat'), r"log$_{10}$ SFR(H$_\alpha$) [M$_\odot$ /yr]"

	if limit10Mpc:
		id_range = np.where(t1["D"]<=10)
		t1 = t1[id_range]
		t2 = t2[id_range]

	#Perform the fit for each type
	t1_tot, t2_tot = [], []
	SFR_par_FUV, eSFR_par_FUV = [], []
	text_b_val_FUV = []
	print('\n')
	t1_xref_FUV = -2 
	for i, t_bnds in enumerate(typeGal_bnds):
		print('1D FUV results for ntype =',i+1)
		t1_typeFUV, t2_typeFUV, parFUV, eparFUV, text_b_val = load_type_results_1D(t1, t2, t_bnds[0], t_bnds[1],t1_xref_FUV)
		t1_tot.append(t1_typeFUV)
		t2_tot.append(t2_typeFUV)
		SFR_par_FUV.append(parFUV)
		eSFR_par_FUV.append(eparFUV)
		text_b_val_FUV.append(text_b_val)

	#Group and plot
	t1_tot, t2_tot= vstack(t1_tot), vstack(t2_tot)
	plot(t1_tot, t2_tot, title1, title2,eSFR_par_FUV, typeGal_bnds, text_b_val_FUV)

	###### Computation: Ha vs M*, including type #####################################
	#Load the tables containing Ha and FUV info
	t1, title1 = LoadM('M_Ha_FUV.dat'), r"log$_{10}$ SFR$_{\rm fit}$(M$_*$,T) [M$_\odot$ /yr]"#vstack([LoadM('M_Ha_FUV.dat'),LoadM('M_Ha.dat')])
	t2, title2 = LoadHa('M_Ha_FUV.dat'), r"log$_{10}$ SFR(H$_\alpha$) [M$_\odot$ /yr]"#vstack([LoadHa('M_Ha_FUV.dat'),LoadHa('M_Ha.dat')])

	if limit10Mpc:
		id_range = np.where(t1["D"]<=10)
		t1 = t1[id_range]
		t2 = t2[id_range]

	#Perform the fit for each type
	t1_tot, t2_tot = [], []
	SFR_par_M, eSFR_par_M = [], []
	text_b_val_M = []
	print('\n')
	t1_xref_M = 9 
	for i, t_bnds in enumerate(typeGal_bnds):
		print('1D M results for ntype =',i+1)
		t1_typeM, t2_typeM, parM, eSFRM, text_b_val = load_type_results_1D(t1, t2, t_bnds[0], t_bnds[1],t1_xref_M)
		t1_tot.append(t1_typeM)
		t2_tot.append(t2_typeM)
		SFR_par_M.append(parM)
		eSFR_par_M.append(eSFRM)
		text_b_val_M.append(text_b_val)

	#Group and plot
	t1_tot, t2_tot= vstack(t1_tot), vstack(t2_tot)
	plot(t1_tot, t2_tot, title1, title2, eSFR_par_M, typeGal_bnds, text_b_val_M)


	#Creates a meta-table, adding extra colums to Karachentsev 
	#logSFR, e(logSFR), method
	#method 1: if(Ha), SFR = SFR(Ha), eSFR = eSFR(FUV,T)
	#method 2: elif(FUV), SFR = SFR(FUV,T), eSFR = eSFR(FUV,T)
	#method 3: elif(M), SFR = SFR(M,T), eSFR = eSFR(M,T)
	if(output_table):
		filename_in = "Karachentsev18_1029Gal_LV.dat"
		filename_out = "Karachentsev18_1029Gal_LV_Augmented.dat"

		t = Table.read(filename_in, format='ascii')
		t = t.filled()
		logSFR = [None] * t["Name"].size
		elogSFR = [None] * t["Name"].size
		method = [None] * t["Name"].size
		t.add_column(Column(logSFR, dtype=float, length=4),name="logSFR")
		t.add_column(Column(elogSFR, dtype=float, length=4),name="elogSFR")
		t.add_column(Column(method, dtype=float, length=4),name="method")

		def convert_to_str(a):
			res = "{0:.2f}".format(np.float(a))
			return res

		def is_number(s):
			try:
				s
				float(s)
				return True
			except ValueError:
				return False

		for i, sfrHa in enumerate(t["logSFRHa"]):
			if(t["T"][i].size>0 and is_number(t["T"][i])):
				T = np.float(t["T"][i])
				num_type = 0
				if(T>=Tbound0 and T<Tbound1):
					num_type = 1
				elif(T>=Tbound1):
					num_type = 2
				if(is_number(sfrHa)):
					t["method"][i] = 1
					t["logSFR"][i] = convert_to_str(sfrHa)
					t["elogSFR"][i] = convert_to_str(eSFR_par_FUV[num_type])
				elif(is_number(t["logSFRFUV"][i])):
					t["method"][i] = 2
					t["logSFR"][i] = convert_to_str(SFR_par_FUV[num_type][0]+ SFR_par_FUV[num_type][1]*np.float(t["logSFRFUV"][i]))
					t["elogSFR"][i] = convert_to_str(eSFR_par_FUV[num_type])
				elif(is_number(t["logM*"][i])):
					t["method"][i] = 3
					t["logSFR"][i] = convert_to_str(SFR_par_M[num_type][0]+ SFR_par_M[num_type][1]*np.float(t["logM*"][i]))
					t["elogSFR"][i] = convert_to_str(eSFR_par_M[num_type])
				else:
					print("No SFR for galaxy:",t["Name"][i])
			else:
				print("No SFR for galaxy:",t["Name"][i])


		t.write(filename_out, format = 'ascii', overwrite='True')
	else:
		#Draw the plot
		plt.show()

###### Main ########################################################################
if __name__ == "__main__":
	show_no_type()
	show_type()
