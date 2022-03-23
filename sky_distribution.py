#!/usr/bin/python

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib

font = {'size'   : 20}
matplotlib.rc('font', **font)

def get_model(model_name,distribution_parameters_list,r,z,make_pdf_plot_location):
    models = {"exponential": ExponentialSpatialDist,
    		  "modified_exponential": ModifiedExponentialDist} #change name to Lorimer_case_dist
    #          "radial_GDP": RadialGaussianDensityProfile,
    #          "SNR": SuperNovaRemenant,
    #          "PWN": PulsarWindNebula,
    #          }


    # NOTE:
    # distribution_parameters_list = [r_0,    -------0 index
    #								  z_0,    -------1
    #								  alpha,  -------2
    #								  beta,   -------3
    #								  h]      -------4
	#

    if not model_name in models.keys():
    	raise NotImplementedError(model_name+ " Model not found ")


    if model_name =="exponential":
    	param_used=[distribution_parameters_list[0],distribution_parameters_list[1]]
    elif model_name=="modified_exponential":
    	param_used=[distribution_parameters_list[2],distribution_parameters_list[3],distribution_parameters_list[4]]
    else:
    	print("Wrong Model name; Should not get this message if notIMPLEMENTED ERROR code is working, Please check code")
    	exit()

    if make_pdf_plot_location!=None:
    	x_arr_pdf_r = np.arange(0.001,15,0.1)
    	x_arr_pdf_z = np.arange(0.001,3,0.1)

    	y_arr_pdf_r = []
    	y_arr_pdf_z = []
    	for r_pdf_val in x_arr_pdf_r:
    		y_arr_pdf_r.append(models[model_name](param_used,r_pdf_val,0))
    	for z_pdf_val in x_arr_pdf_z:
    		y_arr_pdf_z.append(models[model_name](param_used,0,z_pdf_val))

    	fig, ax = plt.subplots(1,2,figsize=(18,9),dpi=100)

    	ax[0].plot(x_arr_pdf_r,y_arr_pdf_r)
    	ax[0].set_xlabel('Radial distance (r) kpc')
    	ax[0].set_ylabel('PDF')

    	ax[1].plot(x_arr_pdf_z,y_arr_pdf_z)
    	ax[1].set_xlabel('Vertical Height (|z|) kpc')
    	ax[1].set_ylabel('PDF')

    	plt.savefig(make_pdf_plot_location+"pdf_used_for_distribution_%s.png"%(model_name),bbox_inches="tight")

    	return
    else:
    	return models[model_name](param_used,r,z)
    




def ExponentialSpatialDist(param_used,r,z): # param_used = r_0,z_0
	"""Simple model where exponential functions are used for distribution
	   See Eq 1 of https://iopscience.iop.org/article/10.3847/2041-8205/832/1/L6/pdf

	Note that: dN/dV is prop to e^r/r_0 e^z/z0
	When dN/dr is caluclated,  a factor of r is added
	So in norm_r also a factor of maximum_pt is added

	"""
	#print("BASIC EXPONENTIAL MODEL IS USED")
	r_0 = param_used[0]
	z_0 = param_used[1]
	r_solar=8.5
	z_solar=0.015

	
	maximum_pt = r_0 # Found Using derivatve

	
	norm_r     = 1/(maximum_pt*np.exp(-1.*(maximum_pt/r_0)))
	
	
	exp_dist = norm_r * r * np.exp(-1.*r/r_0)
	norm_z     = 1 
	exp_z    = norm_z * np.exp(-1.* abs(z)/z_0)

	if r==0:
		return (exp_z)
	elif z==0:
		return (exp_dist)
	else:
		return (exp_z*exp_dist)

def ModifiedExponentialDist(param_used,r,z): # param_used = alpha,beta,h

							
	"""Derived using Equation 7 of 
	   https://arxiv.org/pdf/1505.03156.pdf
	   which can be modified based on class of sources (SNR/PWN)
	Note that: dN/dV is prop to (r/r_s)^alpha e^-beta(r-r_s/r_s) e^z/z0
	When dN/dr is caluclated,  a factor of r is added
	So in norm_r also a factor of maximum_pt is added

	"""
	#print("IMPROVED EXPONENTIAL MODEL IS USED: \n can be modified based on class of sources (SNR/PWN) ")
	alpha = param_used[0]
	beta  = param_used[1]
	h     = param_used[2]

	r_solar=8.5
	z_solar=0.015
	maximum_pt = (alpha+1)*r_solar/beta # Found Using derivatve
	
	norm_r     = 1/(maximum_pt*((maximum_pt/r_solar)**alpha)*np.exp(-1.*beta*((maximum_pt-r_solar)/r_solar)))
	
	exp_dist = norm_r * (r*(r/r_solar)**alpha)*np.exp(-1.*beta*((r-r_solar)/r_solar))
	

	norm_z     = 1 
	exp_z    = norm_z*np.exp(-1.* abs(z)/h)

	if r==0:
		return (exp_z)
	elif z==0:
		return (exp_dist)
	else:
		return (exp_z*exp_dist)





