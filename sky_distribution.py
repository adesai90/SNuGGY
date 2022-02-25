#!/usr/bin/python

import numpy as np
import scipy


def get_model(model_name,distribution_parameters_list,r,z):
    models = {"exponential": ExponentialSpatialDist,
    		  "modified_exponential": ModifiedExponentialDist}
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
    	return models[model_name](distribution_parameters_list[0],distribution_parameters_list[1],r,z)
    elif model_name=="modified_exponential":
    	return models[model_name](distribution_parameters_list[2],distribution_parameters_list[3],distribution_parameters_list[4],r,z)
    else:
    	return




def ExponentialSpatialDist(r_0,z_0,r,z):
	"""Simple model where exponential functions are used for distribution
	   See Eq 1 of https://iopscience.iop.org/article/10.3847/2041-8205/832/1/L6/pdf
	"""
	#print("BASIC EXPONENTIAL MODEL IS USED")
	exp_dist = np.exp(-1.*r/r_0)
	exp_z    = np.exp(-1.* abs(z)/z_0)
	norm     = 1. 

	return (norm*exp_z*exp_dist)

def ModifiedExponentialDist(alpha,
							beta,
							h,
							r,
							z):
	"""Derived using Equation 7 of 
	   https://arxiv.org/pdf/1505.03156.pdf
	   which can be modified based on class of sources (SNR/PWN)
	"""
	#print("IMPROVED EXPONENTIAL MODEL IS USED: \n can be modified based on class of sources (SNR/PWN) ")
	r_solar=8.5
	exp_dist = ((r/r_solar)**alpha)*np.exp(-1.*beta*((r-r_solar)/r_solar))
	exp_z    = np.exp(-1.* abs(z)/h)
	norm     = 1. # simple case rho_0=1 instead of ~0.9
	if r!=0:
		return (norm*exp_z*exp_dist)
	else:
		return (norm*exp_z)







