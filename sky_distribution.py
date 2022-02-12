#!/usr/bin/python

import numpy as np
import scipy


def get_model(model_name,r_0,z_0,r,z):
    models = {"exponential": ExponentialSpatialDist}
    		  
    #          "radial_GDP": RadialGaussianDensityProfile,
    #          "SNR": SuperNovaRemenant,
    #          "PWN": PulsarWindNebula,
    #          }

    if not model_name in models.keys():
    	raise NotImplementedError(model_name+ " Model not found ")


    return models[model_name](r_0,z_0,r,z)




def ExponentialSpatialDist(r_0,z_0,r,z):
	"""Simple model where exponential functions are used for distribution
	   See Eq 1 of https://iopscience.iop.org/article/10.3847/2041-8205/832/1/L6/pdf
	"""
	exp_dist = np.exp(-1.*r/r_0)
	exp_z    = np.exp(-1.* abs(z)/z_0)
	norm     = 1.

	return (norm*exp_z*exp_dist)







