#!/usr/bin/python

import numpy as np
import scipy

#Astropy
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates.representation import CartesianRepresentation


def get_flux_distribution(method_name,astopy_coodinates,
							diffuse_flux_given,
							index_given,
							ref_energy):
    methods = {"StandardCandle": standard_candle,
    		   "LogNormal": log_normal,
    		  "Fermi-LAT_pi0": fermilatpi0}
   

    if not method_name in methods.keys():
    	raise NotImplementedError(method_name+ " Model not found ")

    if method_name=="StandardCandle":
    	return methods[method_name](astopy_coodinates,
    								diffuse_flux_given,
									index_given,
									ref_energy)
    elif method_name=="LogNormal":
    	return methods[method_name](astopy_coodinates,
    								diffuse_flux_given,
									index_given,
									ref_energy)


def standard_candle(astopy_coodinates,
					diffuse_flux_given, #TeV-1cm-2s-1
					index_given,
					ref_energy):

	#ASSUMPtiON : No redshifts are used in the caluclation.

	# Find Distance Along line of sight
	distance_array = astopy_coodinates.transform_to(coord.ICRS).distance.to(u.cm)

	diffuse_flux_given_at_ref_energy = diffuse_flux_given*((ref_energy/100)**index_given)

	sum_f_by_L_all_sources=0
	# Get the luminosity'
	try: 
		len(distance_array) #IF ERROR BECAUSE OF 1 SOURCE, THEN except condition is run where array is infact a scalar object
	except:
		sum_f_by_L_all_sources += 1/(4*np.pi*(distance_array**2))
	else:
		for los_distance in distance_array:
			sum_f_by_L_all_sources += 1/(4*np.pi*(los_distance**2))
	luminosity_per_source = diffuse_flux_given_at_ref_energy/sum_f_by_L_all_sources

	indi_flux_vals = luminosity_per_source/(4*np.pi*(distance_array**2)) # Val in TeVcm-2s-1

	indi_flux_vals_norm = indi_flux_vals/((ref_energy/100)**index_given) # Norm in TeV-1cm-2s-1

	return indi_flux_vals_norm,luminosity_per_source








### BELOW: TO BE UPDATED ##########

def log_normal(astopy_coodinates,diffuse_flux_given,nsource,stdev_sigma_L=1.4,mean_luminosity=10**(32.1)):
	"""
	From  https://arxiv.org/pdf/1705.00806.pdf
	Probability Density is given by:
	p(L) = (1/(sigma_L L sqrt(2pi))) exp(-(lnL-lnLmed)^2/(2 sigma_L^2))
	
	L=Luminosity
	L_med = Median luminosity
	ln(Lmed) =  mean of normal distribution in ln(L)
	sigma_L = standard deviation  of normal distribution in ln(L)

	Mean and Median Luminosity Values are taken from here.
	"""

	

	mean = mean_luminosity		 # mean lum is in erg/s
	logmean = np.log(mean)       # log mean luminosity
	sigma = stdev_sigma_L        # sigma is given in ln
	mu = logmean-sigma**2./2.  # log median luminosity

	try: 
		len(distance_array) #IF ERROR BECAUSE OF 1 SOURCE, THEN except condition is run where array is infact a scalar object
	except:
		nsource=1
	else:
		nsource=len(distance_array)

	rng = np.random.RandomState()
	sample_distribution = rng.lognormal(mu, sigma, nsource)
	return sample_distribution


def pdf(self, lumi):
        """ Gives the value of the PDF at lumi.

        Parameters:
            lumi: float or array-like, point where PDF is evaluated.

        Notes:
            PDF given by:
                     1                 /     (ln(x) - mu)^2   \
            -------------------- * exp | -  ----------------  |
             x sigma sqrt(2 pi)        \       2 sigma^2      /
        """
        return np.exp(self.mu) * \
            lognorm.pdf(lumi, s=self.sigma, scale=np.exp(self.mu))
def cdf(self, lumi):
        """ Gives the value of the CDF at lumi.

        Parameters:
            lumi: float or array-like, point where CDF is evaluated.

        Notes:
            CDF given by:
             1     1       /  (ln(x) - mu)^2   \
            --- + --- erf |  ----------------   |
             2     2       \   sqrt(2) sigma   /
        """
        return lognorm.cdf(lumi, s=self.sigma, scale=np.exp(self.mu))


def fermilatpi0(astopy_coodinates):
	#
	# USE HEALPY TO LOAD TEMPLATES
	#

	template = np.load("Fermi-LAT_pi0_map.npy")

	# HEALPY SETUP TAKEN FROM CSKY 

	npix = template.shape[0]
	nside = hp.npix2nside(npix) # npix = 12*nside**2
	pixarea = hp.nside2pixarea(nside) 
	# np.r_[:npix] is [0,1,2......npix]
	pix_zenith,pix_ra = hp.pix2ang(nside, np.r_[:npix]) #converts from pixel index to spherical polar coordinates;
	pix_dec = np.pi/2 - pix_zenith
	template /= template.sum() * pixarea		


	
	"""
	LOGIC USED TO DERIVE FLUXES:

	The pi0 decay template gives the decay values at a given position.
	
	We know from pp interactions:
		pp −→ Nπ[π+ + π− + π0] + X
		π+ −→ µ+ + ¯νµ
		µ+ −→ e+ +νe + ¯νµ

		π0 −→ γ + γ


	We assume the simplest case where the neutrino flux is directly proportional to the pi0 decay,
	i.e. ignoring any constants as we are just interested on how the flux is scaled as a function of distance

	The diffuse flux normalization is then used to give the per source flux.

	No Energy information is used (yet)

	"""

	#
	# GET COORDINATES OF SOURCES AND LOAD ASTROPY FORMAT
	#
	
	


	#
	# CONVERT GALACTIC ASTROPY COORDINATES TO HEALPY READABLE FORMAT
	#

	# Convert Galactic coordinates to ICRS
	# Healpy uses theta and phi, where phi=ra (radians) and dec=pi/2-theta (NOT SURE ABOUT THIS! CHECK)

	phi  = astropy_coords_in_galactic.transform_to(coord.ICRS).ra.radian
	theta = 0.5 * np.pi - astropy_coords_in_galactic.transform_to(coord.ICRS).dec.radian
	radius = 0.2 #Radius (DEG) of the circle on the map to locate point. Kept minimum

	xyz = hp.ang2vec(theta, phi)


	# IF conditions are used in case no point is found on template. So radius is increased
	flux_norm_per_source=[]
	try: 
		print("NUMBER OF SOURCES = ",len(phi))
	except:
		print("ONLY 1 SOURCE!")
		ipix_disc = hp.query_disc(nside, xyz, np.deg2rad(radius))
		if len(template[ipix_disc])==0:
			ipix_disc = hp.query_disc(nside, xyz, np.deg2rad(radius+0.1))
		if len(template[ipix_disc])==0:
			ipix_disc = hp.query_disc(nside, xyz, np.deg2rad(radius+0.2))
		flux_norm_per_source.append(template[ipix_disc][0])
	else:
		for index_indi_point in range(len(xyz)):
			ipix_disc = hp.query_disc(nside, xyz[index_indi_point], np.deg2rad(radius))
			if len(template[ipix_disc])==0:
				ipix_disc = hp.query_disc(nside, xyz[index_indi_point], np.deg2rad(radius+0.1))
			if len(template[ipix_disc])==0:
				ipix_disc = hp.query_disc(nside, xyz[index_indi_point], np.deg2rad(radius+0.2))
			flux_norm_per_source.append(template[ipix_disc][0])


	#
	# USE THE SCALING DERIVED FROM TEMPLATE TO GET INDIVIDUAL FLUXES BASED ON DIFFUSE FLUX
	#

	
	flux_norm_per_source=np.asarray(flux_norm_per_source)

	flux_norm_per_source = flux_norm_per_source*(diffuse_flux_given/flux_norm_per_source.sum())


	return 







