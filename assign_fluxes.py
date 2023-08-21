#!/usr/bin/python

import numpy as np
import scipy

#Astropy
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates.representation import CartesianRepresentation
from sampling	import	InverseCDF
import matplotlib.pyplot as plt
import matplotlib

font = {'size'   : 20}
matplotlib.rc('font', **font)
"""
First two definitions are used to get neutrino or gamma ray fluxes provided the other is known

Next defintions estimate the flux for the case of different luminosity function cases.
This can be used for both neutrinos and gamma rays

"""
def get_gamma_from_nu(astropy_coords_in_galactic,
					  simulated_nu_fluxes,
					  nu_ref_energy, #Nu ref energy
					  index_given,
					  pp_or_pgamma):
	"""
	This method will make use of the neutrino-gamma ray equation, as described by eq 3 of https://arxiv.org/pdf/1805.11112.pdf
	given as:
	1/3sum(Ev^2 Qv(Ev)) ~ (Kpi/4) * [Eg^2 Qg(Eg)]Eg=2Ev
	"""
	E_nu = nu_ref_energy
	E_gamma = nu_ref_energy*2.
	if pp_or_pgamma=="pp":
		Kpi = 2.
	elif pp_or_pgamma=="pgamma":
		Kpi = 1.
	else:
		print("ERROR!!!! Give pp or pgamma as string")
		exit()
	E2Qv = E_nu * E_nu * simulated_nu_fluxes #neutrino E^2*flux
	E2Qgamma = (1./3.) * E2Qv * (4./Kpi)
	flux_at_Egamma = E2Qgamma/(E_gamma*E_gamma)
	# Find Distances of simulated sources Along line of sight
	distance_array = (astropy_coords_in_galactic.transform_to(coord.ICRS).distance.to(u.cm)).value
	# Now convert this to get the luminosity per source
	mean_distance = 2.469e+22 # 8 kpc in cm, close to peak in distributions 
	luminosity_per_source = flux_at_Egamma * energy_integral_with_index(index_given,E0_ref=E_gamma) * (4*np.pi*(mean_distance**2)) * 1.60218 # TeV/s -> erg/s
	return flux_at_Egamma,np.asarray(luminosity_per_source) # Note that this is the flux at ref_energy/2 so 50 TeV if ref_energy is 100 TeV


def get_nu_from_gamma(astropy_coords_in_galactic,
					  simulated_gamma_fluxes,
					  gamma_ray_ref_energy, #gamma ref energy
					  index_given,
					  pp_or_pgamma):
	"""
	This method will make use of the neutrino-gamma ray equation, as described by eq 3 of https://arxiv.org/pdf/1805.11112.pdf
	given as:
	1/3sum(Ev^2 Qv(Ev)) ~ (Kpi/4) * [Eg^2 Qg(Eg)]Eg=2Ev
	"""
	E_nu = gamma_ray_ref_energy/2.
	E_gamma = gamma_ray_ref_energy
	if pp_or_pgamma=="pp":
		Kpi = 2.
	elif pp_or_pgamma=="pgamma":
		Kpi = 1.
	else:
		print("ERROR!!!! Give pp or pgamma as string")
		exit()
	E2Qgamma = E_gamma * E_gamma * simulated_gamma_fluxes
	E2Qv = (3./1.) * (Kpi/4.) * E2Qgamma 
	flux_at_Enu = E2Qv/(E_nu * E_nu)
	# Find Distances of simulated sources Along line of sight
	distance_array = (astropy_coords_in_galactic.transform_to(coord.ICRS).distance.to(u.cm)).value
	# Now convert this to get the luminosity per source
	mean_distance = 2.469e+22 # 8 kpc in cm, close to peak in distributions 
	luminosity_per_source = flux_at_Enu * energy_integral_with_index(index_given,E0_ref=E_gamma) * (4*np.pi*(mean_distance**2)) * 1.60218 # TeV/s -> erg/s
	return flux_at_Enu,np.asarray(luminosity_per_source) # Note that this is the flux at ref_energy/2 so 50 TeV if ref_energy is 100 TeV


"""
Now comes the defintions which make use of different luminosty functions
"""


def get_flux_distribution(method_name,astopy_coodinates,
							diffuse_flux_given,
							index_given,
							ref_energy,
							mean_luminosity,
							stdev_sigma_L,
							energy_range_low,
							energy_range_high):
    methods = {"StandardCandle": standard_candle,
    		"Forced_standardCandle": standard_candle_forced, #sum of simulated fluex is exactly equal to diffuse flux
    		"LogNormal": log_normal}
    #"observed_sample": source_count_obsevred}
   

    if not method_name in methods.keys():
    	raise NotImplementedError(method_name+ " Model not found ")

    if method_name=="StandardCandle":
    	return methods[method_name](astopy_coodinates,
    								diffuse_flux_given,
									index_given,
									ref_energy,
									energy_range_low,
									energy_range_high)
    elif method_name=="Forced_standardCandle":
    	return methods[method_name](astopy_coodinates,
    								diffuse_flux_given,
									index_given,
									ref_energy,
									energy_range_low,
									energy_range_high)
    elif method_name=="LogNormal":
    	return methods[method_name](astopy_coodinates,
    								diffuse_flux_given,
									index_given,
									ref_energy,
									mean_luminosity,
									stdev_sigma_L,
									energy_range_low,
									energy_range_high)


def energy_integral_with_index(index_given,
						   emin=1e1, #TeV
						   emax=1e4, #TeV
						   E0_ref=1e2): 
	""" 
	integal_{emin}^{emax} E*(E/E0)^(-index) dE
	"""
	index_val = abs(index_given) # index_given is negative so index is positive

	if index_val != 2.0:
		integral = (emax**(2-index_val)-emin**(2-index_val)) / (2-index_val)
	else:
		integral = np.log(emax) - np.log(emin)
	return E0_ref**index_val *integral # E0 is in denominator so its 1/E^(-index_val) 


def standard_candle(astopy_coodinates,
					diffuse_flux_given, #TeV-1cm-2s-1
					index_given,
					ref_energy,
					energy_range_low,
					energy_range_high):

	""" ASSUMPTION : No redshifts are used in the calculation.
		This definition ensures that the standard candle luminosity is the same for any simulation of N number of sources.
		SC Luminosity only depends on the total diffuse flux and the number of sources and not on the position of the sources
		The total flux from these sources may or may not be exactly equal to the diffuse flux.

		This is an update of the standard_candle_forced definition.
		
	""" 
	E0 = ref_energy

	# Find Distances of simulated sources Along line of sight
	distance_array = (astopy_coodinates.transform_to(coord.ICRS).distance.to(u.cm)).value
	# Given the total flux, find individual flux contribution
	indi_flux_contribution = diffuse_flux_given /len(distance_array) # in TeV-1cm-2s-1 

	# Now convert this to get the luminosity per source
	print()
	mean_distance = 1.543e+22 # 5 kpc in cm, close to peak in distributions 
	

	luminosity_per_source = indi_flux_contribution * energy_integral_with_index(index_given,E0_ref=E0,emin=energy_range_low,emax=energy_range_high) * (4*np.pi*(mean_distance**2)) * 1.60218 # TeV/s -> erg/s

	all_lum_d = 1/(4*np.pi*(distance_array**2))
	del distance_array
	
	indi_flux_vals = luminosity_per_source*all_lum_d / (energy_integral_with_index(index_given,E0_ref=E0,emin=energy_range_low,emax=energy_range_high) * 1.60218) # Val in TeV-1cm-2s-1

	return np.asarray(indi_flux_vals),np.asarray(float('{:0.3e}'.format(luminosity_per_source)))

def standard_candle_forced(astopy_coodinates,
					diffuse_flux_given, #TeV-1cm-2s-1
					index_given,
					ref_energy,
					energy_range_low,
					energy_range_high):

	# ASSUMPtiON : No redshifts are used in the caluclation.
	# The code forces the simulation to ensure that the sum of fluxes is exactly equalt to the diffuse flux
	# Thus, Simulating same number of sources multiple times will give diffrerent standard candle luminosities

	# Find Distance Along line of sight
	print("Using Forced Standard Candle Approach")
	distance_array = (astopy_coodinates.transform_to(coord.ICRS).distance.to(u.cm)).value
	all_lum_d = 1/(4*np.pi*(distance_array**2))
	del distance_array
	#Should be the same as we give E0 as reference enrgy and diffuse flux at that energy as input
	diffuse_flux_given_at_ref_energy = diffuse_flux_given*((ref_energy/ref_energy)**index_given)

	sum_f_by_L_all_sources=0
	
	# Get the luminosity'
	try: 
		len(all_lum_d) #IF ERROR BECAUSE OF 1 SOURCE, THEN except condition is run where array is infact a scalar object
	except:
		sum_f_by_L_all_sources += all_lum_d
	else:
		sum_f_by_L_all_sources = all_lum_d.sum()
	luminosity_per_source = diffuse_flux_given_at_ref_energy* energy_integral_with_index(index_given,E0_ref=ref_energy,emin=energy_range_low,emax=energy_range_high)* 1.60218/sum_f_by_L_all_sources


	indi_flux_vals = luminosity_per_source*all_lum_d # Val in TeVcm-2s-1

	indi_flux_vals_norm = indi_flux_vals/(energy_integral_with_index(index_given,E0_ref=ref_energy,emin=energy_range_low,emax=energy_range_high)* 1.60218) # Norm in TeV-1cm-2s-1

	return np.asarray(indi_flux_vals_norm),np.asarray(luminosity_per_source)









def pdf_fuction_for_ln(L,
						L_mean,
						sigma_L):
	# If Taking ln() based on 2.1 of https://arxiv.org/pdf/1705.00806.pdf you need to give MEDIAN luminosity
	# We have mean so taking log() similar to 4.2 of https://arxiv.org/pdf/2112.09699.pdf
	# Format for reference when luminosities are in non-log units (e.g 1e25): 
	# (np.log10(np.exp(1))/(sigma_L*L*np.sqrt(2*np.pi)))*np.exp(-((np.log10(L)-np.log10(L_mean))**2)/(2*(sigma_L**2)))
	if L_mean-100>0:
		print("Make sure log-luminosities are given")
		return
	else:
		#return (np.log10(np.exp(1))/(sigma_L*(10**L)*np.sqrt(2*np.pi)))*np.exp(-((L-L_mean)**2)/(2*(sigma_L**2)))
		return ((1/(sigma_L*(10**L)*np.sqrt(2*np.pi)))*np.exp(-((np.log(10**L)-np.log(10**L_mean))**2)/(2*(sigma_L**2))))


def log_normal(astopy_coodinates,
				diffuse_flux_given,
				index_given,
				ref_energy,
				mean_luminosity,
				stdev_sigma_L,
				energy_range_low,
				energy_range_high):
	"""
	From  https://arxiv.org/pdf/1705.00806.pdf
	and https://arxiv.org/pdf/2112.09699.pdf
	Probability Density is given by:
	p(L) = (log10(e)/(sigma_L L sqrt(2pi))) exp(-(log10(L)-log10(Lmean))^2/(2 sigma_L^2))
	
	
	L_mean = mean luminosity
	If mean luminosity is not given it is derived using the diffuse flux.  
	This is done by first finding the SC luminosity and then using that value as the mean to get the log normal distribution.

	Be Careful as the energy range of interst here is really high probably for eg. 100 TeV for neutrinos and 50 TeV for gamma-rays
	So luminosity values might be lower at those reference energies. Change the reference energy based on luminosity value
	"""

	#Find Distance Along line of sight

	print("Input Values are: Index - %.2e \n E0 - %.2e \n mean_luminosity %s \n stdev_sigma_L %.2f Emin,Emax-%.2f,%.2f"%(index_given,
				ref_energy,
				mean_luminosity,
				stdev_sigma_L,
				energy_range_low,
				energy_range_high))

	distance_array = (astopy_coodinates.transform_to(coord.ICRS).distance.to(u.cm)).value
	all_lum_d = 1/(4*np.pi*(distance_array**2))

	if mean_luminosity==None:
		print("Getting SC luminosity from diffuse flux")
		
		#Diff flux Should be the same as we give E0 as reference enrgy and diffuse flux at that energy as input
		diffuse_flux_given_at_ref_energy = diffuse_flux_given*((ref_energy/ref_energy)**index_given)
		
		sum_f_by_L_all_sources=0
		# Get the luminosity'
		try: 
				len(all_lum_d) #IF ERROR BECAUSE OF 1 SOURCE, THEN except condition is run where array is infact a scalar object
		except:
				sum_f_by_L_all_sources += all_lum_d
		else:
				sum_f_by_L_all_sources = all_lum_d.sum()

		mean_luminosity = diffuse_flux_given_at_ref_energy* energy_integral_with_index(index_given,E0_ref=ref_energy,emin=energy_range_low,emax=energy_range_high)* 1.60218/sum_f_by_L_all_sources
		print("Derived luminosity of: ",mean_luminosity)
	

	L_mean = np.log10(mean_luminosity)      # log mead luminosity
	sigma_L = stdev_sigma_L        # sigma is given in ln
	

	
	log_bins = np.arange(L_mean-20,L_mean+20,(2*L_mean)/1000) #Simulating 1000 points

	pdf_lognorm = pdf_fuction_for_ln(log_bins,L_mean,sigma_L)

	pdf_lognorm = pdf_lognorm/np.sum(pdf_lognorm)


	
	invCDF_log_lum	=	InverseCDF(log_bins,	pdf_lognorm)

	try: 
		len(distance_array) #IF ERROR BECAUSE OF 1 SOURCE, THEN except condition is run where array is infact a scalar object
	except:
		nsource=1
	else:
		nsource=len(distance_array)


	rng = np.random.RandomState()
	rng_arr = rng.uniform(0,	1,size=nsource)
	sampled_luminosities = 10**invCDF_log_lum(rng_arr) 
	### sigma_L can cause a shift, so make sure mean is reproduced.
	# We used median above but given value is mean so correct for that
	difference_from_mean =np.mean(sampled_luminosities)/10**L_mean
	sampled_luminosities_corr = sampled_luminosities/difference_from_mean
	selected_lum =sampled_luminosities_corr # Here the log luminosities will be selected so they are converted back to luminosities

	# Now we have the log luminosities, we need to convert them to fluxes.

	indi_flux_vals = np.asarray(selected_lum)*np.asarray(all_lum_d)/ (energy_integral_with_index(index_given,E0_ref=ref_energy,emin=energy_range_low,emax=energy_range_high)* 1.60218) # Val in TeV-1cm-2s-1 

	
	return np.asarray(indi_flux_vals),np.asarray(selected_lum)


