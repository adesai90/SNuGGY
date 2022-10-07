#!/usr/bin/python

import numpy as np
import scipy

#Astropy
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates.representation import CartesianRepresentation


def get_gamma_flux_distribution(method_name,astopy_coodinates,
							diffuse_flux_given,
							index_given,
							ref_energy):
    methods = {"gamma_StandardCandle": gamma_standard_candle,
    		   "gamma_LogNormal": gamma_log_normal}
   

    if not method_name in methods.keys():
    	raise NotImplementedError(method_name+ " Model not found ")

    if method_name=="gamma_StandardCandle":
    	return methods[method_name](astopy_coodinates,
    								gamma_diffuse_flux_given,
									index_given,
									ref_energy)
    elif method_name=="gamma_LogNormal":
    	return methods[method_name](astopy_coodinates,
    								gamma_diffuse_flux_given,
									index_given,
									ref_energy)

def get_gamma_from_nu(astropy_coords_in_galactic,
					  simulated_nu_fluxes,
					  ref_energy,
					  pp_or_pgamma):
	"""
	This method will make use of the neutrino-gamma ray equation, as described by eq 3 of https://arxiv.org/pdf/1805.11112.pdf
	given as:
	1/3sum(Ev^2 Qv(Ev)) ~ (Kpi/4) * [Eg^2 Qg(Eg)]Eg=2Ev
	"""

	E_nu = ref_energy
	E_gamma = ref_energy/2.

	if pp_or_pgamma=="pp":
		Kpi = 2.
	elif pp_or_pgamma=="pgamma":
		Kpi = 1.
	else:
		print("ERROR!!!! Give pp or pgamma as string")
		exit()


	E2Qv = E_nu * E_nu * simulated_nu_fluxes

	E2Qgamma = (1./3.) * E2Qv * (4./Kpi)

	flux_at_Egamma = E2Qgamma/(E_gamma*E_gamma)

	return flux_at_Egamma # Not that this is the flux at ref_energy/2 so 50 TeV if ref_energy is 100 TeV



def gamma_standard_candle(astopy_coodinates,
					gamma_diffuse_flux_given, #TeV-1cm-2s-1
					index_given,
					ref_energy):

	### UPDATE FOR GAMMA BELOW

	""" ASSUMPTION : No redshifts are used in the calculation.
		This definition ensures that the standard candle luminosity is the same for any simulation of N number of sources.
		SC Luminosity only depends on the total diffuse flux and the number of sources and not on the position of the sources
		The total flux from these sources may or may not be exactly equal to the diffuse flux.

		This is an update of the standard_candle_forced definition.
		N
	""" 
	E0 = ref_energy

	# Find Distances of simulated sources Along line of sight
	distance_array = (astopy_coodinates.transform_to(coord.ICRS).distance.to(u.cm)).value
	# Given the total flux, find individual flux contribution
	indi_flux_contribution = diffuse_flux_given /len(distance_array) # in TeV-1cm-2s-1 

	# Now convert this to get the luminosity per source
	mean_distance = 2.469e+22 # 8 kpc in cm, close to peak in distributions 
	

	luminosity_per_source = indi_flux_contribution * energy_integral_with_index(index_given,E0_ref=E0) * (4*np.pi*(mean_distance**2)) * 1.60218 # TeV/s -> erg/s

	all_lum_d = 1/(4*np.pi*(distance_array**2))
	del distance_array
	
	indi_flux_vals = luminosity_per_source*all_lum_d / (energy_integral_with_index(index_given,E0_ref=E0) * 1.60218) # Val in TeV-1cm-2s-1

	return np.asarray(indi_flux_vals),np.asarray(float('{:0.3e}'.format(luminosity_per_source)))


