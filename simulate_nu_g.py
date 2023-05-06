#!/usr/bin/python
# -*- coding: utf-8 -*-
#	General	imports
import	os,sys
import	argparse
#	Numpy	/	Scipy  / Matplotlib /healpy
import	numpy	as	np
#Astropy
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates.representation import CartesianRepresentation
# Definitions from SNuGGY
from sky_distribution import get_model
from sampling	import	InverseCDF
from assign_fluxes import *
from plotter import *

def Write2File(msg,logfile):

    if logfile!="":
        log = open(logfile, "a")
        log.write(msg+"\n")
    log.close

    return

def convert_to_galactic(r_conv,z_conv,theta_conv):
	"""
	r is radial distance
	z is vertical distance
	theta is azimuth angle
	"""
	c = coord.CylindricalRepresentation(rho=r_conv * u.kpc,phi=theta_conv * u.radian,z=z_conv * u.kpc)
	c2= c.represent_as(CartesianRepresentation)

	c3=coord.Galactocentric(x=c2.x,y=c2.y,z=c2.z)
	del c,c2

	transformed_coord = c3.transform_to(coord.Galactic)

	del c3

	return [transformed_coord,[np.asarray(transformed_coord.l).astype(np.float16),np.asarray(transformed_coord.b).astype(np.float16),np.asarray(transformed_coord.distance).astype(np.float16)]]
	

def	simulate_positions(output_file= None,
			distribution_model    ="exponential",
			number_sources_used	  =	1000,
			seed                  =	None,
			plot_dir              = None, #to make plot in xyz
			plot_aitoff_dir_icrs  = None,
			plot_aitoff_dir_gal   = None,
			filename              = None,
			make_pdf_plot_location= None, # To make pdf plots
			# LIMITS Below, generally set to high vals for inf
			z_max	              = 300.0,  #kpc
			z_min	              = 1e-06, #kpc
			r_max		          =	1000000.0, #kpc
			r_min	              =	0.001,  #kpc
			# r_0 z_0 for distributions made using simple exponential
			z_0		              =	0.6,  #kpc
			r_0	                  =	3.0,   #kpc
			# alpha beta h for distributions made using modified exponential ,fixed to SNR as default
			alpha	              = 2,
			beta	              = 3.53,
			h	                  = 0.181, #kpc
			#number of bins
			bins                  = 1000):
	

	rng = np.random.RandomState(seed)

	distribution_parameters_list = [r_0,z_0,alpha,beta,h,r_min,r_max,z_min,z_max]

	if make_pdf_plot_location!=None:
		get_model(distribution_model,distribution_parameters_list,0,1,make_pdf_plot_location)
		make_pdf_plot_location=None
	
	if make_pdf_plot_location!=None:
		print("Something went wrong")
	
	# CONVERTING 2D PDF TO 1D TO GET SAMPLES

	binning_used_z = np.geomspace(z_min,z_max,num=bins)
	binning_used_r = np.geomspace(r_min,r_max,num=bins)

	vertical_height_z_pdf = get_model(distribution_model,distribution_parameters_list,0,binning_used_z,make_pdf_plot_location)
	distance_pdf=	2*np.pi*get_model(distribution_model,distribution_parameters_list,binning_used_r,0,make_pdf_plot_location)


	invCDF_vertical_height_z	=	InverseCDF(binning_used_z,	vertical_height_z_pdf)

	invCDF_distance_r	=	InverseCDF(binning_used_r,	distance_pdf)


	# CREATE R and z BINS TO SAMPLE FROM
	distance_bins	=	np.logspace(-5,	np.log10(r_max),	bins)
	vertical_height_z_bins	=	np.arange(z_min,	z_max,	(z_max-z_min)/float(bins))

	
	
	rng = np.random.RandomState(seed)
	
	random_val_used1	=	rng.uniform(0,	1,size=number_sources_used)
	selected_z	=	invCDF_vertical_height_z(random_val_used1)
	selected_r	=	invCDF_distance_r(random_val_used1)
	del random_val_used1
	np.random.shuffle(selected_r)
	np.random.shuffle(selected_z)
	selected_z[:int(number_sources_used/2)]=-1*selected_z[:int(number_sources_used/2)]
	np.random.shuffle(selected_z)
	

	#
	# GETTING RANDOM PHI (AZIMUTH ANGLE) VALUES AND THEN PERFORM CORRD CONVERSION
	#
	selected_angles = np.random.uniform(0,2*np.pi,len(selected_r))
	
	array_coords_in_Galactocentric = [selected_r,selected_z,selected_angles] #kpc kpc rad
	coord_conversion = convert_to_galactic(selected_r,selected_z,selected_angles)
	astropy_coords_in_galactic = coord_conversion[0]
	array_coords_in_galactic = coord_conversion[1]

	
	del coord_conversion

	# PLOTTING SECTIONS, REPLACE NONE BY PATHS WHILE CALLING DEF TO PLOT.
	if plot_dir!=None:
		if os.path.isdir(plot_dir)!=True:
			os.mkdir(plot_dir)
			plot_dir
	if plot_dir!= None:
		make_3d_plot(astropy_coords_in_galactic,plot_dir)
	if plot_aitoff_dir_gal!= None:
		make_gal_aitoff_plot(astropy_coords_in_galactic,plot_aitoff_dir_gal)
	if plot_aitoff_dir_icrs!= None:
		make_icrs_aitoff_plot(astropy_coords_in_galactic,plot_aitoff_dir_icrs)
	


	if filename != None and output_file!=None:
		output_file_full_path = output_file+filename
	elif output_file!=None and filename == None:
		output_file_full_path = output_file+"simulated_%s_source_coordinates.npz"%(number_sources_used)
	else:
		output_file_full_path = None

	if output_file_full_path!=None:
		print("Saving output toe file: ",output_file)
		print("\n Please Note that the output will be saved as:")
		print("Galactocentric Coords (r,phi,z) then Galactic Coords (l deg,b deg,d kpc)")

		line="#r_galcen phi_galcen z_galcen l_gal b_gal d_gal"
		Write2File(line,output_file_full_path)

		
		
		if os.path.isfile(output_file_full_path)==True:
			os.remove(output_file_full_path)

		
		
		np.savez_compressed(output_file_full_path,r=selected_r,
									  phi=selected_angles,
									  z=selected_z,
									  gal_l=astropy_coords_in_galactic.l.deg,
									  gal_b=astropy_coords_in_galactic.b.deg,
									  gal_d=astropy_coords_in_galactic.distance.kpc)

	
	return [array_coords_in_Galactocentric,array_coords_in_galactic,astropy_coords_in_galactic]




def	Get_flux_from_positions(galcentric_coords_r_phi_z   = None,
							method_used               = "StandardCandle", #Neutrino
							gamma_ray_method_used     = "StandardCandle", #Gamma_ray
							diffuse_flux_given          = 2.14e-15, # Tev-1cm-2s-1 Neutrino Isotropic flux
							diffuse_gamma_flux_given    = 2.14e-15, # Tev-1cm-2s-1 Gamma_ray Isotropic flux
							print_output                = False,
							full_path                   = None, #"./ default.npy",
							index_given                 = 2.7, #Neutrino
							index_gamma_given           = 2.7, #Gamma_ray
							ref_energy                  = 100.0, #NEUTRINO reference energy
							simulate_gamma_ray_frm_nu	= False,
							simulate_nu_frm_gamma       = False,
							pp_or_pgamma				= "pp",
							median_luminosity_nu        = None,
							median_luminosity_gamma     = None,
							stdev_sigma_L=1.): #TeV

	nu_ref_energy = ref_energy
	gamma_ray_ref_energy = nu_ref_energy/2.

	print("Reference energy:\n Neutrino:%s TeV \n Gamma-rays:%s TeV"%(nu_ref_energy,gamma_ray_ref_energy))

	if simulate_gamma_ray_frm_nu==True and simulate_nu_frm_gamma==True:
		print("Both  simulate_gamma_ray_frm_nu and simulate_nu_frm_gamma cannot be TRUE!!!") 
		exit()
	elif simulate_gamma_ray_frm_nu==True:
		print("This method computes neutrino flux and then converts it to gamma ray fluxes based on neutrino gamma ray relation.")

		index_given=index_given*(-1.0)

		if galcentric_coords_r_phi_z==None:
			print("Error: Give Source Positions in r,z,phi Coordinates for ARRAY FORMAT")
			exit()

		astropy_coords_in_galactic = convert_to_galactic(galcentric_coords_r_phi_z[0],galcentric_coords_r_phi_z[1],galcentric_coords_r_phi_z[2])[0]
	
    

		array_l=np.asarray(astropy_coords_in_galactic.l.deg).astype(np.float16)
		array_b=np.asarray(astropy_coords_in_galactic.b.deg).astype(np.float16)
		array_distance=np.asarray(astropy_coords_in_galactic.distance.kpc).astype(np.float16)


		nu_fluxes,nu_luminosity = get_flux_distribution(method_used,astropy_coords_in_galactic,
																diffuse_flux_given, #TeV-1cm-2s-1
																index_given,
																nu_ref_energy,
																median_luminosity_nu,
																stdev_sigma_L)
		

		
		# NEXT PORTION GIVES GAMMA RAY FLUXES WITH LUMINOSITY PER SOURCE AT Ref energy = ref_energy/2
		gamma_fluxes, gamma_luminosity = get_gamma_from_nu(astropy_coords_in_galactic,
											nu_fluxes,
											nu_ref_energy,
											index_given,
											pp_or_pgamma)
				
	elif simulate_nu_frm_gamma==True:
		print("This method computes Gamma ray flux and then converts it to neutrino fluxes based on neutrino gamma ray relation.")

		index_gamma_given=index_gamma_given*(-1.0)

		if galcentric_coords_r_phi_z==None:
			print("Error: Give Source Positions in r,z,phi Coordinates for ARRAY FORMAT")
			exit()

		astropy_coords_in_galactic = convert_to_galactic(galcentric_coords_r_phi_z[0],galcentric_coords_r_phi_z[1],galcentric_coords_r_phi_z[2])[0]
	
   
		array_l=np.asarray(astropy_coords_in_galactic.l.deg).astype(np.float16)
		array_b=np.asarray(astropy_coords_in_galactic.b.deg).astype(np.float16)
		array_distance=np.asarray(astropy_coords_in_galactic.distance.kpc).astype(np.float16)


		gamma_fluxes, gamma_luminosity = get_flux_distribution(gamma_ray_method_used,astropy_coords_in_galactic,
																diffuse_gamma_flux_given, #TeV-1cm-2s-1
																index_gamma_given,
																gamma_ray_ref_energy,
																median_luminosity_gamma,
																stdev_sigma_L)


		
		# NEXT PORTION GIVES GAMMA RAY FLUXES WITH LUMINOSITY PER SOURCE AT Ref energy = ref_energy/2
		nu_fluxes,nu_luminosity = get_nu_from_gamma(astropy_coords_in_galactic,
											gamma_fluxes,
											gamma_ray_ref_energy,
											index_given,
											pp_or_pgamma)

	else:
		print("This method computes Gamma ray flux and neutrino fluxes seperately.")

		index_gamma_given=index_gamma_given*(-1.0)

		if galcentric_coords_r_phi_z==None:
			print("Error: Give Source Positions in r,z,phi Coordinates for ARRAY FORMAT")
			exit()

		astropy_coords_in_galactic = convert_to_galactic(galcentric_coords_r_phi_z[0],galcentric_coords_r_phi_z[1],galcentric_coords_r_phi_z[2])[0]
	
   
		array_l=np.asarray(astropy_coords_in_galactic.l.deg).astype(np.float16)
		array_b=np.asarray(astropy_coords_in_galactic.b.deg).astype(np.float16)
		array_distance=np.asarray(astropy_coords_in_galactic.distance.kpc).astype(np.float16)


		gamma_fluxes, gamma_luminosity = get_flux_distribution(gamma_ray_method_used,astropy_coords_in_galactic,
																diffuse_gamma_flux_given, #TeV-1cm-2s-1
																index_gamma_given,
																gamma_ray_ref_energy,
																median_luminosity,
																stdev_sigma_L)


		
		# NEXT COMPUTE NU FLUXES
		index_given=index_given*(-1.0)

		
		nu_fluxes,nu_luminosity = get_flux_distribution(method_used,astropy_coords_in_galactic,
																diffuse_flux_given, #TeV-1cm-2s-1
																index_given,
																nu_ref_energy,
																median_luminosity,
																stdev_sigma_L)

	gamma_fluxes = np.float32(np.log10(np.asarray(gamma_fluxes)))
	gamma_luminosity = np.float32(np.log10(np.asarray(gamma_luminosity)))
	nu_fluxes = np.float32(np.log10(np.asarray(nu_fluxes)))
	nu_luminosity = np.float32(np.log10(np.asarray(nu_luminosity)))
	
	if print_output == True:
		np.savez_compressed(full_path,[array_l,array_b,array_distance,nu_fluxes,nu_luminosity,gamma_fluxes, gamma_luminosity])

	print("Output ARRAY is in form form of: \n \
			0->astropy coordinates,  \n \
			1->log neutrino fluxes,  \n \
			2->log neutrino luminosities,  \n \
			3->log gamma-ray fluxes,  \n \
			4->log gamma-ray luminosities ")
	return [astropy_coords_in_galactic,nu_fluxes,nu_luminosity,gamma_fluxes, gamma_luminosity]

		



### IN DEVELOPMENT
"""
def	Simulate_gamma_ray_fluxes(galcentric_coords_r_phi_z   = None,
							method_used               = "FermiLATpi0",
							diffuse_flux_given          = 2.14e-15, # Tev-1cm-2s-1 Isotropic flux
							print_output                = False,
							full_path                   = "./ default.npy",
							index_given                 = 2.7,
							ref_energy                  = 100.0): #TeV

	index_given=index_given*(-1.0)

	if galcentric_coords_r_phi_z==None:
		print("Error: Give Source Positions in r,z,phi Coordinates for ARRAY FORMAT")
		exit()

	astropy_coords_in_galactic = convert_to_galactic(galcentric_coords_r_phi_z[0],galcentric_coords_r_phi_z[1],galcentric_coords_r_phi_z[2])[0]
	

	simulated_fluxes, sc_luminosity = get_flux_distribution(method_used,astropy_coords_in_galactic,
															diffuse_flux_given, #TeV-1cm-2s-1
															index_given,
															ref_energy)
    

	array_l=np.asarray(astropy_coords_in_galactic.l.deg).astype(np.float16)
	array_b=np.asarray(astropy_coords_in_galactic.b.deg).astype(np.float16)
	array_distance=np.asarray(astropy_coords_in_galactic.distance.kpc).astype(np.float16)



	if print_output == True:
		np.savez_compressed(full_path,[array_l,array_b,array_distance,simulated_fluxes,sc_luminosity])


	return [astropy_coords_in_galactic,simulated_fluxes,sc_luminosity]
"""















