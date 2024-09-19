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
	"""
    Writes input string to a text file.

    Parameters:
        msg (str): The input string to be written to the log file.
        logfile (str): The path to the text file.

	"""
	if logfile!="":
		log = open(logfile, "a")
		log.write(msg+"\n")
		log.close
	return

def convert_to_galactic(r_conv,z_conv,theta_conv):
	"""
    Converts coordinates from cylindrical representation to galactic coordinates.

    Parameters:
        r_conv (float): Radial distance.
        z_conv (float): Vertical distance.
        theta_conv (float): Azimuth angle.

    Returns:
        list: A list containing [galactic_astropy_coordinates,[l,b,d]] ; where [l,b,d] is an array

    Notes:
        - Requires the `astropy.coordinates` package.
        - The input angles should be in radians and distances in kpc

    """
	c = coord.CylindricalRepresentation(rho=r_conv * u.kpc,phi=theta_conv * u.radian,z=z_conv * u.kpc)
	c2= c.represent_as(CartesianRepresentation)

	c3=coord.Galactocentric(x=c2.x,y=c2.y,z=c2.z)
	del c,c2

	transformed_coord = c3.transform_to(coord.Galactic())

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
	
	"""
    Simulates sources in galactic coordinates.

    Parameters:
        output_file (str, optional): The path where the output file will be saved. If not provided, the output will not be saved.
        distribution_model (str, optional): The distribution model to use for generating positions. Default is "exponential".
        number_sources_used (int, optional): The number of sources to simulate. Default is 1000.
        seed (int, optional): The seed for the random number generator. If not provided, a random seed will be used.
        plot_dir (str, optional): The directory to save the 3D plot of galactic coordinates. If not given; plot will not be saved.
        plot_aitoff_dir_icrs (str, optional): The directory to save the Aitoff projection plot in ICRS coordinates. If not given; plot will not be saved.
        plot_aitoff_dir_gal (str, optional): The directory to save the Aitoff projection plot in galactic coordinates. If not given; plot will not be saved.
        filename (str, optional): The name of the output file. If output_file is given but filename is not provided, a default name will be used.
        make_pdf_plot_location (str, optional): The location to save the PDF plots. Default is None.
        z_max (float, optional): The maximum vertical distance (z) in kpc. Default is 300.0.
        z_min (float, optional): The minimum vertical distance (z) in kpc. Default is 1e-06.
        r_max (float, optional): The maximum radial distance (r) in kpc. Default is 1000000.0.
        r_min (float, optional): The minimum radial distance (r) in kpc. Default is 0.001.
        z_0 (float, optional): The vertical scale parameter for the exponential distribution. Default is 0.6.
        r_0 (float, optional): The radial scale parameter for the exponential distribution. Default is 3.0.
        alpha (float, optional): The alpha parameter for the modified exponential distribution. Default is 2.
        beta (float, optional): The beta parameter for the modified exponential distribution. Default is 3.53.
        h (float, optional): The h parameter for the modified exponential distribution. Default is 0.181.
        bins (int, optional): The number of bins used for sampling. Default is 1000.

    Returns:
        list: A list containing [array of coordinates in Galactocentric, array of coordinates in galactic coordinates, astropy galactic coordinates].

    Notes:
        - Requires the `astropy.coordinates` package.
        - The output is not saved if both `output_file` and `filename` are not provided.
        - The PDF plots are not created if `make_pdf_plot_location` is not provided.

    """

	rng = np.random.RandomState(seed)

	distribution_parameters_list = [r_0,z_0,alpha,beta,h,r_min,r_max,z_min,z_max]

	if make_pdf_plot_location!=None:
		get_model(distribution_model,distribution_parameters_list,0,1,make_pdf_plot_location)
		make_pdf_plot_location=None
	
	if make_pdf_plot_location!=None:
		make_pdf_plot_location=None
		print("Warning: Manually foced make_pdf_plot_location=None")
	
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
		print("Saving output to file: ",output_file)
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
							nu_method_used              = "StandardCandle", #Neutrino
							gamma_ray_method_used       = "StandardCandle", #Gamma_ray
							diffuse_nu_flux_given       = 1e-15, # Tev-1cm-2s-1 Neutrino Isotropic flux
							diffuse_gamma_flux_given    = 1e-15, # Tev-1cm-2s-1 Gamma_ray Isotropic flux
							print_output                = False,
							full_path                   = None, #"./ default.npy",
							index_nu_given              = 2.7, #Neutrino
							index_gamma_given           = 2.7, #Gamma_ray
							nu_ref_energy               = 100.0, #NEUTRINO reference energy
							gamma_ray_ref_energy        = 50.0, #Gamma reference energy
							simulate_gamma_ray_frm_nu	= False,
							simulate_nu_frm_gamma       = False,
							pp_or_pgamma				= "pp",
							mean_luminosity_nu        = None,
							mean_luminosity_gamma     = None,
							stdev_sigma_L_nu            = 1.,
							stdev_sigma_L_gamma         = 1.,
							gamma_energy_range_low		= 1e0,
							gamma_energy_range_high		= 5e1,
							nu_energy_range_low			= 1e1,
							nu_energy_range_high		= 1e4
							): 

	"""
    Computes fluxes of neutrinos and gamma rays based on source positions.

    Parameters:
        galcentric_coords_r_phi_z (tuple, required): Source positions in galactocentric coordinates as an array [r, z , phi]. Code will not run if not given.
        nu_method_used (str, optional): The method used for computing neutrino flux. Default is "StandardCandle".
        gamma_ray_method_used (str, optional): The method used for computing gamma ray flux. Default is "StandardCandle".
        diffuse_nu_flux_given (float, optional): The diffuse neutrino flux in TeV^-1 cm^-2 s^-1. Default is 1e-15.
        diffuse_gamma_flux_given (float, optional): The diffuse gamma ray flux in TeV^-1 cm^-2 s^-1. Default is 1e-15.
        print_output (bool, optional): Whether to print the output or not. Default is False.
        full_path (str, optional): The full path where the output array will be saved as an npz file. Default is None.
        index_nu_given (float, optional): The index of the power-law spectrum for neutrinos. Default is 2.7.
        index_gamma_given (float, optional): The index of the power-law spectrum for gamma rays. Default is 2.7.
        nu_ref_energy (float, optional): The reference energy for neutrinos in TeV. Default is 100.0.
        gamma_ray_ref_energy (float, optional): The reference energy for gamma rays in TeV. Default is 50.0.
        simulate_gamma_ray_frm_nu (bool, optional): Whether to simulate gamma ray flux from neutrino flux. Default is False.
        simulate_nu_frm_gamma (bool, optional): Whether to simulate neutrino flux from gamma ray flux. Default is False.
        pp_or_pgamma (str, optional): The type of interaction used for computing fluxes. Default is "pp".
        mean_luminosity_nu (float, optional): The mean luminosity for neutrinos. Default is None.
        mean_luminosity_gamma (float, optional): The mean luminosity for gamma rays. Default is None.
        stdev_sigma_L_nu (float, optional): The standard deviation of the luminosity distribution for neutrinos. Default is 1.0.
        stdev_sigma_L_gamma (float, optional): The standard deviation of the luminosity distribution for gamma rays. Default is 1.0.
        gamma_energy_range_low (float, optional): The lower energy range for gamma rays in TeV. Default is 1e0.
        gamma_energy_range_high (float, optional): The higher energy range for gamma rays in TeV. Default is 5e1.
        nu_energy_range_low (float, optional): The lower energy range for neutrinos in TeV. Default is 1e1.
        nu_energy_range_high (float, optional): The higher energy range for neutrinos in TeV. Default is 1e4.

    Returns:
        list: The output array is in the form:
            - 0 -> Astropy coordinates
            - 1 -> Logarithm of neutrino fluxes
            - 2 -> Logarithm of neutrino luminosities
            - 3 -> Logarithm of gamma ray fluxes
            - 4 -> Logarithm of gamma ray luminosities

    Raises:
        ValueError: If both `simulate_gamma_ray_frm_nu` and `simulate_nu_frm_gamma` are set to True.

    Notes:
        - This function computes neutrino flux and/or gamma ray flux based on source positions.
        - If neutrino and gamma ray fluxes are required to be simulated seperately, set `simulate_gamma_ray_frm_nu` and `simulate_nu_frm_gamma` to False
        - If `print_output` is True, the output array will be saved as an npz file at `full_path`.

    
    """

	index_nu_given=index_nu_given*(-1.0)
	index_gamma_given=index_gamma_given*(-1.0)
	
	if simulate_gamma_ray_frm_nu==True and simulate_nu_frm_gamma==True:
		print("Both  simulate_gamma_ray_frm_nu and simulate_nu_frm_gamma cannot be TRUE!!!") 
		exit()
	elif simulate_gamma_ray_frm_nu==True:
		gamma_ray_ref_energy = nu_ref_energy/2.
		print("Reference energy:\n Neutrino:%s TeV \n Gamma-rays:%s TeV"%(nu_ref_energy,gamma_ray_ref_energy))

		print("This method computes neutrino flux and then converts it to gamma ray fluxes based on neutrino gamma ray relation.")



		if galcentric_coords_r_phi_z==None:
			print("Error: Give Source Positions in r,z,phi Coordinates for ARRAY FORMAT as [r,z,phi]")
			exit()

		astropy_coords_in_galactic = convert_to_galactic(galcentric_coords_r_phi_z[0],galcentric_coords_r_phi_z[1],galcentric_coords_r_phi_z[2])[0]
	
    

		array_l=np.asarray(astropy_coords_in_galactic.l.deg).astype(np.float16)
		array_b=np.asarray(astropy_coords_in_galactic.b.deg).astype(np.float16)
		array_distance=np.asarray(astropy_coords_in_galactic.distance.kpc).astype(np.float16)


		nu_fluxes,nu_luminosity = get_flux_distribution(nu_method_used  ,astropy_coords_in_galactic,
																diffuse_nu_flux_given, #TeV-1cm-2s-1
																index_nu_given,
																nu_ref_energy,
																mean_luminosity_nu,
																stdev_sigma_L_nu,
																nu_energy_range_low,
																nu_energy_range_high)
		

		
		# NEXT PORTION GIVES GAMMA RAY FLUXES WITH LUMINOSITY PER SOURCE AT Ref energy = ref_energy/2
		gamma_fluxes, gamma_luminosity = get_gamma_from_nu(astropy_coords_in_galactic,
											nu_fluxes,
											nu_ref_energy,
											index_nu_given,
											pp_or_pgamma)
				
	elif simulate_nu_frm_gamma==True:
		nu_ref_energy = gamma_ray_ref_energy*2.
		print("Reference energy:\n Neutrino:%s TeV \n Gamma-rays:%s TeV"%(nu_ref_energy,gamma_ray_ref_energy))

		print("This method computes Gamma ray flux and then converts it to neutrino fluxes based on neutrino gamma ray relation.")

	
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
																mean_luminosity_gamma,
																stdev_sigma_L_gamma,
																gamma_energy_range_low,
																gamma_energy_range_high)


		
		# NEXT PORTION GIVES GAMMA RAY FLUXES WITH LUMINOSITY PER SOURCE AT Ref energy = ref_energy/2
		nu_fluxes,nu_luminosity = get_nu_from_gamma(astropy_coords_in_galactic,
											gamma_fluxes,
											gamma_ray_ref_energy,
											index_nu_given,
											pp_or_pgamma)

	else:
		print("This method computes Gamma ray flux and neutrino fluxes seperately.")
		print("Reference energy:\n Neutrino:%s TeV \n Gamma-rays:%s TeV"%(nu_ref_energy,gamma_ray_ref_energy))

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
																mean_luminosity_gamma,
																stdev_sigma_L_gamma,
																gamma_energy_range_low,
																gamma_energy_range_high)


		
		# NEXT COMPUTE NU FLUXES
		
		nu_fluxes,nu_luminosity = get_flux_distribution(nu_method_used  ,astropy_coords_in_galactic,
																diffuse_nu_flux_given, #TeV-1cm-2s-1
																index_nu_given,
																nu_ref_energy,
																mean_luminosity_nu,
																stdev_sigma_L_nu,
																nu_energy_range_low,
																nu_energy_range_high)

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

		



