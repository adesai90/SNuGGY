#!/usr/bin/python



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
from assign_fluxes import get_flux_distribution
from plotter import *

#################################                                                                                                                                                                                             
def Write2File(msg,logfile):

    if logfile!="":
        log = open(logfile, "a")
        log.write(msg+"\n")
    log.close

    return
#################################


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
	

	
	#bins=1000

	rng = np.random.RandomState(seed)

	distribution_parameters_list = [r_0,z_0,alpha,beta,h]
	"""
	Change	this	based	on	firesong	later

	SAMPLING	FROM	bivariate	distribution	with	joint	pdf	𝑝𝑋,𝑌(𝑥,𝑦)
	https://stats.stackexchange.com/questions/471781/sampling-from-a-continuous-2-dimensional-probability-distribution-function-for-i
	Px(x)	=	integrateP(x,y)dy

	"""
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
		"""
		for index_coords in range(number_sources_used):
			#array_coords_in_Galactocentric = [selected_z,selected_r,selected_angles]
			
			line="%s %s %s %s %s %s"%(selected_r[index_coords],
									  selected_angles[index_coords],
									  selected_z[index_coords],
									  astropy_coords_in_galactic.l.deg[index_coords],
									  astropy_coords_in_galactic.b.deg[index_coords],
									  astropy_coords_in_galactic.distance.kpc[index_coords],
											)
			
			Write2File(line,output_file_full_path)
		"""

	
	return [array_coords_in_Galactocentric,array_coords_in_galactic,astropy_coords_in_galactic]




def	Get_flux_from_positions(galcentric_coords_r_phi_z   = None,
							method_used               = "StandardCandle",
							 plot_healpy_template_dir    = None, # Given only with Fermi-LAT_pi0 template
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




### IN DEVELOPMENT
"""
def	Simulate_gamma_ray_fluxes(galcentric_coords_r_phi_z   = None,
							method_used               = "StandardCandle",
							 plot_healpy_template_dir    = None, # Given only with Fermi-LAT_pi0 template
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
















