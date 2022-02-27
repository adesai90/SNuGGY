#!/usr/bin/python



#	General	imports
import	os,sys
import	argparse
#	Numpy	/	Scipy  / Matplotlib /healpy
import	numpy	as	np
import	scipy
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
import healpy as hp

font = {'size'   : 20}
matplotlib.rc('font', **font)

#Astropy
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates.representation import CartesianRepresentation

# Definitions from SNuGGY
from sky_distribution import get_model
from sampling	import	InverseCDF
from assign_fluxes import get_flux_distribution

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

	return [transformed_coord,[transformed_coord.l,transformed_coord.b,transformed_coord.distance]]
	

def	simulate_positions(output_file= None,
			distribution_model    ="exponential",
			number_sources_used	  =	1000,
			seed                  =	None,
			plot_dir              = None,
			plot_aitoff_dir_icrs  = None,
			plot_aitoff_dir_gal   = None,
			filename              = None,
			# LIMITS Below, generally set to high vals for inf
			z_max	              = 300.0,  #kpc
			z_min	              = 1e-06, #kpc
			r_max		          =	1000000.0, #kpc
			r_min	              =	0.001,  #kpc
			# r_0 z_0 for distributions made using simple exponential
			z_0		              =	0.6,  #kpc
			r_0	                  =	3.0,   #kpc
			# alpha beta h for distributions made using modified exponential
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

	# CONVERTING 2D PDF TO 1D TO GET SAMPLES

	binning_used_z = np.geomspace(z_min,z_max,num=bins)
	binning_used_r = np.geomspace(r_min,r_max,num=bins)

	vertical_height_z_pdf	=	np.ones(bins)
	for	index_pdf	in	range(len(vertical_height_z_pdf)):
		vertical_height_z_pdf[index_pdf]	=	get_model(distribution_model,distribution_parameters_list,0,binning_used_z[index_pdf])

	distance_pdf	=	np.ones(bins)
	for	index_pdf	in	range(len(vertical_height_z_pdf)):
		distance_pdf[index_pdf]	=	get_model(distribution_model,distribution_parameters_list,binning_used_r[index_pdf],0)


	invCDF_vertical_height_z	=	InverseCDF(binning_used_z,	vertical_height_z_pdf)

	invCDF_distance_r	=	InverseCDF(binning_used_r,	distance_pdf)


	# CREATE R and z BINS TO SAMPLE FROM
	distance_bins	=	np.logspace(-5,	np.log10(r_max),	bins)
	vertical_height_z_bins	=	np.arange(z_min,	z_max,	(z_max-z_min)/float(bins))

	selected_z=[]
	selected_r=[]

	for	index_distribution	in	range(0,number_sources_used):
		rng = np.random.RandomState(seed)	
		random_val_used1	=	rng.uniform(0,	1)
		z_selected	=	invCDF_vertical_height_z(random_val_used1)
		if index_distribution<=number_sources_used/2:
			selected_z.append(float(z_selected))
		else:
			selected_z.append(float(z_selected*(-1)))

		random_val_used2	=	rng.uniform(0,	1)
		r_selected	=	invCDF_distance_r(random_val_used1)
		selected_r.append(float(r_selected))
	
	np.random.shuffle(selected_r)
	np.random.shuffle(selected_z)

	"""	
	# OLD SETUP: CHECK AND DELETE

	vertical_height_z_pdf	=	np.ones(len(vertical_height_z_bins))
	for	index_loop	in	range(len(vertical_height_z_pdf)):
		vertical_height_z_pdf[index_loop]	=	scipy.integrate.quad(lambda	r_integrate:	get_model(distribution_model,distribution_parameters_list,r_integrate,vertical_height_z_bins[index_loop]),r_min,r_max)[0]

	
	invCDF_vertical_height_z	=	InverseCDF(vertical_height_z_bins,	vertical_height_z_pdf)

	selected_z=[]
	selected_r=[]

	for	index_distribution	in	range(0,number_sources_used):	
		random_val_used1	=	rng.uniform(0,	1)
		z_selected	=	invCDF_vertical_height_z(random_val_used1)
		selected_z.append(float(z_selected))

		# NOW GETTING R
		distance_pdf=np.ones(len(distance_bins))
		# ABHISHEK: Can spEed up this by integrating into cdf or other definition
		for	index_loop_per_z	in	range(len(distance_bins)):
			distance_pdf[index_loop_per_z]	=	get_model(distribution_model,distribution_parameters_list,distance_bins[index_loop_per_z],z_selected)

		invCDF_distance	=	InverseCDF(distance_bins,	distance_pdf)
		random_val_used2	=	rng.uniform(0,	1)
		r_selected	=	invCDF_distance(random_val_used2)
		selected_r.append(float(r_selected))

	"""

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
		fig = plt.figure(figsize=(15,15),dpi=200)
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(astropy_coords_in_galactic.transform_to(coord.Galactocentric).x,
					astropy_coords_in_galactic.transform_to(coord.Galactocentric).y,
					astropy_coords_in_galactic.transform_to(coord.Galactocentric).z) # plot the point (2,3,4) on the figure
		ax.set_title("Simulated Sources in Galactocentric Coordinates")
		ax.set_xlabel("\n x (kpc)")
		ax.set_ylabel("\n y (kpc)")
		ax.set_zlabel("\n z (kpc)")
		plt.savefig(plot_dir+"3d_plot_of_simulated_sources.png",bbox_inches="tight")


	if plot_aitoff_dir_gal!= None:
		fig = plt.figure(figsize=(15,9),dpi=250)
		fig.add_subplot(111, projection='aitoff')
		ra = astropy_coords_in_galactic.l.wrap_at(180 * u.deg).radian
		dec = astropy_coords_in_galactic.b.wrap_at(180 * u.deg).radian
		plt.plot(ra,dec,'.', label="Simulated Sources in Galactic Coordinates",alpha=0.5,zorder=0)
		plt.legend(loc="lower right")
		plt.xlabel('l',fontsize=18)
		plt.ylabel('b.',fontsize=18)
		plt.grid(True)
		plt.savefig(plot_aitoff_dir_gal+"galactic_aitoff_of_simulated_sources.png",bbox_inches="tight")


	if plot_aitoff_dir_icrs!= None:
		fig = plt.figure(figsize=(15,9),dpi=250)
		fig.add_subplot(111, projection='aitoff')
		ra2 = astropy_coords_in_galactic.transform_to(coord.ICRS).ra.wrap_at(180 * u.deg).radian
		dec2 = astropy_coords_in_galactic.transform_to(coord.ICRS).dec.radian
		ra2, dec2 = zip(*sorted(zip(ra2,dec2)))
		plt.plot(ra2,dec2,'.', label="Simulated Sources in ICRS Coordinates",alpha=0.5,zorder=0)
		plt.legend(loc="lower right")
		plt.xlabel('R.A.',fontsize=18)
		plt.ylabel('Decl.',fontsize=18)
		plt.grid(True)
		plt.savefig(plot_aitoff_dir_icrs+"icrs_aitoff_of_simulated_sources.png",bbox_inches="tight")


	if filename != None and output_file!=None:
		output_file_full_path = output_file+filename
	elif output_file!=None and filename == None:
		output_file_full_path = output_file+"simulated_%s_source_coordinates.txt"%(number_sources_used)
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

	
	return [array_coords_in_Galactocentric,array_coords_in_galactic,astropy_coords_in_galactic]




def	Get_flux_from_positions(galcentric_coords_r_phi_z   = None,
							method_used               = "StandardCandle",
							 plot_healpy_template_dir    = None, # Given only with Fermi-LAT_pi0 template
							diffuse_flux_given          = 1.81e-14, # Tev-1cm-2s-1 Isotropic flux
							print_output                = False,
							full_path                   = "./ default.npy",
							index_given                 = 2.0,
							ref_energy                  = 100): #TeV

	index_given=index_given*(-1.0)

	if galcentric_coords_r_phi_z==None:
		print("Error: Give Source Positions in r,z,phi Coordinates for ARRAY FORMAT")
		exit()

	astropy_coords_in_galactic = convert_to_galactic(galcentric_coords_r_phi_z[0],galcentric_coords_r_phi_z[1],galcentric_coords_r_phi_z[2])[0]
	

	simulated_fluxes, sc_luminosity = get_flux_distribution(method_used,astropy_coords_in_galactic,
															diffuse_flux_given, #TeV-1cm-2s-1
															index_given,
															ref_energy)
    

	#
	# PLOT FERMIPI0 TEMPLATE USING HEALPY IF DIR IS SPECIFIED
	#

	if method_used=="Fermi-LAT_pi0" and plot_healpy_template_dir != None:
		fig = plt.figure(figsize=(15,9),dpi=250)
		hp.mollview(template,
					coord=["C", "G"],
					norm="hist", 
					unit="$\pi^0$-decay",
					title="Fermi LAT $\pi^0$-decay template with histogram equalized color mapping",
					hold=True)
		plt.savefig(plot_healpy_template_dir+"fermi_pi0_decay_template_in_galactic_coords.png",bbox_inches="tight")



	if print_output == True:
		np.savez_compressed(full_path,[astropy_coords_in_galactic,simulated_fluxes,sc_luminosity])


	return [astropy_coords_in_galactic,simulated_fluxes,sc_luminosity]
















