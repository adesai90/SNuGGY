import matplotlib.pyplot as plt
import matplotlib

#Astropy
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates.representation import CartesianRepresentation


font = {'size'   : 25}
matplotlib.rc('font', **font)
 

def make_3d_plot(astropy_coords_in_galactic,plot_dir):
	fig = plt.figure(figsize=(15,15),dpi=200)
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(astropy_coords_in_galactic.transform_to(coord.Galactocentric()).x,
				astropy_coords_in_galactic.transform_to(coord.Galactocentric()).y,
				astropy_coords_in_galactic.transform_to(coord.Galactocentric()).z) 
	ax.set_title("Simulated Sources in Galactocentric Coordinates")
	ax.set_xlabel("\n x (kpc)")
	ax.set_ylabel("\n y (kpc)")
	ax.set_zlabel("\n z (kpc)")
	plt.savefig(plot_dir+"3d_plot_of_simulated_sources.png",bbox_inches="tight")
	return

def make_gal_aitoff_plot(astropy_coords_in_galactic,plot_aitoff_dir_gal):
	fig = plt.figure(figsize=(15,9),dpi=250)
	fig.add_subplot(111, projection='aitoff')
	ra = astropy_coords_in_galactic.l.wrap_at(180 * u.deg).radian
	dec = astropy_coords_in_galactic.b.wrap_at(180 * u.deg).radian
	plt.plot(ra,dec,'.', label="Simulated Sources in Galactic Coordinates",alpha=0.5,zorder=0)
	plt.legend(loc="lower right")
	plt.xlabel('l')
	plt.ylabel('b.')
	plt.grid(True)
	plt.savefig(plot_aitoff_dir_gal+"galactic_aitoff_of_simulated_sources.png",bbox_inches="tight")
	return

def make_icrs_aitoff_plot(astropy_coords_in_galactic,plot_aitoff_dir_icrs):
	fig = plt.figure(figsize=(15,9),dpi=250)
	fig.add_subplot(111, projection='aitoff')
	ra2 = astropy_coords_in_galactic.transform_to(coord.ICRS()).ra.wrap_at(180 * u.deg).radian
	dec2 = astropy_coords_in_galactic.transform_to(coord.ICRS()).dec.radian
	ra2, dec2 = zip(*sorted(zip(ra2,dec2)))
	plt.plot(ra2,dec2,'.', label="Simulated Sources in ICRS Coordinates",alpha=0.5,zorder=0)
	plt.legend(loc="lower right")
	plt.xlabel('R.A.')
	plt.ylabel('Decl.')
	plt.grid(True)
	plt.savefig(plot_aitoff_dir_icrs+"icrs_aitoff_of_simulated_sources.png",bbox_inches="tight")
	return




