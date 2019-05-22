import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
import astropy.coordinates as coord
import astropy.units as u

class LocalSheet(coord.BaseCoordinateFrame):
	"""
	Local Sheet Coordinates
	(see McCall 2014, <http://adsabs.harvard.edu/abs/2014MNRAS.440..405M>,
	and references therein).
	"""

	frame_specific_representation_info = {
		coord.SphericalRepresentation: [
			coord.RepresentationMapping('lon', 'lsl'),
			coord.RepresentationMapping('lat', 'lsb')
		],
		coord.CartesianRepresentation: [
			coord.RepresentationMapping('x', 'lsx'),
			coord.RepresentationMapping('y', 'lsy'),
			coord.RepresentationMapping('z', 'lsz')
		],
		coord.CartesianDifferential: [
			coord.RepresentationMapping('d_x', 'v_x', u.km/u.s),
			coord.RepresentationMapping('d_y', 'v_y', u.km/u.s),
			coord.RepresentationMapping('d_z', 'v_z', u.km/u.s)
		],
	}

	default_representation = coord.SphericalRepresentation
	default_differential = coord.SphericalCosLatDifferential

	# North supergalactic pole in Galactic coordinates.
	# Needed for transformations to/from Galactic coordinates.
	n_sgal = coord.SkyCoord(sgl=241.74*u.degree, sgb=82.05*u.degree, frame="supergalactic")
	_nlsp_gal = n_sgal.galactic


@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, LocalSheet)
def gal_to_ls():
	mat1 = rotation_matrix(90, 'z')
	mat2 = rotation_matrix(90 - LocalSheet._nlsp_gal.b.degree, 'y')
	mat3 = rotation_matrix(LocalSheet._nlsp_gal.l.degree, 'z')
	return matrix_product(mat1, mat2, mat3)

@frame_transform_graph.transform(coord.StaticMatrixTransform, LocalSheet, coord.Galactic)
def ls_to_gal():
	return matrix_transpose(gal_to_ls())

### Main ##########################################################
if __name__ == "__main__":
	#Example shows the Local Sheet plane in Supergalactic coordinates
	ls = LocalSheet(lsl=np.linspace(0, 2*np.pi, 128)*u.radian, lsb=np.zeros(128)*u.radian)
	sgal = ls.transform_to(coord.Supergalactic)

	fig, axes = plt.subplots(2, 1, figsize=(8, 10), subplot_kw={'projection': 'aitoff'})
	axes[0].set_title("LocalSheet")
	axes[0].plot(ls.lsl.wrap_at(180*u.deg).radian, ls.lsb.radian, linestyle='none', marker='.')
	axes[1].set_title("Supergalactic")
	axes[1].plot(sgal.sgl.wrap_at(180*u.deg).radian, sgal.sgb.radian, linestyle='none', marker='.')
	plt.show()
