from astropy import units as u
import numpy as np
from scipy.signal import medfilt
from specutils import Spectrum1D, SpectralRegion
from specutils.fitting import find_lines_derivative
from specutils.fitting import find_lines_threshold 
from specutils.manipulation import noise_region_uncertainty 

#importing the flux and wavelength information
flux = np.load('spectra.npz')['flux']
wvln = np.load('spectra.npz')['wave']

#making the spectrum with appropriate units
spectrum = Spectrum1D(flux=flux*u.erg/u.s/u.cm/u.cm/u.angstrom, spectral_axis=wvln*u.angstrom)

#getting the emission and absorption lines
lines = find_lines_threshold(spectrum, noise_factor=3)

