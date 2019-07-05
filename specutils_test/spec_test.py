from astropy import units as u
import numpy as np
from scipy.signal import medfilt
from specutils import Spectrum1D, SpectralRegion
from specutils.fitting import find_lines_derivative
from specutils.fitting import find_lines_threshold 
from specutils.manipulation import noise_region_uncertainty 
from astropy.modeling import models
from astropy.modeling import fitting
from specutils.fitting import fit_generic_continuum
import pylab as plt
from specutils.fitting import estimate_line_parameters
from specutils.fitting import fit_lines



#importing the flux and wavelength information
flux = np.load('spectra.npz')['flux']
wvln = np.load('spectra.npz')['wave']


#spectrum = Spectrum1D(flux=flux*u.erg/u.s/u.cm/u.cm/u.angstrom, spectral_axis=wvln*u.angstrom)

#Finding where the noisy part of the spectrum is below our sensitivity limit
index = np.where(wvln < 3850)[0][-1] 

#Reducing the data to where this index lies
spec = flux[index:]
wavln = wvln[index:]

spectrum = Spectrum1D(spectral_axis = wavln * u.angstrom,
                     flux = spec * u.erg/u.s/u.cm/u.cm/u.angstrom)

#In order to have the best chance for finding the emission lnes we need to get rid 
#of the continuum shape and for this we need the fitting models we imported above to 
#fit the shape

############
#Example of fitting a continuum with the data that I was able to get.
############

#This fits it with a weird model but we can change it to linear as thats what 
#the program tells me to do
continuum_fit = fit_generic_continuum(spectrum, model=models.Linear1D())

#the above return a function that i can pass in x values to fit the continuum
#ideally this would be the whole spectrum wavelength range

y=continuum_fit(spectrum.spectral_axis)

centered_spec = Spectrum1D(spectral_axis=spectrum.spectral_axis,
                           flux= spectrum.flux - y)

#getting the emission and absorption lines using threshold
#NOTE: We need to have noise in order for me to use this and can be done using
#      noise_region_uncertainty
lines = find_lines_threshold(spectrum, noise_factor=3)

#Another way that we can find emission and absorption lines is by using line derivatives
lines_2 = find_line_derivatives('Centered Spectrum1D Object', flux_threshold=.75)

#to get the emission lines Qtable we run
emission = lines[lines['line_type']=='emission']

#to get where they are locted we use
emission_wvln = emission['line_center']

#we can also get the index where they occur in if we want to look at the non
#Spectrum1D object
emisison_index = emission['line_center_index']


#Once we have the emission lines all taken care of we can then use this information
#to do some fitting and get an estimate of the line flux and start performing some 
#science


#to fit a gaussian using specutils we need to have a region over which to fit
#for this we can cut a region around the emission center and fit a gaussian to that
window = 20*u.angstrom

#here we cut a slice of the spectum to look
sub_region = SpectralRegion('emission center' - window, 'emission center' + window)
sub_spectrum = extract_region('centered flux', sub_region)

#then we can estimate the mean, amplitude and stddev for the gaussian using 
#estimate_line_parameter

param = estimate_line_parameters(sub_spectrum, models.Gaussian1D())

#now with these parameters in place we fit the data of the sub spectrum
gauss_fit = fit_lines(sub_spectrum, 
                      models.Gaussian1D(amplitude=param.amplitude,
                                        stddev=param.stddev
                                        mean=param.mean))
