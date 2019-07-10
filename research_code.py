from astropy.io import fits
import numpy as np
import pylab as plt
import glob as glob
from scipy.stats import mode
from matplotlib.colors import LogNorm
from transform import get_wcs_solution
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from specutils.fitting import find_lines_threshold
from specutils.fitting import find_lines_derivative
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from scipy.stats import norm
from specutils import Spectrum1D, SpectralRegion
from specutils.analysis import line_flux
from astropy import units as u
from specutils.manipulation import noise_region_uncertainty 
from astropy.modeling import models
from astropy.modeling import fitting
from specutils.fitting import fit_generic_continuum
from specutils.fitting import estimate_line_parameters
from specutils.fitting import fit_lines
from specutils.analysis import equivalent_width
from specutils.manipulation import extract_region  

#grabbing all of micaela's fits files from the current directory
files = [x for x in glob.glob('*.fits') if 'SDSS' not in x]

file_num = np.unique([x.split('_')[1] for x in files])

#print(files)
#getting all the sdss fits files so we can do flux calibration
#sdss_files = [x for x in glob.glob('*.fits') if 'SDSS' in x]



#reading in the dat file
ID, redshift = np.genfromtxt('targets.dat', usecols=(0,1), unpack = True, skip_header=2, dtype = 'str')

ID1 = [x.split('.')[0] for x in ID]

filt_z = np.zeros(len(ID1), dtype = bool)

for j, val in enumerate(ID1):
    for i in file_num:
        if val in i:
            filt_z[j] = True 

z = redshift.astype(float)

ID_1 = np.array(ID1)[filt_z]
z_redshift = np.array(z)[filt_z]



line_info = np.genfromtxt('linelist.dat', unpack =True, dtype='str', usecols=(0,1), skip_header = 1)

line_wavelength = line_info[0].astype(float)
line_name = line_info[1]


def fitting_gaussian(data, wavelength):
    
    '''
    This function will fit a gaussian to median combined coulmn data corresponding to a window around the trace. 
    We do this so that the maximum intenisty gets the most weight and points outside the center get less weight
        
    Parameter
    -------------
    data: this is the data that we would like to fit with a gaussian model. 
    
        

    Output
    -------------
    a     
    
    '''
    
    #making a pixel array here for the rows
    x = np.arange(len(data[:,0]))
    
    #definition of the gaussian function
    def f (x, A, mu, sigma):
        
        '''
        Making a Gaussian Function to fit the columns using curve_fit
        '''
        
        prefactor = A
        factor = np.exp(-(x-mu)**2/(2*sigma**2))
        return prefactor * factor
    
    sig = []
    wvln = []
        
    N = 21
    skip = 50
    
    start = 0
    end = N
    
    i = 0
    #plt.figure(figsize = (10,10))
    #plt.title('Sigma vs wvln')
    
    while True:
        
        #print(start)
        #print(end)
        #print()
        
        median_data = np.median(data[:, start:end], axis = 1)
        med_wvln = np.median(wavelength[start:end])
        
        if np.amax(median_data) == 0:
            start = start + N + skip
            end = start + N
            continue
        
        popt, covar = curve_fit(f, x, median_data, 
                                p0 = [np.amax(median_data), len(median_data)//2, 40], 
                                bounds=[(0, 0, 1), (np.amax(median_data), len(data[:, 0]), 100)])
        std_dev = popt[-1]
        
        sig.append(std_dev)
        wvln.append(med_wvln)
        
        start = start + N + skip
        end = start + N
        
        #print(start)
        #print(end)
        #print()
        #plt.plot(wvln, medfilt(sig, kernel_size = 7))
        #plt.show()
        
        if start > len(data[0,:]) or end > len(data[0,:]):
            #print(i)
            #i+=1
            break
        
        #p0 = [np.amax(data[:,i]), len(data[:,0])//2, 25],
        #popt, covar = curve_fit(f, x, data[:, i], bounds=[(0, 0, 1), (np.amax(data), len(data[:, 0]), 50)])
        
        #this part makes the gaussian function with the parameters fit from above
        #y = f(x, *popt)
        
        #y = norm.pdf(x, popt[-2], std_dev)
        
        #test = np.random.normal(popt[-2], std_dev, 1000)
        
    #plt.plot(wvln, medfilt(sig, kernel_size = 7))
    #plt.show()
    #line = np.polyfit(wvln[4:], sig[4:], deg=1)
    
    #x = np.linspace(wvln[0], wvln[-1], 1000)
    #y = line[0]*x + line[-1]
    
    y = norm.pdf(x, len(median_data)//2, 4)
    renorm_y = y/np.amax(y)
    
    #plt.title('Gaussian Fit to the Spectrum Data')
    #plt.plot(x, renorm_y)
    #plt.show()
    #print(len(y))
    
    return renorm_y

def finding_row_center(data):
    
    '''
    This function will try to find the approximate row index where the spectrum resides.

    Parameters
    --------------
    data: this is the what I call the cut data so not the entire fits.getdata. Instead we remove some of the 
          boundary of the data on each side and have a smaller subset.

    Output:
    row_ind: this will be the row index from the cut array where the center of the spectrum resides.

    Note to get the exact value of the array you have to take the value returned by this function and add it to 
    your row_min variable used to cut the original 2D array
    '''
    
    #making a row index array so that I can find the index of greatest number for a given column. Then I will pass this into mode
    #to see which row occurs the most. This shoul dbe the center of the spectrum or very close to it.
    row_index = []
    
    #this code below checks each column and finds the maximum value for the respective column. 
    for i in range(len(data[0, :])):
        
        #this index variable holds the index where the max value occurs in the column which translates to the row
        index = np.argmax(data[:, i])
        row_index.append(index)
    
    #Getting the row index using mode
    row_ind = mode(row_index)[0][0]

    return row_ind



def spectrum(file):
    '''
    This function will try to convert any 2D spectral array and convert that into a 1D array.

    Parameters
    --------------
    file: this is the filename of the file we want to extract a 1D spectrum from.

    Output
    --------------
    spectrum: this is the 1D reduced spectrum gathered from the 2D array we passed in.
    '''

    #gathering the data from the fits file we pass in
    data = fits.getdata(file)
    
    #getting the header information here as this will be used below in the code to calculate the wavelength array
    hdr = fits.getheader(file)

    '''
    #this is the cutting of the array were we get rid of the edges on all sides
    row_min = len(data[:,0])//3
    row_max = len(data[:,0]) - row_min
    column_min = len(data[0,:])//3
    column_max = len(data[0,:])-column_min
    
    '''
    
    ###############################
    #code below will try to determine what would be the appropriate boxing for the data
    ###############################
    
    #first we split up the filename of the file as each file will have different row_min and row_max
    x = file.split('_')
    
    #making the column_mins and max values as this will be true for all the files
    #column_min = 500
    #column_max = 3720
    
    #This code here checks the filename and if the file has the following things it assigns to them the appropriate row_min and row_max
    #These were found by using ds9 and were picked so as to not include the slits
    
    #filt is supposed to filter out the data so that my finding row spectrum function works
    #this should keep only a window where the spectrum lies
    filt = np.ones(data.shape)
    
    #making a window variable sp that i can change it accordingly to what I want to look at
    window = 0
    
    
    if '_cem.fits' in file:
            
        row_min = 220
        row_max = 500
        
        filt[:row_min, :] = np.zeros(len(data[0,:]))
        filt[row_max:, :] = np.zeros(len(data[0,:]))
        
        window = 50
    
    if 'b_ce.fits' in file:

        row_min = 1410
        row_max = 1890

        filt[:row_min, :] = np.zeros(len(data[0,:]))
        filt[row_max:, :] = np.zeros(len(data[0,:]))
        window = 50
    
    if 'r_ce.fits' in file:

        row_min = 1350
        row_max = 1800

        filt[:row_min, :] = np.zeros(len(data[0,:]))
        filt[row_max:, :] = np.zeros(len(data[0,:]))
        window = 50
    
    
    #plt.figure(figsize = (10,10))
    #plt.imshow(filt*data, cmap = 'gray', origin = 'lower', norm = LogNorm())
    #plt.show()
    

    #declaring the cut_data array below, which is a simplified version of the data gathered above. It should only
    #contain the spectrum with a box excluding any slit effects
    #cut_data = data[row_min:row_max, column_min:column_max] 
    
    #plt.figure(figsize = (14, 8))
    #plt.imshow(cut_data, origin = 'lower', cmap = 'gray',norm = LogNorm())
    #plt.show()
    
    
    #This one calculates where in the original data array the correct row_index corresponding to the center of the spectrum lies
    #We used a function called finding_row_center to find the row index of the masked data
    row_spectrum = finding_row_center(data*filt)
    
    #given the row where spectrum is at we are looking at that row +/- windows
    boxed_below = row_spectrum - window
    boxed_above = row_spectrum + window
    
    #getting the boxed data
    boxed_data = data[boxed_below : boxed_above ,:]
    
    #getting polynomial that transform from pixels to wavelength
    p = get_wcs_solution(hdr)
    
    #getting x and y values for the pixels
    X, Y = np.meshgrid(range(len(data[0,:])), range(len(data[:,0])))
    
    #wavelength array from the X and Y we put in
    wvln_arr = p(X, Y)
    
    #getting the spectrum wavelength
    wvln_spec = wvln_arr[row_spectrum,:]
    
    #just adding the boxed_data
    adding_target = np.sum(boxed_data, axis = 0)
    
    #this returns a gaussian fit to the spectrum so that more 
    gauss_mult = fitting_gaussian(boxed_data, wvln_spec)
     
    #plt.figure(figsize = (10,10))
    #plt.imshow(gauss_mult*boxed_data, origin='lower', cmap='gray', norm = LogNorm())
    #plt.colorbar()
    #plt.show()
    
    gauss_filtered = (boxed_data.T * gauss_mult/np.amax(gauss_mult)).T
    
    gauss_added = np.sum(gauss_filtered, axis = 0)
    
    #plt.figure(figsize = (16, 6))
    #plt.plot(wvln_spec, gauss_added)
    #plt.show()
    
    #np.savez('spectra.npz', flux=gauss_added, wave = wvln_spec)
    
    return adding_target, wvln_spec  

def fitting_continuum(wavelength_spec, spectra, z, line_lam, file):
    
    '''
    This function will make a boolean filter to mask out the emission lines and any zeros we have in the spectra as well 
    as very noisy parts in the spectra
    
    
    Parameter
    ------------
    wavelength_spec: this is the wavelength of the extracted spectrum we got from the spectrum function
    spectra: the values of the extracted spectra, need this to mask out all the zero value
    
    NOTE: wavelength_spec and spectra need to be the same length
    
    z: redshift
    line_lam: this is a list of all the lines we are interested and their rest frame wavelengths
    
    Output
    ------------
    filt: boolean filter masking out emission lines and places where zeros occur.
    
    '''
    
    #making a boolean filter the length of wavelength and spectra 
    filt = np.ones(len(wavelength_spec), dtype = bool)
    filt_noise = np.ones(len(wavelength_spec), dtype = bool)
    
    #checking spectra values to see if we get zero if we do then we change those index values to False
    for i, val in enumerate(spectra):
        
        #checking if the spectra has zeros
        if val == 0:
            filt[i] = False
    
    #this will mask out the emission lines so that we can fit the continuum 
    for j in line_lam:
        
        if j * (1+z) < wavelength_spec[0] or j * (1+z) > wavelength_spec[-1]:
            continue
        else:
            #print('Before Subtraction')
            #wavelength_spec - j*(1+z)
            #print('After Subtraction')
            
            index = np.abs(wavelength_spec - j*(1+z)).argmin()
            window = 20
            filt[index-window:index+window] = False    
        
    if 'cem.fits' in file:

        ind = np.where(wavelength_spec < 4000)
        filt[ind] = False
        filt_noise[ind] = False
        
    elif 'mods1b' in file:

        ind = np.where(wavelength_spec < 3700)
        ind2 = np.where(wavelength_spec > 5500)
        ind_tot = np.concatenate((ind, ind2), axis=None)
        filt[ind_tot] = False
        filt_noise[ind_tot] = False

    elif 'mods1r' in file:

        ind = np.where(wavelength_spec < 5500)
        ind2 = np.where(wavelength_spec > 8700)
        ind_tot = np.concatenate((ind, ind2), axis=None)
        filt[ind_tot] = False
        filt_noise[ind_tot] = False

    ##############
    #testing the masking filter and see how well it fits the continuum
    ##############
    
    #making a specutils spectrum object from the filtered data
    spectrum = Spectrum1D(spectral_axis=wavelength_spec[filt]*u.angstrom, flux = spectra[filt]* u.erg/u.s/u.cm/u.cm/u.angstrom)

    continuum_fit = fit_generic_continuum(spectrum, model=models.Linear1D())
    y_continuum = continuum_fit(wavelength_spec*u.angstrom)
    
    #print(len(spectra))        
    #print(len(spectra[filt]))
    
    #peaks, prop = find_peaks(spectra, height=.2, distance=40, width = 3)
    
    #testing code by plotting
    '''
    fig = plt.figure(figsize=(14, 9))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex = ax1, sharey = ax1)
    #ax3 = fig.add_subplot(313, sharex = ax1, sharey = ax1)
    
    y = np.zeros(len(wavelength_spec))
    
    ax1.set_title(file)
    ax1.set_ylim(-5, 50)
    #ax2.set_ylim(-5, 75)
    
    ax1.plot(wavelength_spec, spectra)
    ax1.plot(wavelength_spec, y)
    
    ax2.set_title('Continuum Subtracted Spectrum')
    ax2.plot(wavelength_spec, spectra-y_continuum.value)
    ax2.plot(wavelength_spec, y)
    
    #ax3.set_title('Medfiltered Continuum Subtracted Spec')
    #ax3.plot(wavelength_spec, medfilt(spectra-y_continuum.value, kernel_size = 5))
    #ax3.plot(wavelength_spec, y)
    
    fig.tight_layout()
    plt.show()
    
    #plt.figure(figsize = (16, 8))
    #plt.title('Testing')
    #plt.xlabel(r'Wavelength [$\AA$]')
    #plt.plot(wavelength_spec, spectra)
    #plt.plot(wavelength_spec[filt], spectra[filt])
    #plt.show()
    '''
    
    continuum_subtracted_spec = medfilt(spectra-y_continuum.value, kernel_size = 5)

    return continuum_subtracted_spec, continuum_fit, filt_noise

def analysis(flux, wavelength, filt, emission_lines, z, continuum_func):
    '''
    
    This function will take in a 1D extracted spectrum and the emission QTables gathered from the 
    find_lines_derivatives or find_lines_derivatives.
    
    Parameters
    -----------------
    flux: This is the flux of the spectrum we will do our analysis for
    wavelength: The wavlength corresponding to the flux
    emission_lines: a list of emission lines to look for
    
    Output
    ------------------
    Analysis of the emissions lines passed in.
    
    emission_lines: the line center in angstrom
    lines_flux: the flux of the lines given in ergs/(s cm^2)
    e_width: the equivalent width of the line
    
    '''
    
    line_wave = emission_lines * (1+z)
    
    spectrum = Spectrum1D(spectral_axis=wavelength[filt]*u.angstrom, flux =flux[filt]*u.erg/u.s/u.cm/u.cm/u.angstrom)
    
    lines = find_lines_derivative(spectrum, flux_threshold=4)
    emission = lines[lines['line_type'] == 'emission']
    
    #plt.figure(figsize=(10,10))
    #plt.title('Checking Emission Lines')
    #plt.plot(spectrum.spectral_axis, spectrum.flux)
    
    #for i in emission['line_center'].value:
    #    plt.axvline(i)
    
    #plt.show()
    
    #making a window to look around the line center so that I can do some analysis
    window = 15*u.angstrom
    
    #making the lists so that I can append the analysis later
    emission_lines = []
    lines_flux = []
    e_width = []
    manual_ew = []
    line_f= []
    
    #for loop that goes through each of the emission lines and does the necessary analysis
    for i in emission['line_center']:
        
        emission_lines.append(i)
        lines_flux.append(line_flux(spectrum, SpectralRegion(i-window, i+window)))
        e_width.append(equivalent_width(spectrum, regions=SpectralRegion(i-window, i+window), continuum=continuum_func(i)))
        manual_ew.append(line_flux(spectrum, SpectralRegion(i-window, i+window))/continuum_func(i))
        
        #fitting a gaussian to the data and extracting the line flux
        sub_region = SpectralRegion(i - window, i + window)
        sub_spectrum = extract_region(spectrum, sub_region)
        param = estimate_line_parameters(sub_spectrum, models.Gaussian1D())
        g_init = models.Gaussian1D(amplitude=param.amplitude, mean=param.mean, stddev=param.stddev)
        g_fit = fit_lines(spectrum, g_init)
        y_fit = g_fit(sub_spectrum.spectral_axis)
        
        flux_line = np.sqrt(2*np.pi)*param.amplitude.value*param.stddev.value
        
        line_f.append(flux_line)
    
    for i in range(len(e_width)):
        print(emission_lines[i].value, lines_flux[i].value, e_width[i].value, manual_ew[i].value, line_f[i])
        
    #returning the lists declared above
    #return emission_line, lines_flux, e_width
    
def smooth_function(spectrum, wavelength, window):
    
    
    smoothed_spec = []
    smoothed_wavelength = []
    
    for i in range(len(spectrum)):
        
        if i + window > len(spectrum):
            break
        else:
            median = np.median(spectrum[i: i + window])
            smoothed_spec.append(median)
            smoothed_wavelength.append(np.median(wavelength[i: i + window]))
    
    return np.array(smoothed_spec), np.array(smoothed_wavelength)

#file = 'median_J014707+135629_cem.fits'
#file = 'median_J021306+005612_mods1b_ce.fits'
#m, k = spectrum(file)

#fig = plt.figure(figsize = (16, 10))
#ax1 = fig.add_subplot(311)
#ax2 = fig.add_subplot(312)
#ax3 = fig.add_subplot(313)

for i in files[:3]:
    
    if 'cem.fits' in i:
        
        sp1 = i.split('_')[1]
        sp2 = sp1.split('+')[0]
        
        match = ID_1 == sp2
        #print(match)
        z1 = z_redshift[match]
        #print(z1)
        
        spec, wvln = spectrum(i)
        
        spectra, cont_func, filt= fitting_continuum(wvln, spec, z1, line_wavelength, i)
        analysis(spectra, wvln, filt, line_wavelength, z1, cont_func)
        
        #ax1.plot(wvln, spec, label = i)
        
    if 'mods1b' in i:
        
        sp1 = i.split('_')[1]
        sp2 = sp1.split('+')[0]
        
        match = ID_1 == sp2
        
        #print(match)
        z1 = z_redshift[match]
        #print(z1)
        spec, wvln = spectrum(i)
        spectra, cont_func, filt= fitting_continuum(wvln, spec, z1, line_wavelength, i)
        analysis(spectra, wvln, filt, line_wavelength, z1, cont_func)
        
        #ax2.plot(wvln, spec, label = i)
    
    if 'mods1r' in i:
        
        sp1 = i.split('_')[1]
        sp2 = sp1.split('+')[0]
        
        match = ID_1 == sp2
        #print(match)
        z1 = z_redshift[match]
        #print(z1)
        spec, wvln = spectrum(i)
        spectra, cont_func, filt= fitting_continuum(wvln, spec, z1, line_wavelength, i)
        analysis(spectra, wvln, filt, line_wavelength, z1, cont_func)
        
        #ax3.plot(wvln, spec, label = i)

#ax1.set_ylim(-2, 75)
#ax2.set_ylim(-2, 75)
#ax3.set_ylim(-2, 75)
#ax1.legend(loc='best')
#ax2.legend(loc='best')
#ax3.legend(loc='best')
#plt.show()    

'''            
def gain_calculations(file_spec, file_sdss, z, line_lam):
    
    
    This function calculates the gain as a function of wavelength so that our spectra is flux calibrated properly.
    
    Parameters
    ---------------
    file_spec: This is the filename of the spectrum we want to extract, Should be the median_mods...
    file_sdss: these are the sdss filenames as this is what we will be comparing it to.
    z: the redshift corresponding to the file galaxy
    line_lam: this is the list of wavelength we have to be on the lookout
    
    
    Output:
    ---------------
    the gain as a function of wavelength
    
    
    #getting the extracted spectrum from the filename
    add, avged, spec_wave = spectrum(file_spec)
    
    #opening up the sdss files
    hdu = fits.open(file_sdss)
    sdss_flux = hdu[1].data['flux']
    sdss_wave = 10**hdu[1].data['loglam']
    
    #masking the emission lines and zeros of our spectrums
    filt_spec = masking_spectra(spec_wave, avged, z, line_lam)
    filt_sdss = masking_spectra(sdss_wave, sdss_flux, z, line_lam)
    
    #reducing out the spectras
    reduced_spec, reduced_wvln = avged[filt_spec], spec_wave[filt_spec]
    reduced_sdss_spec, reduced_sdss_wvln = sdss_flux[filt_sdss], sdss_wave[filt_sdss]
    
    #making an interpolation function of the file_spec
    f = interp1d(sdss_wave, sdss_flux)

    #here we map the sdss flux stuff to the reduced_wvln
    gain_spec = f(reduced_wvln)
    
    #we smooth out the two fluxes in a given window-size
    spec_smooth, wave_smooth = smooth_function(reduced_spec, reduced_wvln, 21)
    sdss_smooth, sdss_smooth_wave = smooth_function(gain_spec, reduced_wvln, 21)
    
    #calculating the gain function
    c = sdss_smooth/spec_smooth
    
    plt.figure(figsize = (10, 10))
    plt.title('Gain')
    #plt.ylim(-10000,10000)
    #plt.plot(reduced_wvln, c)
    #plt.plot(wave_smooth, spec_smooth)
    #plt.plot(sdss_smooth_wave, sdss_smooth)
    plt.ylim(-10000, 10000)
    #plt.plot(wave_smooth, c)
    plt.show()


#masking_spectra(spec_wave, avged, z)
filt_wave, filt_spec = masking_spectra(spec_wave2, avged2, z, line_wavelength)
filt_sdss_wave, filt_sdss_flux = masking_spectra(sdss_wave, sdss_flux, z, line_wavelength)
plt.figure(figsize = (16, 8))
plt.plot(filt_wave, filt_spec, label = 'My Spectrum')
plt.plot(filt_sdss_wave, filt_sdss_flux, label = 'SDSS')
plt.legend(loc='best')
plt.show()


add1, avged1, spec_wave1 = spectrum(file_spec)
z=0.057
sdss_flux1 = fits.getdata(file_sdss, ext=1)['flux']
sdss_wave1 = 10**fits.getdata(file_sdss, ext=1)['loglam']

filt_wave, filt_spec = masking_spectra(spec_wave1, avged1, z, line_wavelength)
filt_sdss_wave, filt_sdss_flux = masking_spectra(sdss_wave1, sdss_flux1, z, line_wavelength)

plt.figure(figsize = (16, 8))
plt.plot(filt_wave, filt_spec, label = 'My Spectrum')
plt.plot(filt_sdss_wave, filt_sdss_flux, label = 'SDSS')
plt.legend(loc='best')
plt.show()
'''
