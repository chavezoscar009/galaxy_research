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
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from scipy.stats import norm
from specutils import Spectrum1D, SpectralRegion
from specutils.analysis import line_flux
from astropy import units as u



#grabbing all of micaela's fits files from the current directory
files = [x for x in glob.glob('*.fits') if 'SDSS' not in x]

#getting all the sdss fits files so we can do flux calibration
sdss_files = [x for x in glob.glob('*.fits') if 'SDSS' in x]

#gettting only good file by getting rid of my box_data.fits file
#good_files = files[1:]

#reading in the dat file
ID, redshift = np.genfromtxt('targets.dat', usecols=(0,1), unpack = True, skip_header=2, dtype = 'str')
z = redshift.astype(float)

line_info = np.genfromtxt('linelist.dat', unpack =True, dtype='str', usecols=(0,1), skip_header = 1)

line_wavelength = line_info[0].astype(float)
line_name = line_info[1]


def fitting_gaussian(data, wavelength):
    
    '''
    This function will find the gaussian fit to the column data after it is boxed. Meaning we know where the center of the
    spectrum is at and we added plus or minus 50, in our case to it. Then make a spectrum from it by finding the maximum.

    Parameter
    -------------
    data: this is the data that we would like to fit with a gaussian model. This could be anything from emission lines to columns of the boxed
          data sets.

    Output
    -------------
        
    
    '''
    
    x = np.arange(len(data[:,0]))
    
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
    
    while True:
        
        median_data = np.median(data[:, start:end], axis = 1)
        med_wvln = np.median(wavelength[start:end])
        
        #print(median_data)
        #print(x)
        #print()
        
        #if True in np.isnan(median_data) or True in np.isinf(median_data):
        #    print('Got an Error Here!!!')
        
        #print(x)
        #print()
        popt, covar = curve_fit(f, x, median_data, p0 = [np.amax(median_data), len(median_data)//2, 40], bounds=[(0, 0, 1), (np.amax(median_data), len(data[:, 0]), 50)])
        
        std_dev = popt[-1]
        
        sig.append(std_dev)
        wvln.append(med_wvln)
        
        start = start + N + skip
        end = start + N
        
        #print(start)
        #print(end)
        #print()
        
        if start > len(data[0,:]) or end > len(data[0,:]):
            break
            
        p0 = [np.amax(data[:,i]), len(data[:,0])//2, 25],
        popt, covar = curve_fit(f, x, data[:, i], bounds=[(0, 0, 1), (np.amax(data), len(data[:, 0]), 50)])
        
        #this part makes the gaussian function with the parameters fit from above
        #y = f(x, *popt)
        
        #y = norm.pdf(x, len(median_data)//2, 4)
        
        #test = np.random.norm(len(median_data)//2, 4, 1000)
        
        #plt.hist(test)
        #plt.plot(x, y)
    
    
    #line = np.polyfit(wvln[4:], sig[4:], deg=1)
    
    #x = np.linspace(wvln[0], wvln[-1], 1000)
    #y = line[0]*x + line[-1]
    
    y = norm.pdf(x, len(median_data)//2, 4)
    #print(len(y))
    
    return y

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
    #We used a function called finding_row_center to find the index of the masked data
    row_spectrum = finding_row_center(data*filt)

    #window = 50
    
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
    
    #trying out the gaussian function
    gauss_mult = fitting_gaussian(boxed_data, wvln_spec)
     
    #plt.figure(figsize = (10,10))
    #plt.imshow(gauss_mult*boxed_data, origin='lower', cmap='gray', norm = LogNorm())
    #plt.colorbar()
    #plt.show()
    
    gauss_filtered = (boxed_data.T * gauss_mult/np.amax(gauss_mult)).T
    
    plt.figure(figsize = (10,10))
    plt.imshow(gauss_filtered, origin='lower', cmap='gray', norm = LogNorm())
    plt.colorbar()
    plt.show()
    
    gauss_added = np.sum(gauss_filtered, axis = 0)
    
    #plt.figure(figsize = (16, 6))
    #plt.plot(wvln_spec, gauss_added)
    #plt.show()
    
    np.savez('spectra.npz', flux=gauss_added, wave = wvln_spec)
    
    return adding_target, wvln_spec
    
    #straight summin up the columns together
    #spectrum = np.sum(boxed_data, axis = 0)
    
    '''
    wavelength = p(x, y)

    #code that plots it so that I can see what the 1D spectrum looks like
    fig = plt.figure(figsize = (14,8))
    
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex = ax1)

    ax1.set_title('Graph of Spectrum using Summation ')
    ax1.set_ylabel('Intensity')
    ax1.plot(wavelength, spectrum)
    
    ax2.set_title('Graph of the Spectrum using Max Column Values')
    ax2.set_ylabel('Intensity')
    ax2.set_xlabel(r'Wavelength [$\AA$]')
    ax2.plot(wavelength, max_values_col)
    
    fig.tight_layout()

    plt.show()
    

    return spectrum, max_values_col
    
    '''   

def masking_spectra(wavelength_spec, spectra, z, line_lam):
    
    '''
    This function will make a boolean filter to mask out the emission lines and any zeros we have in the spectra
    
    
    Parameter
    ------------
    wavelength_spec: this is the wavelength of the extracted spectrum we got from the spectrum function
    spectra: the values of the extracted spectra, need this to mask out all the zero value
    
    NOTE: wavelength_spec and spectra need to be the same length
    
    z: redshift
    line_lam: this is a list of all the line we are interested and their rest frame wavelength
    
    Output
    ------------
    filt: boolean filter masking out emission lines and places where zeros occur.
    
    '''
    
    #making a boolean filter the length of wavelength and spectra 
    filt = np.ones(len(wavelength_spec), dtype = bool)
    
    
    #checking spectra values to see if we get zero if we do then we change those index values to False
    for i, val in enumerate(spectra):
        
        #checking if the spectra has zeros
        if val == 0:
            filt[i] = False
    
    #this will mask out the emission lines so that we can fit the continuum and get a reasonable gain fit
    for j in line_lam:
        
        if j * (1+z) < wavelength_spec[0] or j * (1+z) > wavelength_spec[-1]:
            continue
        else:    
            index = np.abs(wavelength_spec - j*(1+z)).argmin()
            #masking emission lines
            #I can copy and paste this until we have the right number of emmission lines
            window = 30
            filt[index-window:index+window] = False
            
    #print(len(spectra))        
    #print(len(spectra[filt]))
    
    #peaks, prop = find_peaks(spectra, height=.2, distance=40, width = 3)
    
    #testing code by plotting
    
    #plt.figure(figsize = (16, 8))
    #plt.title('Testing')
    #plt.xlabel(r'Wavelength [$\AA$]')
    #plt.plot(wavelength_spec, spectra)
    #plt.plot(wavelength_spec[filt], spectra[filt])
    #plt.show()
    
    return filt

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

file = 'median_J014707+135629_cem.fits'

x = spectrum(file)

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
