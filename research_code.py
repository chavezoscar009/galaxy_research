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
from astropy.table import Table, Column, vstack
import time
#from sdss_catalog_data import plotting_BPT

start = time.time()

#grabbing all of micaela's fits files from the current directory
files = [x for x in glob.glob('*.fits') if 'SDSS' not in x and 'galSpec' not in x]

file_num = np.unique([x.split('_')[1] for x in files])

#print(files)
#getting all the sdss fits files so we can do flux calibration
#sdss_files = [x for x in glob.glob('*.fits') if 'SDSS' in x]



#reading in the dat file
ID, redshift, see = np.genfromtxt('targets.dat', usecols=(0,1, 8), unpack = True, skip_header=2, dtype = 'str')

ID1 = [x.split('.')[0] for x in ID]

filt_z = np.zeros(len(ID1), dtype = bool)

for j, val in enumerate(ID1):
    for i in file_num:
        if val in i:
            filt_z[j] = True 

z = redshift.astype(float)

ID_1 = np.array(ID1)[filt_z]
z_redshift = np.array(z)[filt_z]

seeing = np.array(see.astype(float))[filt_z]


line_info = np.genfromtxt('linelist.dat', unpack =True, dtype='str', usecols=(0,1), skip_header = 1)

line_wavelength = line_info[0].astype(float)
line_name = line_info[1]


def fitting_gaussian(data, wavelength, filename, window):
    
    '''
    This function will fit a gaussian to median combined coulmn data corresponding to a window around the trace. 
    We do this so that the maximum intensity gets the most weight and points outside the center get less weight
    
    Parameter
    -------------
    data: this is the data that we would like to fit with a gaussian model, This should be the 2D spectrum data with a filter. 
    wavelength: the eavelength for every column pixel
    filename: name of the file
    window: the window used in the box to get the data
        

    Output
    -------------
    renorm_y: a normalized gaussian with the parameters that best suits trace of the data
    
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
    
    #this will hold the sigma valueas and wavelength value of the median gaussian fit
    sig = []
    wvln = []
    
    #N represent how big of a chunk to median 
    N = 21
    
    #skips 50 columns so that we do not get alot of data points
    skip = 50
    
    #start and end are where in the column array we focus and median out for the gaussian fit
    start = 0
    end = N
    
    #counter variable used for debugging purposes
    i = 0
    
    #plotting code to see how the sigma vs wavelength relation was like
    #plt.figure(figsize = (10,10))
    #plt.title(filename + ': Sigma vs wvln')
    
    while True:
        
        #here we take the median of the column data between start and end
        median_data = np.median(data[:, start:end], axis = 1)
        
        #here we take the median wavelength between start and end
        med_wvln = np.median(wavelength[start:end])
        
        # this checks to see if the maximum median value is zero we increment start and end and we pass on this data
        #this would not work with the gaussina fitting method
        if np.amax(median_data) == 0:
            start = start + N + skip
            end = start + N
            continue
        
        ################
        #FITTING
        #--------------
        #If everything is good then we fit the data with a gaussian using curve_fit below
        ################
        
        #We pass into curve fit: x which is the pixel values of the rows
        #                        median_data which is the median column 
        #Initial Guesses: A: max value of the median_data, Mean: middle of the median data, sigma: 40
        #
        #Bounds: A: 0-max(median_data)
        #        mean: 0-len(median_data)
        #        sigma: 1-100
        
        try:
            #popt is the best fit parameter values
            #covar is error on the parameters
            popt, covar = curve_fit(f, x, median_data, 
                                    p0 = [np.amax(median_data), len(median_data)//2, window], 
                                    bounds=[(0, 0, 1), (np.amax(median_data), len(data[:, 0]), 100)])
        except RuntimeError:
            #incrementing start and end column values
            start = start + N + skip
            end = start + N
            continue
        
        #std_dev holds the standard deviation for the gaussian fit
        std_dev = popt[-1]
        
        #appending the sigma and wavelength to the list
        sig.append(std_dev)
        wvln.append(med_wvln)
        
        #incrementing start and end column values
        start = start + N + skip
        end = start + N
        
        #print(start)
        #print(end)
        #print()
        #plt.plot(wvln, medfilt(sig, kernel_size = 7))
        #plt.show()
        
        #checks to see if start or end is beyond the length of the column of the 2D array
        #if it is then this breaks from the while loop
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


    #plotting code for debugging
    #plt.title('Sigma vs Wavelength: ' + filename[7:-5])
    #plt.plot(wvln, sig, 'r-')
    #plt.show()
    
    if 'cem.fits' in filename:
        
        #median filters the sigma list
        sig_filt = medfilt(sig, kernel_size = 7)
        
        #gets the maximum sigma in the resulting list
        sigma = np.amax(sig_filt)
        
        #constructs a normalized gaussian with the mean at the halfway point of the median data and sigma from above
        y = norm.pdf(x, len(median_data)//2, sigma)
        
        #this renormalizes the gaussian
        renorm_y = y/np.amax(y)
        
        return renorm_y
        
        '''
        #this variable holds the middle portion of the filename
        l = t[1]
        
        test = np.percentile(sig_filt, 70, interpolation='midpoint')
        mu = np.mean(sig_filt)
        
        v = test * np.ones(len(sig_filt))
        w = 2*mu * np.ones(len(sig_filt))
        U = mu * np.ones(len(sig_filt))
        
        #plotting code
        plt.plot(wvln, sig_filt, 'r-', alpha = .7)
        plt.plot(wvln, v, 'k--', label = 'Percentile')
        plt.plot(wvln, w, 'y--', label = '2*Mean')
        plt.plot(wvln, U, 'b--', label = 'Mean')
        plt.legend(loc='best')
        #plt.savefig('Sigma_vs_Wavelength_'+l+'_cem.pdf')
        plt.show()
        '''
    
    elif 'mods1b' in filename:
        
        #in this we have to do a prior step where we have to remove bad portions of the detector because of low sensitivity
        removing_bad_sensitivity = np.array([True if x > 3500 and x < 5500 else False for x in wvln])
        
        #this is the maximum of the filtered sigma list
        sigma = np.amax(np.array(sig)[removing_bad_sensitivity])
        
        #makes a gaussian with the paremeters mean = center of median data and sigma from above
        y = norm.pdf(x, len(median_data)//2, sigma)
        
        #normalizes the gaussian
        renorm_y = y/np.amax(y)
        
        return renorm_y
    
    elif 'mods1r' in filename:
        
        #in this we have to do a prior step where we have to remove bad portions of the detector because of low sensitivity
        removing_bad_sensitivity = np.array([True if x > 6000 and x < 9000 else False for x in wvln])
        
        sigma = np.amax(np.array(sig)[removing_bad_sensitivity])
        
        #this is the maximum of the filtered sigma list
        y = norm.pdf(x, len(median_data)//2, sigma)
        
        #normalizes the gaussian
        renorm_y = y/np.amax(y)
        
        return renorm_y
        
    '''
        #the two variables below hold the middle portions of the filename 
        #these are the distiguishing features as they tell us which object it is
        l = t[1]
        m = t[2]
        
        test = np.percentile(sig_filt, 90)
        v = test * np.ones(len(sig_filt))
        
        mu = np.mean(sig_filt)
        w = 2*mu * np.ones(len(sig_filt))
        U = mu * np.ones(len(sig_filt))
        
        #plotting code
        plt.plot(wvln, sig_filt, 'r-', alpha = .7)
        plt.plot(wvln, v, 'k--')
        plt.plot(wvln, w, 'y--', label = '2*Mean')
        plt.plot(wvln, U, 'b--', label = 'Mean')
        plt.legend(loc='best')
        #plt.savefig('Sigma_vs_Wavelength_'+l+m+'.pdf')
        plt.show()
    
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
    '''

def fitting_lines(spectrum, sig = 4):
    '''
    This function fits the emission lines from a spectrum object. The way that this is used for is if you have a sub_spectrum of where you think the emission line is at +/- a window.
    Ie. if you have a line at 5009 angstroms and a window of 10 angstroms then the spectrum passed in must be a subspectrum between 4999 and 5019. Essential your emission line
    
    Parameters
    ------------------------
    spectrum: This is a sub_spectrum that holds your emission line
    
    Returns
    -----------------------
    param: this is the parameters for the gaussian function param = [Amplitude, Mean, Sigma]
    '''
    
    #defining the gaussian function I would like to fit to the data
    def f(x, A, mu, sig):
        return A * np.exp(-(x-mu)**2/(2*sig**2))
    
    #code that tries to fit the data with a gaussian
    try:
        #gets the mean value from the spectrum passed in
        mu = np.mean(spectrum.spectral_axis.value)
        #performs the fitting
        param, covar = curve_fit(f, spectrum.spectral_axis.value, spectrum.flux.value, 
                                 p0 = [np.amax(spectrum.flux.value), mu, sig])
        return param
    
    #If it could not fit the data and we get a Runtime Error we return an array of -999. 
    #We can use this as a filter because we know that wavelengths won't be negative, so we get a param[1] < 0
    except RuntimeError:
        return np.array([-999,-999,-999])
        
    #return param
    #amp = param[0]
    #mu = param[1]
    #sig = param[-1]
    
    #except RuntimeError:
    #    pass

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
    #to see which row occurs the most. This should be the center of the spectrum or very close to it.
    row_index = []
    
    #this code below checks each column and finds the maximum value for the respective column. 
    for i in range(len(data[0, :])):
        
        #this index variable holds the index where the max value occurs in the column which translates to the row
        index = np.argmax(data[:, i])
        row_index.append(index)
    
    #Getting the row index using mode
    row_ind = mode(row_index)[0][0]

    return row_ind

def extraction_error(data, trace_center, window, filename, wavelength, gauss_mult):
    
    #making an array that will go from 10 below and 10 above the trace and try to extract a 1D spectrum at each increment
    away_from_center = np.linspace(-10, 10, 21)
    
    spec1D = []
    
    for i in away_from_center:
        
        start = int((trace_center + i) - window)
        end = int((trace_center + i) + window)
        
        #print(type(start))
        #print(type(end))
        
        data_one = data[start : end, :]
        
        #gauss_mult = fitting_gaussian(data_one, wavelength, filename, window)
        
        gauss_filtered = (data_one.T * gauss_mult).T
    
        gauss_added = np.sum(gauss_filtered, axis = 0)
        
        spec1D.append(gauss_added)
        
    return np.array(spec1D)

def spec_error(spectras):
    
    subset_of_spec = spectras[2::4]
    
    consecutive_specs = spectras[7:14]
    
    center_trace = consecutive_specs[3]
    
    nrow = 5
    ncol = 1
    #fig, axs = plt.subplots(nrow, ncol, figsize = (14, 10))
    #for i, ax in enumerate(fig.axes):
    #    ax.plot(subset_of_spec[i])
    #fig.tight_layout 
    #plt.show()
    
    #nrow = 11
    #ncol = 2
    #fig, axs = plt.subplots(nrow, ncol, figsize = (14, 10), sharex=True, sharey=True,  constrained_layout=True)
    #y = np.zeros(len(center_trace))
    #for i, ax in enumerate(fig.axes):
    #    if i == len(spectras):
    #        continue
    #    ax.plot(spectras[i]-center_trace, 'k', alpha = .5)
    #    ax.plot(y, 'y-')
    #    ax.set_ylim(-5, 5)
    #plt.show()

def spectrum(file):
    
    '''
    This function will try to convert any 2D spectral array and convert that into a 1D array.

    Parameters
    --------------
    file: this is the filename of the file we want to extract a 1D spectrum from.

    Output
    --------------
    spectrum: this is the 1D reduced spectrum gathered from the 2D array we passed in.
    wavelength: the wavelengths corresponding to the extracted spectrum
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
    
    #making a window variable s0 that i can change it accordingly to what I want to look at
    window = 0
    
    
    if '_cem.fits' in file:
            
        row_min = 220
        row_max = 500
        
        filt[:row_min, :] = np.zeros(len(data[0,:]))
        filt[row_max:, :] = np.zeros(len(data[0,:]))
        
        window = 25
    
    if 'b_ce.fits' in file:

        row_min = 1410
        row_max = 1890

        filt[:row_min, :] = np.zeros(len(data[0,:]))
        filt[row_max:, :] = np.zeros(len(data[0,:]))
        
        if 'J021306' in file:
            window = 40
    
        if 'J082540' in file:
            window = 35
        
        if 'J073149' in file:
            window = 30
            
    if 'r_ce.fits' in file:

        row_min = 1350
        row_max = 1800

        filt[:row_min, :] = np.zeros(len(data[0,:]))
        filt[row_max:, :] = np.zeros(len(data[0,:]))
        
        if 'J021306' in file:
            window = 40
    
        if 'J082540' in file:
            window = 35
        
        if 'J073149' in file:
            window = 40
    
    #print(file)
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
    
    y = row_spectrum* np.ones(len(data[0,:]))
    
    #plt.figure(figsize = (10,10))
    #plt.imshow(data*filt, origin = 'lower', norm = LogNorm(), cmap = 'gray')
    #plt.plot(y)
    #plt.show()
    
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
    
    #getting the spectrum wavelength and converting it to rest frame wavelengths
    wvln_spec = wvln_arr[row_spectrum,:]
    
    #just adding the boxed_data
    adding_target = np.sum(boxed_data, axis = 0)
    
    #this returns a gaussian fit to the spectrum so that more 
    gauss_mult = fitting_gaussian(boxed_data, wvln_spec, file, window)
    
    #this gets the distribution of spectrums as we move away from the trace
    spec1D_dist = extraction_error(data, row_spectrum, window, file, wvln_spec, gauss_mult)
    
    #gets the error for the spectrum extraction
    spec_error(spec1D_dist)
    
    #this is the gaussian filtered data set after getting multiplied by the gaussian along the trace
    gauss_filtered = (boxed_data.T * gauss_mult).T
    
    #adding the resulting array to get the 1D spectrum
    gauss_added = np.sum(gauss_filtered, axis = 0)
    
    ''' 
    plt.figure(figsize = (10,10))
    plt.imshow((gauss_mult*boxed_data.T).T, origin='lower', cmap='gray', norm = LogNorm())
    plt.colorbar()
    plt.show()
    
    fig = plt.figure(figsize = (16, 10))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex = ax1)
    ax3 = fig.add_subplot(313, sharex = ax2)
    
    ax1.set_title('Adding Rows')
    ax1.plot(wvln_spec, medfilt(adding_target, kernel_size = 5))
    
    ax2.set_title('Gaussian Multiplied Rows')
    ax2.plot(wvln_spec, medfilt(gauss_added, kernel_size = 5))
    
    ax3.set_title('Both')
    ax3.plot(wvln_spec, medfilt(adding_target, kernel_size = 5), 'r-', alpha = .6, label = 'Adding')
    ax3.plot(wvln_spec, medfilt(gauss_added, kernel_size = 5), 'k--', label = 'Gauss Added')
    ax3.legend(loc = 'best')
    
    fig.tight_layout()
    
    plt.show()
    '''
    
    #np.savez('spectra.npz', flux=gauss_added, wave = wvln_spec)
    
    return medfilt(gauss_added, kernel_size = 5), wvln_spec 
  

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
    
    #making boolean filters the length of wavelength and spectra 
    filt = np.ones(len(wavelength_spec), dtype = bool)
    filt_noise = np.ones(len(wavelength_spec), dtype = bool)
    filt_z = np.ones(len(wavelength_spec), dtype = bool)
    
    #checking spectra values to see if we get zero if we do then we change those index values to False
    for i, val in enumerate(spectra):
        
        #checking if the spectra has zeros
        if val == 0:
            filt[i] = False
    
    #going through the catalog with rest frame emission lines to mask out emission lines
    for j in line_lam:
        
        #this checks if the line we are looking at is within the wavelength range of our spectrum if its not then we go to the next one
        if j *(1+z) < wavelength_spec[0] or j*(1+z) > wavelength_spec[-1]:
            continue
        else:
            
            #this finds the index in our wavelength array that is nearest to the redshifted rest line
            index = np.abs(wavelength_spec - (j*(1+z))).argmin()
            
            #makes a window of 20 indices to maske out
            window = 20
            
            #if the index minus window is negative this means that the left window is close to the beginning wavelength regime 
            #and so we mask out the beginning up to index + window
            if (index-window) < 0: 
                
                filt[:index+window] = False    
            
            #if index + window is bigger than the len of the wavelngth array then that means it is out of bounds and 
            #so we mask from index-window to the end of the wavelength array
            if index + window > len(filt):
                filt[index-window:] = False
            #if all the above conditions fail then it masks out a region around the emission line
            
            else:
                filt[index-window:index + window] = False
    
    if 'cem.fits' in file:
        
        #finds the index where noisy areas are at in the cem files
        ind = np.where(wavelength_spec < 3824)
        
        #masking them out in the respective filters
        filt[ind] = False
        filt_noise[ind] = False
        filt_z[ind] = False
        
    elif 'mods1b' in file:
        
        #getting the indices where the noisy areas are at in the spectrum
        ind = np.where(wavelength_spec < 3700)
        ind2 = np.where(wavelength_spec > 5500)
        
        #making a master index list whih we will pass into filt and make those false
        ind_tot = np.concatenate((ind, ind2), axis=None)
        
        #making each filter assignment to false
        filt[ind_tot] = False
        filt_noise[ind_tot] = False
        filt_z[ind_tot] = False
        
    elif 'mods1r' in file:
        
        #finding the indices where noisy parts are
        #these two are better used for the continuum fitting
        ind = np.where(wavelength_spec < 5700)
        ind2 = np.where(wavelength_spec > 8700)
        
        #This one will be joined with ind to make a more complete emission line spectrum
        ind3 = np.where(wavelength_spec > 10000)
        
        #making a master indices 
        ind_tot = np.concatenate((ind, ind2), axis=None)
        
        #filtering out noise
        filt[ind_tot] = False
        filt_z[ind_tot] = False
        
        #filtering less noise
        ind_noise = np.concatenate((ind, ind3), axis=None)
        filt_noise[ind_noise] = False
    
    
    ##############
    #testing the masking filter and see how well it fits the continuum
    ##############
    
    #making a specutils spectrum object from the filtered data
    spectrum = Spectrum1D(spectral_axis=wavelength_spec[filt]*u.angstrom, flux = spectra[filt]* u.erg/u.s/u.cm/u.cm/u.angstrom)
    
    
    

    filt_wvln = wavelength_spec[filt]
    filt_flux = spectra[filt]
    
    std_dev = []
    med_noise = []
    
    for k in line_lam:
        
        if k < filt_wvln[0] or k > filt_wvln[-1]:
            std_dev.append(-999)
            med_noise.append(-999)
            continue
            
        nearest_ind = abs(k - filt_wvln).argmin()
        noise_window = 15
        
        start = nearest_ind - window
        end = nearest_ind + window
        
        std = np.std(filt_flux[start : end])
        med = np.median(filt_flux[start : end])
        
        std_dev.append(std)
        med_noise.append(med)
    
    #for i in range(len(line_lam)):
    #    print(' %10.2f -------------- %5.2f' %(line_lam[i], std_dev[i]))
    #print()
    
    
    #this funciton fits the continuum and gives back a function that we can pass in x values
    continuum_fit = fit_generic_continuum(spectrum, model=models.Linear1D())
    
    #getting the continuum
    y_continuum = continuum_fit(wavelength_spec*u.angstrom)
    
    '''
    #testing code by plotting
    fig = plt.figure(figsize=(14, 9))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex = ax1, sharey = ax1)
    #ax3 = fig.add_subplot(313, sharex = ax1, sharey = ax1)
    
    y = np.zeros(len(wavelength_spec))
    
    ax1.set_title(file + ' Continuum Subtraction')
    ax1.set_ylim(-5, 50)
    #ax2.set_ylim(-5, 75)
    
    ax1.plot(wavelength_spec, spectra)
    ax1.plot(wavelength_spec, y_continuum, label = 'y = 0')
    
    ax2.set_title('Continuum Subtracted Spectrum')
    ax2.plot(wavelength_spec, spectra-y_continuum.value)
    ax2.plot(wavelength_spec, y, label = 'y = 0')
    
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
    
    #this subtracts the continuum from the spectra
    continuum_subtracted_spec = medfilt(spectra-y_continuum.value, kernel_size = 5)
    
    #returns a continuum subtracted spectrum, the continuum fit function, filt_noise, filt_z for the spec_redshift function, st
    return continuum_subtracted_spec, continuum_fit, filt_noise, filt_z, np.array(std_dev), np.array(med_noise)

def spec_redshift(flux, wavelength, filt, line, z, percentile, filename, line_name):
    
    '''
    This function will calculate the spectroscopic redshift of the galaxy by finding the max flux value of each file and comparing that 
    to the max emission line we expect to see there. In our case the cem and mods1b files will have OIII and the mods1r will have Halpha
    
    Parameters
    ----------------------
    flux: an array for the flux values of the spectrum
    wavelength: this is the wavlength of the spectrum
    filt: this is a filter to help get rid of noisy areas of the spectrum 
          and we need that because we do not want to get false emisison lines due to very noisy areas in spectrum
    line: the rest frame line catalog that we want to check against, This is used in some debugging code
    percentile: the percentile needed to automatically find the strongest emission lines which we set to 99
    filename: the name of the file
    line_name: the name of the rest line in the catalog
    
    Outputs
    ----------------------
    The redshoft for the galaxy found using spectroscopy
    '''
    
    #inintializing the gauss function
    def f(x, A, mu, sig):
        return A * np.exp(-(x-mu)**2/(2*sig**2))
    
    #making a spectrum 1D object
    spectrum = Spectrum1D(spectral_axis=wavelength[filt]*u.angstrom, 
                              flux =flux[filt]*u.erg/u.s/u.cm/u.cm/u.angstrom)
    
    #sorting the data to pass into np.percentile
    data = np.sort(flux[filt])
    
    #making the threshold for the flux to be above a percentile value
    threshold = np.percentile(data, q=percentile, interpolation='midpoint')
    
    #finding lines using specutils line_derivative function
    lines = find_lines_derivative(spectrum, flux_threshold=threshold)

    #getting only the emission lines
    emission = lines[lines['line_type'] == 'emission']
    
    #making the lists so that I can append the analysis later
    
    #has the emission lines found by specutils
    emission_lines = []
    
    #emission line center found by curve_fitting gaussians
    curve_line = []
    
    #the line flux found by specutils 
    lines_flux = []
    
    #line flux found by gaussian analytical integration
    line_f = []
    
    #plt.plot(spectrum.spectral_axis, spectrum.flux, alpha = .5)
    #plt.title(filename[7:-5])
    
    #for loop that goes through all the emission lines that we specutils have found
    for i in emission['line_center']:
       
        #making a window to look around the line center so that I can do some analysis using specutils
        #as well as my own homemade functions
        window = 15*u.angstrom
        
        #looking at the sub_region around where the line center is located at +/- 15 Angstroms
        sub_region = SpectralRegion(i - window, i + window)
        
        #geting only that part of the spectrum
        sub_spectrum = extract_region(spectrum, sub_region)
        
        #calculating the emission line of the sub_region using specutils functions
        lines_flux.append(line_flux(sub_spectrum).value)
        
        #appending the emission line center found by specutils
        emission_lines.append(i.value)
        
        #this calls a function which fits the sub region with a gaussian
        par = fitting_lines(sub_spectrum)
        
        #if fitting lines could not fit the line it will return aan array of [-999. -999, -999] an so we check the wavlength center
        #par[1] if its negative we simply skip over this line
        if par[1] < 0:
            continue
        
        #plt.plot(sub_spectrum.spectral_axis, sub_spectrum.flux, alpha = .5)
        #plt.plot(x.value, f(x.value, *par), label = 'Curve Fit')
        #plt.axvline(i.value, linewidth=.5, linestyle='--', color = 'red')
        #plt.axvline(par[1], linewidth=.5, linestyle='--', color = 'blue')
        
        #caluclates the analytical gaussian value
        flux_line = np.sqrt(2*np.pi)*abs(par[0])*abs(par[-1])
        
        #appends the line center and the line flux calculated above
        curve_line.append(par[1])
        line_f.append(flux_line)
    
    #plt.show()
    max_specutils_flux = np.array(lines_flux).argmax()
    max_gauss_flux = np.array(line_f).argmax()
    
    #debugging code and a check to see if my analytical flux calculations match up okay with specutils 
    #for i in range(len(line_f)):
    #    print('%9.2f ----------- %9.2f ' %(line_f[i], curve_line[i]))
    #    print('%9.2f ' %(lines_flux[i]))
    #    print()
    
    #intializing the redshift variable to zero but will change depending on the file we have
    redshift = 0
    
    if 'cem.fits' in filename or 'mods1b' in filename:
        
        #checking the cem and mods1b files against the OIII5007 line
        OIII = 5007
        
        #this is a strong line and should be the one with the maximum flux value so we take argmax of line_f and pass that
        #index into the curve_line
        
        #then subtract that line center from the rest frame
        delta_lambda = curve_line[max_gauss_flux] - OIII
        
        #get redhsift by taking that difference and dividing it by rest frame wavelength
        redshift = round(delta_lambda/OIII, 5)
        #print(redshift)
    
    if 'mods1r' in filename:
        
        #for mods1r files we use the halpha line to get the redshift as that is the prominant emission line feature
        halpha = 6562.8
        
        #gtting the difference between observed and rest frame line
        delta_lambda = curve_line[max_gauss_flux] - halpha
        
        #caluclating the corresponding redshift
        redshift = round(delta_lambda/halpha, 5)
        #print(redshift)
    
    '''
    #debugging code below to check if redshift is working
   
    within_range = np.array([True if x*(1+redshift)>spectrum.spectral_axis.value[0] and x*(1+redshift)<spectrum.spectral_axis.value[-1] else False for x in line])
    #print(redshift)
    #print(within_range)
    
    reduced_line = line[within_range]
    reduced_names = line_name[within_range]
    
    
    plt.plot(spectrum.spectral_axis/(1+redshift), spectrum.flux, alpha = .5)
    plt.title(filename[7:-5])
    
    for val in curve_line:
        if val < 0:
            continue
        
        nearest = abs((val/(1+redshift)) - reduced_line).argmin()
        
        plt.axvline(val/(1+redshift), linestyle = '--', linewidth = .5, color = 'black')
        plt.axvline(reduced_line[nearest], linestyle = '--', linewidth = .5, color = 'red')
        plt.text(reduced_line[nearest], np.amax(spectrum.flux.value)/3, reduced_names[nearest], rotation = 270, fontsize = 'x-small')
        
    plt.show() 
    '''
    
    #returns the spectroscopic redshift
    return redshift
    
def analysis(flux, wavelength, filt, line, z, continuum_func, percentile, line_name, filename, seeing, conversion, line_noise, noise):
    
    '''
    
    This function will take in a 1D extracted spectrum and the emission QTables gathered from the 
    find_lines_derivatives or find_lines_derivatives.
    
    Parameters
    -----------------
    flux: This is the flux of the spectrum we will do our analysis for needs to be continuum subtracted
    wavelength: The wavelength corresponding to the flux
    filt: this is a filter to mask out noisy regions of the spectrum to so that we do not get false emission lines due to noisy regions
    line: the rest frame wavelengths of the catalog
    z = redshift of the galaxy
    continuum_func: this is the function that models the continum and we need that for manual EW calculations
    percentile: the percentile of the data we want to take as our flux threshold for finding emission lines
    line_name: the name of the lines from the catalog
    filename: name of the file
    seeing: the seeing for each object
    conversion: used for spatial analysis
    line_noise: this is a noise standard deviation that was given around emission lines and this is useful for S/N calculations
    noise: the noise value around an emission line
    
    Output
    ------------------
    Analysis of the emissions lines passed in in the form of an astropy table. will have calculated center, flux value, EW, noise and noise std_dev
    
    '''
    
    #making a gaussian function so that i can use curve_fit later to fit emission lines
    def f(x, A, mu, sig):
        return A * np.exp(-(x-mu)**2/(2*sig**2))
    
    #making a spectrum object
    spectrum = Spectrum1D(spectral_axis=wavelength[filt]*u.angstrom, 
                              flux =flux[filt]*u.erg/u.s/u.cm/u.cm/u.angstrom)
    #intializing threshold to zero but we will append to it later for each file that gets passed through
    threshold = 0
    
    #i notcied that some files did not line up nicely to some lines and so i put custom thresholds for the following files
    if 'mods1r' in filename:

        if 'J073149+404513' in filename:
            threshold = 1.4
        if 'J082540+184617' in filename:
            threshold = .4    
        
        else:    
    
            #sorting the data
            data = np.sort(flux[filt])

            #making the threshold for the flux to be above a percentile value
            threshold = np.percentile(data, q=percentile, interpolation='midpoint')
        
    elif 'J030903+003846' in filename and 'cem' in filename:

        #making the threshold for the flux to be above a percentile value
        threshold = 1.5
    
    else:    
    
        #sorting the data
        data = np.sort(flux[filt])

        #making the threshold for the flux to be above a percentile value
        threshold = np.percentile(data, q=percentile, interpolation='midpoint')
    
    '''
    
    threshold_limit_test = np.array([99, 98, 97, 96, 95, 94])
    
    plt.figure(figsize = (14, 6))
    plt.plot(spectrum.spectral_axis, spectrum.flux, 'b-')
    
    for i, val in enumerate(threshold_limit_test):
        
        data = np.sort(flux[filt])
        threshold = np.percentile(data, q=val, interpolation='midpoint')
        
        #finding lines using specutils line_derivative function
        lines = find_lines_derivative(spectrum, flux_threshold=threshold)

        #getting only the emission lines
        #emission = lines[lines['line_type'] == 'emission']

        y = (threshold)*np.ones(len(spectrum.spectral_axis))
        
        plt.plot(spectrum.spectral_axis, y , label = str(val) +' Percentile')
        
        #plt.legend(loc ='best', ncol=2)
        #for i in emission['line_center'].value:
            #plt.axvline(i, linestyle= '--', color = 'red')
            
    #y_2 = (threshold/2)*np.ones(len(spectrum.spectral_axis))
    
    #plt.plot(spectrum.spectral_axis, y_2 , label = str(threshold/2))
    plt.legend(loc ='best', ncol=2)
    plt.show()
       
    '''
    
    #making a way to automatically check for emisison lines. For this I sort the data and then pick a percentile 
    #from which anything above that will be considered an emission line.
    
    #finding lines using specutils line_derivative function
    lines = find_lines_derivative(spectrum, flux_threshold=threshold)

    #getting only the emission lines
    emission = lines[lines['line_type'] == 'emission']
    
    #making the lists so that I can append the analysis later
    
    #has the emission lines found by specutils
    emission_lines = []
    
    #finds the line center using scipy curve fit
    emission_line_fit = []
    
    #has the line fluxes caluclated by specutils
    lines_flux = []
    
    #has equivalent widths calculated using specutils ew function
    e_width = []
    
    #this has the ew width from me calculating it myself
    manual_ew = []
    
    #this obtains the line flux calculated manually using sqrt(2 pi)*A*sigma
    line_f= []
    
    #this holds the value of the continuum at the peak of the emission line
    continuum_val = []
    
    #quick check to see if emission has all the emission lines
    #plt.figure(figsize = (12, 6))
    #plt.title('Checking Emission Lines')
    #plt.plot(spectrum.spectral_axis, spectrum.flux, alpha = .5)
    
    #for i in emission['line_center'].value:
    #    plt.axvline(i, linewidth = .5, linestyle= '--')
    
    #plt.show()    
    
    #plt.figure(figsize = (12, 6))
    #plt.title('Sub Spectrum and Fitting')
    
    #for loop that goes through each of the emission lines and does flux and equivalent width analysis
    for i in emission['line_center']:
       
        #making a window to look around the line center so that I can do some analysis using specutils
        #as well as my own homemade functions
        window = 10*u.angstrom
        
        #looking at the sub_region around where the line center is located at and +/- 15 Angstroms
        sub_region = SpectralRegion(i - window, i + window)
        sub_spectrum = extract_region(spectrum, sub_region)

        #this calls a function which fits the sub region with a gaussian
        par = fitting_lines(sub_spectrum)
        
        #############
        #Note that if for some reason a gaussian cannot be fit it will return values of [-999, -999, -999] and we do not want those so we can omit these
        #essentially we will not be fitting the line and getting a flux value or EW
        #############
        
        #checks to make sure fit worked if not it skips it
        if par[1] < 0:
            continue
       
        #calculating the emission line of the sub_region
        lines_flux.append(line_flux(sub_spectrum))
        
        #appending the emission line center
        emission_lines.append(i.value)
        
        ###############
        #using specutils tools to fit lines with gaussian
        ###############
        
        #getting an initial guess on the Gaussian parameters
        param = estimate_line_parameters(sub_spectrum, models.Gaussian1D())
        
        #making an intial guess of the gaussian
        g_init = models.Gaussian1D(amplitude=param.amplitude, mean=param.mean, stddev=param.stddev)
        
        #fitting the emission line to the gaussian using values from above
        g_fit = fit_lines(spectrum, g_init)
        
        #making an x and y array for plotting, Used this for debugging purposes and check quality
        #of fit
        x = np.linspace(sub_spectrum.spectral_axis[0].value, sub_spectrum.spectral_axis[-1].value, 1000)*u.angstrom
        
        #y_values from specutils fit
        y_fit = g_fit(x)
        
        #y_values from curve_fit fit
        y_curve = f(x.value, *par)

        #getting the equivalent width of the subregion using specutils function
        e_width.append(equivalent_width(sub_spectrum,
                                        continuum=continuum_func(par[1]*u.angstrom)))
        
        #getting the flux of the line using scipy curve fit parameters
        flux_line = np.sqrt(2*np.pi)*abs(par[0])*abs(par[-1])
        
        #getting the equivalent width from flux calculation above
        manual_ew.append(flux_line/continuum_func(par[1]*u.angstrom).value)
        
        #getting the center of the emission peak from curve_fit and appending it
        emission_line_fit.append(par[1])
        
        #line_center_index.append(abs(spectrum.spectral_axis.value - par[1]).argmin()) 
        
        #appending the manual flux calculations
        line_f.append(flux_line)
        
        #appending the continuum value
        continuum_val.append(continuum_func(par[1]*u.angstrom))
        
        #plotting code where i was testing which fit was better
        
        #plt.plot(sub_spectrum.spectral_axis, sub_spectrum.flux, 'k-')
        
        #plt.plot(x.value, y_fit, 'r--', label = 'Specutils Fitting')
        
        #plt.plot(x.value, y_curve, 'y--', label = 'Curve Fit')
        
        #plt.axvline(param.mean.value, linestyle = '--', color='red')
        #plt.axvline(i.value, linestyle = '--', color='red', linewidth=.5)
        #plt.legend(loc='best')
        
        #############################
        #Code that will try to get the noise from the spectrum ot calculate the emission line S/N
        #############################
        
        #making a window to look around the line center so that I can do some analysis using specutils
        #as well as my own homemade functions
        #window = 40*u.angstrom
        
        #looking at the sub_region around where the line center is located at and +/- 15 Angstroms
        #sub_region = SpectralRegion(i - window, i + window)
        #sub_spectrum = extract_region(spectrum, sub_region)
        
    #plt.show()    
    
    He_line_names = np.array(['HeI3889', 'HeI6678', 'HeI7065'])
    He_lines = np.array([3889, 6678, 7065]) * (1+z)*u.angstrom
    He_line_flux = np.zeros(len(He_lines))
    
    #plt.figure(figsize = (14, 6))
    #plt.plot(spectrum.spectral_axis, spectrum.flux)
    
    for i, val in enumerate(He_lines):
        
        #print(val)
        
        #plt.axvline(val.value, linestyle = '--', linewidth = .5, color = 'red')
        
        #making a window to look around the line center so that I can do some analysis using specutils
        #as well as my own homemade functions
        window = 8*u.angstrom
        
        if (val - window).value < spectrum.spectral_axis.value[0] or (val - window).value > spectrum.spectral_axis.value[-1]:
            continue
        
        if (val + window).value > spectrum.spectral_axis.value[-1]:
            continue
            
        #print((val - window).value < spectrum.spectral_axis.value[0])
        
        #looking at the sub_region around where the line center is located at and +/- 15 Angstroms
        sub_region = SpectralRegion(val - window, val + window)
        sub_spectrum = extract_region(spectrum, sub_region)
        
        
        #this calls a function which fits the sub region with a gaussian and we pass in the 
        #emission center from specutils as an initial guess
        par = abs(fitting_lines(sub_spectrum))
        
        #getting the flux of the line using scipy curve fit parameters
        flux_line = np.sqrt(2*np.pi)*abs(par[0])*abs(par[-1])
        #print(par)
        #print(flux_line)
        
        He_line_flux[i] = flux_line
    
    '''
    #checking OIII line ratios
    OIII_line_names = np.array(['OIII4363', 'OIII4959', 'OIII5007'])
    OIII_lines = np.array([4363, 4959, 5007]) * (1+z) * u.angstrom
    OIII_line_flux = np.zeros(len(OIII_lines))
    
    #plt.figure(figsize = (14, 6))
    #plt.plot(spectrum.spectral_axis, spectrum.flux)
    
    for i, val in enumerate(OIII_lines):
        
        #print(val)
        
        #plt.axvline(val.value, linestyle = '--', linewidth = .5, color = 'red')
        
        #making a window to look around the line center so that I can do some analysis using specutils
        #as well as my own homemade functions
        window = 15*u.angstrom
        
        if (val - window).value < spectrum.spectral_axis.value[0] or (val - window).value > spectrum.spectral_axis.value[-1]:
            #print('Outside Window: ', val/(1+z[0]))
            continue
        
        if (val + window).value > spectrum.spectral_axis.value[-1]:
            #print('Outside Window', val/(1+z[0]))
            continue
        #print('Inside Window', val/(1+z[0]))    
        #print((val - window).value < spectrum.spectral_axis.value[0])
        
        #looking at the sub_region around where the line center is located at and +/- 15 Angstroms
        sub_region = SpectralRegion(val - window, val + window)
        sub_spectrum = extract_region(spectrum, sub_region)
        
        #this calls a function which fits the sub region with a gaussian and we pass in the 
        #emission center from specutils as an initial guess
        par = fitting_lines(sub_spectrum)
        
        #getting the flux of the line using scipy curve fit parameters
        flux_line = np.sqrt(2*np.pi)*abs(par[0])*abs(par[-1])
        
        OIII_line_flux[i] = flux_line
    
    
        
    #plt.show() 
    
    #print('OIII Line -----------------     Flux')
    
    #for i in range(3):
     #   print('%12.2f ----------------- %9.2f' %(OIII_lines[i].value, OIII_line_flux[i]))
            
    #making a for loop to test spatial component of our spectrums    
    for i in line_center_index:
        
        #the window of pixels to look around
        window = 15
        
        #making an x array that ranges in index from i-window to i+window
        x = range(i-window, i+window)
        
        #maknig the star and end of the splicing of our array
        start = i-window
        end = i+window
        
        #checking to make sure that our starting index is still within the array and doesn't loop back
        if i - window < 0:
            continue
        
        #makes a reduced spectrum from i +/- window
        spe = flux[filt][start:end]
        
        #fits gaussian to it
        param, covar = curve_fit(f, x, spe, p0=[np.amax(spe), i, 7])
        
        #makes another x array this time for fitting purposes
        #x1 = np.linspace(i-window, i+window, 1000)
        
        #the gaussian curve
        #y_curve = f(x1, *param)
        
        #gettign the pixel sigma from seeing/conversion seeing is in arcsec and conversion is in arcsec/pixel
        sig_seeing = seeing/conversion
        
        #makes a gaussian with same amplitude and center but with the sigma from the sig_seeing
        #y_seeing = f(x1, param[0], param[1], sig_seeing)
        
        #plots it for each line
        
        #plt.figure(figsize = (10,10))
        #plt.plot(x, spe)
        #plt.plot(x1, y_curve)
        #plt.plot(x1, y_seeing, 'r--', label = 'Seeing')
        #plt.legend(loc='best')
        #plt.show()
     '''   
    
    ###################################
    #Making the Astropy Table Below
    ###################################
    
    
    #making list to hold values that we are interested in
    
    #this holds the rest frame wavelength
    rest_line = []
    
    #the name of that rest frame line
    name_line = []
    
    #the line center calculated form curve_fit
    line_center = []
    
    #flux found using curve_fit parameters
    flux = []
    
    #EW found using specutils flux and continuum function
    EW = []
    
    #the noise of the line
    noise_ = []
    
    #the standard deviation of the noise by the emission line
    noise_std = []

    #debugging code 
    #plt.figure(figsize = (14, 6))
    #plt.title(filename[7:-5])
    #plt.xlabel(r'Rest Frame Wavelength [$\AA$]')
    #plt.ylabel(r'Flux [$10^{-17} \frac{erg}{s\hspace{.5} cm^2 \hspace{.5} \AA} $]')
    #plt.plot(spectrum.spectral_axis.value/(1+z), spectrum.flux, 'k-', alpha = .6)
    #print()
    #print('    Line Name ----- Rest Line -------  WvlnConv -------     Flux -------     EW  ')
    
    #for loop that goes through the lines calculated by curve_fit
    for i, val in enumerate(emission_line_fit):
        
        #print(val/(1+z))
        
        #if value is negative it skips it
        if val < 0:
            continue
        
        #this holds the index in the line catalog information where the closest match is between our calculated line center to rest frame catalog
        index = abs(line-(val/(1+z))).argmin()
        
        #gets the rest frame wavelength from the catalog
        rest_l = line[index]
        #gets the name of the line
        rest_name = line_name[index]
        
        #this gets the value of the difference at the index we got using curve fit line center
        diff = abs(line-(val/(1+z)))[index]
        
        #plt.axvline(val/(1+z), linestyle = '--', color = 'red', linewidth = .5)
        #plt.axvline(rest_l, linestyle = '-', color = 'blue', linewidth = .5)
        
        #condition needed for line to be accurately determined if diff > 3 then its not our line
        if diff > 3:
            continue

        #this finds the index of where the rest line is closest to in the emission lines that I got
        #this index can be used for equivalent width and flux matching 
        ind = abs(rest_l - emission_line_fit/(1+z)).argmin()
        
        #makes sure the same line is not repeated
        if rest_l in rest_line:
            continue

        else:

            #print('%13s ----- %9.2f ------- %9.2f ------- %9.2f ------- %9.2f' %(rest_name, rest_l, val/(1+z), line_f[ind], manual_ew[ind]))

            #ind_catalog.append(index)
            #ind_calculation.append(ind)
            
            #plt.axvline(rest_l, linestyle = '-', color = 'red', linewidth = .5)
            #plt.axvline(val/(1+z), linestyle = '-', color = 'blue', linewidth = .5)
            #plt.axvline(emission_lines[i]/(1+z), linestyle = '-', color = 'black', linewidth = .5)
            #plt.axvline(emission_lines[i]/(1+z[0]), linestyle = '--', color = 'black', linewidth = .5)
            #plt.text(rest_l , np.amax(spectrum.flux.value)/3, rest_name, rotation = 270, fontsize = 'x-small')


            #appends everything to the lists above
            rest_line.append(rest_l)
            name_line.append(rest_name)
            line_center.append(round(val/(1+z), 4))
            flux.append(round(line_f[ind], 4))
            EW.append(round(manual_ew[ind], 4))
            noise_.append(round(noise[index], 4))
            noise_std.append(round(line_noise[index], 4))
            #print('%13s ----- %9.2f ------- %9.2f ------- %9.2f ------- %9.2f' %(rest_name, rest_l, val/(1+z[0]), line_f[ind], manual_ew[ind]))
            #print()
            
    #plt.savefig(filename[7:-5]+'_z_'+str(z[0])+'.pdf')
    #plt.show()
    
    #making an Astropy table for the file passed in below
    t = Table()
    t['line_name'] = np.array(name_line)
    t['rest_frame_wavelength'] = np.array(rest_line)
    t['calculated_center'] = np.array(line_center)
    t['line_flux'] = np.array(flux)
    t['line_EW'] = np.array(EW)
    t['line_noise'] = np.array(noise_)
    t['line_noise_std_dev'] = np.array(noise_std)
    
    #print(t)
    #print()
    
    return t, He_line_names, He_lines.value/(1+z), He_line_flux
    
    '''
    #print('Emission Line ------ Emission Fit------ ew_spec ------- ew_manual ------- flux_spec ------ manual_flux ------- continuum val')
    #for i in range(len(e_width)):
    #    print('%13.2f ------ %12.2f ------ %7.2f ------- %9.2f ------- %9.2f ------ %11.2f ------- %13.2f' 
    #          %(emission_lines[i]/(1+z), emission_line_fit[i]/(1+z), e_width[i].value, 
    #            manual_ew[i], lines_flux[i].value, line_f[i], continuum_val[i].value))
    
    
    #print() 
    #print()
    
    rest_line = []
    l = []
    
    #print(' Rest Line ------------- LCR Specutils ----------- LCR Curve Fit')
    for i in emission_line_fit:
        
        index = abs(line-(i/(1+z))).argmin()
        
        rest_l = line[index]
        
        ind = abs(rest_l - (emission_line_fit/(1+z))).argmin()

        test_z = (rest_l - emission_line_fit[ind])/rest_l

        
        if rest_l in rest_line:
            continue
        
        else:
        
            #print('%9.2f ------------- %13.2f ----------- %13.2f' %(rest_l, emission_lines[ind]/(1+z), emission_line_fit[ind]/(1+z)))
            rest_line.append(rest_l)
        #plt.axvline(rest_l * (1+z), linestyle = '--', )
    
    #plt.savefig('Testing_redshift_'+str(z)+'.pdf')
    #plt.show()    
    #print()
    '''
    
    #return np.array(emission_lines), np.array(line_f), np.array(manual_ew)
    
'''
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

fig = plt.figure(figsize = (16, 10))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)



file = 'median_J231903+010853_cem.fits'
sp1 = file.split('_')[1]
sp2 = sp1.split('+')[0]

match = ID_1 == sp2
#print(match)
z1 = z_redshift[match]

seeing_t = seeing[match]

conversion = .188
#print(z1)

spec, wvln = spectrum(file)

spectra, cont_func, filt= fitting_continuum(wvln, spec, z1, line_wavelength, file)
ind = np.where(wvln < 4500)
ind2 = np.where(wvln > 6350)
ind_tot = np.concatenate((ind, ind2), axis=None)
filt[ind_tot] = False
percentile = 99
#print(i)
#print('-----------------------')
#print()
table, HeI_name, He_lines, He_line_flux = analysis(spectra, wvln, filt, line_wavelength, 
                                                       z1, cont_func, percentile, line_name, 
                                                       file, seeing_t, conversion)

'''
#fig = plt.figure(figsize = (16, 10))
#ax1 = fig.add_subplot(311)
#ax2 = fig.add_subplot(312)
#ax3 = fig.add_subplot(313)

#this list will hold the various tables for each file
table_list = [] 

#this will hold the line fluxes for HeI
HeI_line_fluxes = []
#this will hold the names ofr the HeI lines
HeI_name = 0

#this should hold the line center of the HeI lines
He_lines = 0

for i in files:
    
    if 'cem.fits' in i:
        
        #print(i)
        #print('-------------')
        
        #splits up the filename by the underscore and getting the ID number
        sp1 = i.split('_')[1]
        
        #using that ID to split it even further across the cross and gettin gthe first ID Number
        sp2 = sp1.split('+')[0]
        
        #we use this to match it with the ID we got from the catalog
        match = ID_1 == sp2
        
        #print(match)
        #gets the redhsift from the catalog
        z1 = z_redshift[match][0]
        
        #gets the seeing from the catalog
        seeing_t = seeing[match][0]
        
        #conversion is dependent on the file passed in this is the conversion for the cem files
        #.188 arcsec/pixel
        
        conversion = .188
        #print(z1)
        
        #gets the 1D spectrum and wavelength
        spec, wvln = spectrum(i)
        
        #fits the continuum
        spectra, cont_func, filt, filt_z, std_dev, noise = fitting_continuum(wvln, spec, z1, line_wavelength, i)

        #assigns the percentile to 97
        percentile = 97
        
        if 'J231903+010853' in i:
            percentile = 99
        
        #print(i)
        #print('-----------------------')
        #print()
        
        #gets the spectroscopic redshift
        spectroscopic_redshift = spec_redshift(spectra, wvln, filt_z, line_wavelength, z1, 99, i, line_name)
        
        #gets the table along with the He_line calculations
        table, HeI_name, He_lines, He_line_flux = analysis(spectra, wvln, filt_z, line_wavelength, 
                                                           spectroscopic_redshift, cont_func, percentile, line_name, 
                                                           i, seeing_t, conversion, std_dev, noise)
        
        #appends the table to the table list above
        table_list.append(table)
        #appends the HeI lines to the list above
        HeI_line_fluxes.append(He_line_flux)
        
        
        #ax1.plot(wavelength, spec, label = i)
        
    if 'mods1b' in i:
        
        #print(i)
        #print('-------------')
        
        #splits up the filename by the underscore and getting the ID number
        sp1 = i.split('_')[1]
        
        #using that ID to split it even further across the cross and getting the first ID Number
        sp2 = sp1.split('+')[0]
        
        #we use this to match it with the ID we got from the catalog
        match = ID_1 == sp2
        
        #print(match)
        
        #getting redshift from the catalog
        z1 = z_redshift[match][0]
        
        #gets the seeing form the catalog
        seeing_t = seeing[match][0]
        
        #assigning the conversion number for this file .12 arcsec/pixel
        conversion = .12
        
        #gets the 1D spectrum and wavelength
        spec, wvln = spectrum(i)
        
        #fits the continuum
        spectra, cont_func, filt, filt_z, std_dev, noise = fitting_continuum(wvln, spec, z1, line_wavelength, i)
       
        #gets the spectroscoic redshift
        spectroscopic_redshift = spec_redshift(spectra, wvln, filt_z, line_wavelength, z1, 99, i, line_name)
        
        #assigning percentile
        percentile = 97
        
        #print(i)
        #print('-----------------------')
        #print()
        
        #getting the table and He_1 lines
        table, He_line_names, He_lines, He_line_flux = analysis(spectra, wvln, filt_z, line_wavelength, 
                                                                spectroscopic_redshift, cont_func, percentile, 
                                                                line_name, i, seeing_t, conversion, std_dev, noise)
        
        #appending table and HeI lines to lists above
        table_list.append(table)
        HeI_line_fluxes.append(He_line_flux)
        #ax2.plot(wavelength, spec, label = i)

    if 'mods1r' in i:
        
        #print(i)
        #print('-------------')
        
        #splits up the filename by the underscore and getting the ID number
        sp1 = i.split('_')[1]
        
        #using that ID to split it even further across the cross and getting the first ID Number
        sp2 = sp1.split('+')[0]
        
        #we use this to match it with the ID we got from the catalog
        match = ID_1 == sp2
        
        #gets the redshift from catalog
        z1 = z_redshift[match][0]
        
        #gets seeing from catalog
        seeing_t = seeing[match][0]
        
        #assigning converison which for these files are .123 arcsec/pixel
        conversion = .123
        
        #getting the 1D spectrum and wavelength
        spec, wvln = spectrum(i)

        #fitting the continuum
        spectra, cont_func, filt, filt_z, std_dev, noise = fitting_continuum(wvln, spec, z1, line_wavelength, i)
        
        #getting the spectroscopic redshift
        spectroscopic_redshift = spec_redshift(spectra, wvln, filt_z, line_wavelength, z1, 99, i, line_name)
        
        #assigning percentile
        percentile = 97
        
        #print(i)
        #print('-----------------------')
        #print()
        
        #getting the table and HeI lines
        table, He_line_names, He_lines, He_line_flux = analysis(spectra, wvln, filt_z, line_wavelength, spectroscopic_redshift, 
                                                                cont_func, percentile, line_name, i, seeing_t, conversion,
                                                                std_dev, noise)
        
        #appending tabe and HeI lines to lists above
        table_list.append(table)
        HeI_line_fluxes.append(He_line_flux)
        
        #ax3.plot(wavelength, spec, label = i)

#ax1.set_ylim(-2, 75)
#ax2.set_ylim(-2, 75)
#ax3.set_ylim(-2, 75)
#ax1.legend(loc='best')
#ax2.legend(loc='best')
#ax3.legend(loc='best')
#plt.show()          

#list that will hold the final table for files looking at the same galaxy but 2 different 
final_table = []

#looping over file_num which holds unique values of the file ID meaning it got rid of repeats
for i in file_num:
    
    #making a filter called same object
    same_obj = np.zeros(len(files), dtype = bool)
    
    #enumerating over all the files I have
    for j, val in enumerate(files):
        
        #checks to see if the ID is in the filename if it is then changes the index valeu of filter to True
        if i in val:
            same_obj[j] = True
    
    #this filters the files that I have to those of the same obj
    obj_files = np.array(files)[same_obj]        
    
    #this holds the number of files in obj_files
    num_of_files = len(obj_files)
    
    #reduced_table holds the tables of onl those of the same object
    reduced_table = np.array(table_list)[same_obj]
    
    #same with the HeI lines only those of the same object
    reduced_He_lines = np.array(HeI_line_fluxes)[same_obj]
    
    #if only one file then we append the table to the final table 
    if num_of_files == 1:
    
        final_table.append(reduced_table[0])
        
    #if there is two then we got some work to do before appending
    if num_of_files == 2:
        
        #first we add up the HeI lines because one file will have [val, 0, 0] and the other one will have [0, val, val]
        He1 = np.round(reduced_He_lines[0] + reduced_He_lines[1], 2)
        
        #making a helium table 
        table_He = Table()
        
        #I add to it the line name, flux and rest frame wavlength
        table_He['line_name'] = HeI_name
        table_He['line_flux'] = He1
        table_He['rest_frame_wavelength'] = He_lines
        
        
        #get the independent tables from reduced table
        table1 = reduced_table[0]
        table2 = reduced_table[1]
        
        #then stack all the tables together
        master_table = vstack([table1, table2, table_He])
        
        
        #then I append this to the final_table
        final_table.append(master_table)

def line_ratio_analysis(master_table):
    '''
    This function will go through the data and calculate all the line ratios that we are interested. 
    These include OIII/Hbeta, NII/Halpha, SII 6716/6730
    
    Parameters
    -------------------
    
    master_table: Table values that include all the emission lines from a file
                  Needs to have the line name une 'line_name' and flux under [line_flux]
                
    Outputs
    -------------------
    
    
    The line ratios of the lines we care about
    '''
    
    #line flux is index three in the columns and we want the rows corresponding to
    #OIII5007, Hbeta, Halpha, [NII]6583
    
    #initializing my variables to zero and will assign them after criteria is met
    #namely that the lines are there
    OIII5007 = -1
    Hbeta = -1
    NII6583 = -1
    Halpha = -1
    
    SII6716 = -1
    SII6730 = -1
    
    HeI6678 = -1
    HeI7065 = -1
    
    OII3725 = -1
    OII3727 = -1
    
    OIII4363 = -1
    OIII4959 = -1
    
    for i, val in enumerate(master_table['line_name']):

        #looking for the row in the master table where this line is at
        if val == '[OIII]5007':
            #extracting the flux
            #flux is at index three
            OIII5007 = master_table[i][3]
            
        if val == '[OIII]4959':
            #extracting the flux
            #flux is at index three
            OIII4959 = master_table[i][3]
        
        if val == '[OIII]4363':
            #extracting the flux
            #flux is at index three
            OIII4363 = master_table[i][3]
            
        #looking for the row in the master table where this line is at
        if val == 'Hbeta':
            #extracting the flux
            #flux is at index three
            Hbeta = master_table[i][3]
        
        if val == 'Halpha':
            #extracting the flux
            #flux is at index three
            Halpha = master_table[i][3]
        
        if val == '[NII]6583':
            #extracting the flux
            #flux is at index three
            NII6583 = master_table[i][3]
        
        if val == 'HeI6678':
            #extracting the flux
            #flux is at index three
            HeI6678 = master_table[i][3]
        
        if val == 'HeI7065':
            #extracting the flux
            #flux is at index three
            HeI7065 = master_table[i][3]
            
        if val == '[SII]6716':
            #extracting the flux
            #flux is at index three
            SII6716 = master_table[i][3]
        
        if val == '[SII]6730':
            #extracting the flux
            #flux is at index three
            SII6730 = master_table[i][3]
            
        if val == '[OII]3725':
            
            OII3725 = master_table[i][3]
        
        if val == '[OII]3727':
            
            OII3727 = master_table[i][3]    
            
    
    ################################
    #Calculating Helium Line Ratios
    ################################
    
    #got this from the isotov paper im sure we will have to change this value up as it seems to be
    #in the Temperature range 15000-20000K
    HeI3889 = .107 * Hbeta
    
    He_7065_6678 =-1
    He_3889_6678 = -1
    
    #calculating line ratio for helium
    if HeI6678 < 0:
        He_7065_6678 = -999
        He_3889_6678 = -999
        
    else:   
        He_7065_6678 = HeI7065/HeI6678
        He_3889_6678 = HeI3889/HeI6678
        
    ################################
    #Silicon Line Ratio
    ################################
    
    if SII6730 < 0:
        SII_6716_6730 = -999
        
    else:   
        SII_6716_6730  = SII6716/SII6730 
        
    ################################
    #Oxygen Line Ratio Part 1: Finding How Ionizing it is
    ################################  
    
    OIII5007_OII3727 = -999
    OIII5007_OII3725 = -999
    
    if OII3727 > 0:
        OIII5007_OII3727 = OIII5007/OII3727
    
    if OII3725 > 0:
        OIII5007_OII3725 = OIII5007/OII3725    
    
    ################################
    #Oxygen Line Ratio Part 2: Getting Temperature
    ################################
    
    OIII_ratio = -1
    T = -999
    
    if OIII5007 < 0 or OIII4959 < 0 or OIII4363 < 0:
        OIII_ratio = -999
    
    else:
        #gets the line ratio of all the OIII lines
        OIII_ratio = (OIII5007 + OIII4959)/OIII4363
        
        #assuming a number density of 100 cm^(-3)
        ne = 100
        
        #makes an array in logspace for the temperature
        Temp = np.logspace(3, 5, 10000)    
        
        #makes a line corresponding to the ratio caluclated above
        ratio_line = OIII_ratio * np.ones(len(Temp)) 
        
        #calculated the numerator and denominator of the equation that will give us the temperature of the gas
        num = 7.9*np.exp(3.29e4/Temp)
        denom = 1+((4.5e-4 * ne))/np.sqrt(Temp)
        
        #holds the values for the temperature ratio
        ratio_T = num/denom
        
        #gets the index where both lines; ratio_line and ratio_T cross
        idx = np.argwhere(np.diff(np.sign(ratio_line - ratio_T))).flatten()
        
        #gets the temperature in this variable
        T = Temp[idx][0]
        
    #########################################
    #BPT Line Ratios: OIII/Hbeta, NII/Halpha
    #########################################
    
    #calculating line ratios
    OIII_Hbeta = 0
    NII_Halpha = 0
    
    #checking the different criterias and if not there then assign ratio -999
    if OIII5007 > 0 and Hbeta > 0 and NII6583 > 0 and Halpha > 0:
        #print('All Good')
        OIII_Hbeta = OIII5007/Hbeta
        NII_Halpha = NII6583/Halpha
        
    
    if (OIII5007 > 0 and Hbeta > 0) and (NII6583 < 0 or Halpha < 0):
        #print('OIII_Hbeta Good')
        OIII_Hbeta = OIII5007/Hbeta
        NII_Halpha = -999
      
    
    if (OIII5007 < 0 or Hbeta < 0 )and (NII6583 > 0 and Halpha > 0):
        #print('Halpha_NII Good')
        OIII_Hbeta = -999
        NII_Halpha = NII6583/Halpha
        
    return [round(OIII_Hbeta, 4), round(NII_Halpha, 4), round(He_7065_6678, 4), round(He_3889_6678, 4), round(SII_6716_6730, 4), round(OIII5007_OII3725, 4), round(OIII5007_OII3727, 4), round(T, 2)]

#Lists that will hold the Emission Line Ratios    
ratio_OIII_hbeta = []
ratio_NII_halpha = []
He_7065_6678 = []
He_3889_6678 = []
SII_6716_6730 = []
OIII_5007_3725 = []
OIII_5007_3727 = []
Temperature = []

for i in range(len(file_num)):
    
    print('File Number: ' + file_num[i])
    print(final_table[i])
    #final_table[i].write(file_num[i]+'_lineinfo.fits', format='fits')
    
    #here HeI_1 is the ratio HeI 7065/6678 HeI_2 is the ratio HeI 3889/6678
    #SII is the SII ratio between 6716 and 6730
    ratio = line_ratio_analysis(final_table[i])
    
    ratio_OIII_hbeta.append(ratio[0])
    ratio_NII_halpha.append(ratio[1])
    He_7065_6678.append(ratio[2])
    He_3889_6678.append(ratio[3])
    SII_6716_6730.append(ratio[4])
    OIII_5007_3725.append(ratio[5])
    OIII_5007_3727.append(ratio[6])
    Temperature.append(ratio[7])
    print()

    
    
#creating table that will hold the emission lines for each galaxy object
#Object name can be found in the ObjectID column

ratio_table = Table()

ratio_table['ObjectID']  = np.array(file_num)
ratio_table['OIII5007/Hbeta'] = np.array(ratio_OIII_hbeta)
ratio_table['NII6583/Halpha'] = np.array(ratio_NII_halpha)
ratio_table['HeI 7065/6678'] = np.array(He_7065_6678)
ratio_table['HeI 3889/6678'] = np.array(He_3889_6678)
ratio_table['SII 6716/6730'] = np.array(SII_6716_6730)
ratio_table['OIII5007/OII3725'] = np.array(OIII_5007_3725)
ratio_table['OIII5007/OII3727'] = np.array(OIII_5007_3727)
ratio_table['Temperature'] = np.array(Temperature)

print(ratio_table)

#plotting_BPT(ratio_table)

end = time.time()

print('Time of Program = ' + str(round((end-start)/60, 2)) + ' minutes!')


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
