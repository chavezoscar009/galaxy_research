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
from astropy.table import Table, Column

#grabbing all of micaela's fits files from the current directory
files = [x for x in glob.glob('*.fits') if 'SDSS' not in x]

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


def fitting_gaussian(data, wavelength, filename):
    
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
    
    #this will hold the sigma valueas and wavelength value of the median gaussian fit
    sig = []
    wvln = []
    mean = []
    
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
        
        #popt is the best fit parameter values
        #covar is error on the parameters
        popt, covar = curve_fit(f, x, median_data, 
                                p0 = [np.amax(median_data), len(median_data)//2, 40], 
                                bounds=[(0, 0, 1), (np.amax(median_data), len(data[:, 0]), 100)])
        
        #std_dev holds the standard deviation for the gaussian fit
        std_dev = popt[-1]
        
        #appending the sigma and wavelength to the list
        sig.append(std_dev)
        wvln.append(med_wvln)
        mean.append(popt[1])
        
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
    
    #this holds the filename split along the underscore
    t = filename.split('_')
    
    if 'cem.fits' in filename:
        
        sig_filt = medfilt(sig, kernel_size = 7)
        
        sigma = np.amax(sig_filt)
        
        y = norm.pdf(x, len(median_data)//2, sigma)
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
    
    else:
        
        sig_filt = medfilt(sig, kernel_size = 7)
        
        sigma = np.mean(sig_filt)
        
        y = norm.pdf(x, len(median_data)//2, sigma)
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

def fitting_lines(spectrum, mu_guess, sig_guess=4):
    
    def f(x, A, mu, sig):
        return A * np.exp(-(x-mu)**2/(2*sig**2))
    
    param, covar = curve_fit(f, spectrum.spectral_axis.value, spectrum.flux.value, p0 = [np.amax(spectrum.flux.value), mu_guess, sig_guess])
    
    #amp = param[0]
    #mu = param[1]
    #sig = param[-1]
    
    return param

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
    gauss_mult = fitting_gaussian(boxed_data, wvln_spec, file)
    
    gauss_filtered = (boxed_data.T * gauss_mult).T
    
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
    
    ax1.set_title(file + ' Continuum Subtraction')
    ax1.set_ylim(-5, 50)
    #ax2.set_ylim(-5, 75)
    
    ax1.plot(wavelength_spec, spectra)
    ax1.plot(wavelength_spec, y, label = 'y = 0')
    
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
    
    continuum_subtracted_spec = medfilt(spectra-y_continuum.value, kernel_size = 5)

    return continuum_subtracted_spec, continuum_fit, filt_noise

def analysis(flux, wavelength, filt, line, z, continuum_func, percentile, line_name, filename, seeing, conversion):
    '''
    
    This function will take in a 1D extracted spectrum and the emission QTables gathered from the 
    find_lines_derivatives or find_lines_derivatives.
    
    Parameters
    -----------------
    flux: This is the flux of the spectrum we will do our analysis for needs to be centered at 0
    wavelength: The wavlength corresponding to the flux
    emission_lines: a list of emission lines to look for
    z = redshift of the galaxy
    continuum_func: this is the funciton that mmodels the continum and we need that for manual EW calculations
    
    
    Output
    ------------------
    Analysis of the emissions lines passed in.
    
    emission_lines: the line center in angstrom
    lines_flux: the flux of the lines given in ergs/(s cm^2)
    e_width: the equivalent width of the line
    
    '''
    
    #making a gaussian function so that i can use curve_fit later to fit emission lines
    def f(x, A, mu, sig):
        return A * np.exp(-(x-mu)**2/(2*sig**2))
    
    spectrum = 0
    threshold = 0
    
    if 'mods1r' in filename:
        if 'J082540+184617' in filename:
            offset = 15
            spectrum = Spectrum1D(spectral_axis=(wavelength[filt]-offset)*u.angstrom, 
                              flux =flux[filt]*u.erg/u.s/u.cm/u.cm/u.angstrom)
            #sorting the data
            data = np.sort(flux[filt])

            #making the threshold for the flux to be above a percentile value
            threshold = np.percentile(data, q=percentile, interpolation='midpoint')/1.8     
        else:
            offset = 18.5
            spectrum = Spectrum1D(spectral_axis=(wavelength[filt]-offset)*u.angstrom, 
                                  flux =flux[filt]*u.erg/u.s/u.cm/u.cm/u.angstrom)
            #sorting the data
            data = np.sort(flux[filt])

            #making the threshold for the flux to be above a percentile value
            threshold = np.percentile(data, q=percentile, interpolation='midpoint')/1.8
    elif 'J073149+404513' in filename and 'mods1b' in filename:
        offset = 8
        spectrum = Spectrum1D(spectral_axis=(wavelength[filt]-offset)*u.angstrom, 
                          flux =flux[filt]*u.erg/u.s/u.cm/u.cm/u.angstrom)
        #sorting the data
        data = np.sort(flux[filt])

        #making the threshold for the flux to be above a percentile value
        threshold = np.percentile(data, q=percentile, interpolation='midpoint')
    
    elif 'J021306+005612' in filename and 'mods1b' in filename:
        offset = 2
        spectrum = Spectrum1D(spectral_axis=(wavelength[filt]+offset)*u.angstrom, 
                          flux =flux[filt]*u.erg/u.s/u.cm/u.cm/u.angstrom)
        #sorting the data
        data = np.sort(flux[filt])

        #making the threshold for the flux to be above a percentile value
        threshold = np.percentile(data, q=percentile, interpolation='midpoint')
    
    elif 'J082540+184617' in filename and 'mods1b' in filename:
        offset = 1
        spectrum = Spectrum1D(spectral_axis=(wavelength[filt]+offset)*u.angstrom, 
                          flux =flux[filt]*u.erg/u.s/u.cm/u.cm/u.angstrom)
        #sorting the data
        data = np.sort(flux[filt])

        #making the threshold for the flux to be above a percentile value
        threshold = np.percentile(data, q=percentile, interpolation='midpoint')    
        
    elif 'J030903+003846' in filename and 'cem' in filename:
        offset = 3
        spectrum = Spectrum1D(spectral_axis=(wavelength[filt]+offset)*u.angstrom, 
                          flux =flux[filt]*u.erg/u.s/u.cm/u.cm/u.angstrom)
        #sorting the data
        data = np.sort(flux[filt])

        #making the threshold for the flux to be above a percentile value
        threshold = np.percentile(data, q=percentile, interpolation='midpoint')
    
    elif 'J231903+010853' in filename and 'cem' in filename:
        offset = 3
        spectrum = Spectrum1D(spectral_axis=(wavelength[filt]+offset)*u.angstrom, 
                          flux =flux[filt]*u.erg/u.s/u.cm/u.cm/u.angstrom)
        #sorting the data
        data = np.sort(flux[filt])

        #making the threshold for the flux to be above a percentile value
        threshold = np.percentile(data, q=percentile, interpolation='midpoint')
    
    elif 'J014707+135629' in filename and 'cem' in filename:
        offset = 3
        spectrum = Spectrum1D(spectral_axis=(wavelength[filt]+offset)*u.angstrom, 
                          flux =flux[filt]*u.erg/u.s/u.cm/u.cm/u.angstrom)
        #sorting the data
        data = np.sort(flux[filt])

        #making the threshold for the flux to be above a percentile value
        threshold = np.percentile(data, q=percentile, interpolation='midpoint')
        
    else:
        #making our Spectrum1D object getting rid of the noise in the outer portion of the spectrum
        spectrum = Spectrum1D(spectral_axis=wavelength[filt]*u.angstrom, 
                              flux =flux[filt]*u.erg/u.s/u.cm/u.cm/u.angstrom)
        #sorting the data
        data = np.sort(flux[filt])
        
        #making the threshold for the flux to be above a percentile value
        threshold = np.percentile(data, q=percentile, interpolation='midpoint')
    
    #making a way to automatically check for emisison lines. For this I sort the data and then pick a percentile 
    #from which anything above that will be considered an emission line.
    
    #NOTE: this percentile changes for different files
    
   
    '''
    threshold_limit_test = np.linspace(1, 1.5, 6)
    plt.figure(figsize = (14, 6))
    plt.plot(spectrum.spectral_axis, spectrum.flux, 'b-')
    for i, val in enumerate(threshold_limit_test):
        
        #finding lines using specutils line_derivative function
        lines = find_lines_derivative(spectrum, flux_threshold=val)

        #getting only the emission lines
        emission = lines[lines['line_type'] == 'emission']

        y = (threshold/val)*np.ones(len(spectrum.spectral_axis))
        
        plt.plot(spectrum.spectral_axis, y , label = str(threshold/val))
        
        plt.legend(loc ='best', ncol=2)
        #for i in emission['line_center'].value:
            #plt.axvline(i, linestyle= '--', color = 'red')
            
    y_2 = (threshold/2)*np.ones(len(spectrum.spectral_axis))
    plt.plot(spectrum.spectral_axis, y_2 , label = str(threshold/2))
    plt.show()
    '''    
    
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
    
    #for loop that goes through each of the emission lines and does flux and equivalent width analysis
    #plt.figure(figsize = (12, 6))
    
    line_center_index = []
    
    for i in emission['line_center']:
       
        #making a window to look around the line center so that I can do some analysis using specutils
        #as well as my own homemade functions
        window = 15*u.angstrom
        
        #looking at the sub_region around where the line center is located at and +/- 15 Angstroms
        sub_region = SpectralRegion(i - window, i + window)
        sub_spectrum = extract_region(spectrum, sub_region)
        
        #calculating the emission line of the sub_region
        lines_flux.append(line_flux(sub_spectrum))
        
        #appending the emission line center
        emission_lines.append(i.value)
        
        #this calls a function which fits the sub region with a gaussian and we pass in the 
        #emission center from specutils as an initial guess
        par = fitting_lines(sub_spectrum, i.value)
        
        ###############
        #using specutils tools to fit lines with gaussian
        ###############
        
        #getting an initial guess on the Gaussian parameters
        param = estimate_line_parameters(sub_spectrum, models.Gaussian1D())
        #making an intial guess of the gaussian
        g_init = models.Gaussian1D(amplitude=param.amplitude, mean=param.mean, stddev=param.stddev)
        
        #fitting the emession line to the gaussian using values from above
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
        flux_line = np.sqrt(2*np.pi)*par[0]*par[-1]
        
        #getting the equivalent width from flux calculation above
        manual_ew.append(flux_line/continuum_func(par[1]*u.angstrom).value)
        
        #getting the center of the emission peak from curve_fit and appending it
        emission_line_fit.append(par[1])
        
        line_center_index.append(abs(spectrum.spectral_axis.value - par[1]).argmin()) 
        
        #appending the manual flux calculations
        line_f.append(flux_line)
        
        #appending the continuum value
        continuum_val.append(continuum_func(par[1]*u.angstrom))
        
        #plotting code where i was testing which fit was better
        #plt.plot(sub_spectrum.spectral_axis, sub_spectrum.flux, 'k-')
        #plt.plot(x.value, y_fit, 'r--',alpha = .6, label = 'Specutils Fitting')
        #plt.plot(x.value, y_curve, 'y--',alpha = .7, label = 'Curve Fit')
        #plt.axvline(param.mean.value, linestyle = '--', color='red')
        #plt.axvline(i.value, linestyle = '--', color='red')
        #plt.legend(loc='best')
    #plt.show()    
        
    #making a for loop to test spatial compnent of our spectrums    
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
        x1 = np.linspace(i-window, i+window, 1000)
        
        #the gaussian curve
        y_curve = f(x1, *param)
        
        #gettign the pixel sigma from seeing/conversion seeing is in arcsec and conversion is in arcsec/pixel
        sig_seeing = seeing/conversion
        
        #makes a gaussian with same amplitude and center but with the sigma from the sig_seeing
        y_seeing = f(x1, param[0], param[1], sig_seeing)
        
        #plots it for each line
        
        #plt.figure(figsize = (10,10))
        #plt.plot(x, spe)
        #plt.plot(x1, y_curve)
        #plt.plot(x1, y_seeing, 'r--', label = 'Seeing')
        #plt.legend(loc='best')
        #plt.show()
        
    ###################################
    #Emission Line Stuff
    ###################################
          
    '''
    total_spectrum = Spectrum1D(spectral_axis=wavelength*u.angstrom, flux =flux*u.erg/u.s/u.cm/u.cm/u.angstrom)
    
    interest = np.zeros(len(line), dtype = bool)
    
    for i, val in enumerate(line_name):
        if val == 'Hbeta' or val == 'Halpha' or val =='[OIII]5007' or val == '[NII]6583':
            interest[i] = True
            
    line_of_interest = np.array(line)[interest]
    line_name_interest = np.array(line_name)[interest]
    
    print(line_name_interest)
    print(line_of_interest)
    
    #plt.figure(figsize = (14, 6))
    
    for l, val in enumerate(line_of_interest):
        
        if val*(1+z) < wavelength[0] or val*(1+z) > wavelength[-1]:
            continue
        
        else:
            
            #making a window to look around the line center so that I can do some analysis using specutils
            #as well as my own homemade functions
            window = 25*u.angstrom

            #looking at the sub_region around where the line center is located at and +/- 15 Angstroms
            sub_region = SpectralRegion(val*(1+z)*u.angstrom - window, val*(1+z)*u.angstrom + window)
            sub_spectrum = extract_region(total_spectrum, sub_region)
            
            plt.figure(figsize = (10, 10))
            plt.plot(sub_spectrum.spectral_axis, sub_spectrum.flux, color='red', alpha = .6, label = 'flux')
            plt.show()
            
            
            par = fitting_lines(sub_spectrum, val*(1+z[0]))
            
            flux_line = np.sqrt(2*np.pi)*par[0]*par[-1]
            manual_ew.append(flux_line/continuum_func(par[1]*u.angstrom).value)
            
            x = np.linspace(sub_spectrum.spectral_axis[0].value, sub_spectrum.spectral_axis[-1].value, 1000)*u.angstrom
            y_curve = f(x.value, *par)
            
            plt.figure(figsize = (10, 10))
            plt.plot(sub_spectrum.spectral_axis, sub_spectrum.flux, color='red', alpha = .6, label = line_name_interest[l])
            plt.plot(x, y_curve, color='black')
            plt.legend(loc='best')
            plt.show()
            
    
    #plt.legend(loc = 'best')
    plt.show()
    
    #plt.figure(figsize = (14, 6))
    #plt.plot(spectrum.spectral_axis, spectrum.flux, alpha = .6)
    '''
    
    #I need a way to test which lines I found and compare that with lines of interest but I also
    #want to keep the information that I have, I also need a way to get rid of duplicates need a criteria
    #to discern an actual line vs not a line:
    
    
    rest_line = []
    name_line = []
    line_center = []
    flux = []
    EW = []
    
    ind_catalog = []
    ind_calculation = []
    
    line_catalog_filt = np.ones(len(line), dtype = bool)
    calculation_filt = np.ones(len(line_f), dtype = bool)

    
    #plt.figure(figsize = (14, 6))
    #plt.title(filename[:-5] + ' at z = ' + str(z[0]))
    #plt.xlabel(r'Rest Frame Wavelength [$\AA$]')
    #plt.ylabel(r'Flux [$10^{-17} \frac{erg}{s\hspace{.5} cm^2 \hspace{.5} \AA} $]')
    #plt.plot(spectrum.spectral_axis.value/(1+z[0]), spectrum.flux, 'k-', alpha = .6)
    #print()
    #print('    Line Name ----- Rest Line -------  WvlnConv -------     Flux -------     EW  ')
    
    for i, val in enumerate(emission_line_fit):
        
        #this will be the index in the line catalog information where the closest match is
        index = abs(line-(val/(1+z[0]))).argmin()
        
        #print(line-(i/(1+z[0])))
        
        #gets the rest frame wavelength from the catalog
        rest_l = line[index]
        rest_name = line_name[index]
        
        
        
        #this finds the index of where the rest line is closest to in the emission lines that I got
        #this index can be used for equivalent width and flux 
        ind = abs(rest_l*(1+z[0]) - emission_line_fit).argmin()
        
        if rest_l in rest_line:
            continue
        
        else:
            
            #print('%13s ----- %9.2f ------- %9.2f ------- %9.2f ------- %9.2f' %(rest_name, rest_l, val/(1+z), line_f[ind], manual_ew[ind]))
            
            ind_catalog.append(index)
            ind_calculation.append(ind)
            
            #plt.axvline(rest_l, linestyle = '--', color = 'red', linewidth = .5)
            #plt.axvline(val, linestyle = '--', color = 'blue', linewidth = .5)
            #plt.text(rest_l  + 2 , np.amax(spectrum.flux.value)/3, rest_name, rotation = 270, fontsize = 'x-small')
            
           
            
            rest_line.append(rest_l)
            name_line.append(rest_name)
            line_center.append(round(val/(1+z[0]), 2))
            flux.append(round(line_f[ind], 2))
            EW.append(round(manual_ew[ind], 2))
            
            #print('%13s ----- %9.2f ------- %9.2f ------- %9.2f ------- %9.2f' %(rest_name, rest_l, val/(1+z[0]), line_f[ind], manual_ew[ind]))
            #print()
            
    #plt.savefig(filename[7:-5]+'_z_'+str(z[0])+'.pdf')
    #plt.show()
    
    line_catalog_filt[ind_catalog] = False
    calculation_filt[ind_calculation] = False
    
    t = Table()
    t['line_name'] = np.array(name_line)
    t['rest_frame_wavelength'] = np.array(rest_line)
    t['calculated_center'] = np.array(line_center)
    t['line_flux'] = np.array(flux)
    t['line_EW'] = np.array(EW)
    
    print(t)
    print()
    '''
    print('Emission Line ------ Emission Fit------ ew_spec ------- ew_manual ------- flux_spec ------ manual_flux ------- continuum val')
    for i in range(len(e_width)):
        print('%13.2f ------ %12.2f ------ %7.2f ------- %9.2f ------- %9.2f ------ %11.2f ------- %13.2f' 
              %(emission_lines[i]/(1+z), emission_line_fit[i]/(1+z), e_width[i].value, 
                manual_ew[i], lines_flux[i].value, line_f[i], continuum_val[i].value))
    
    
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
    
    return np.array(emission_lines), np.array(line_f), np.array(manual_ew)

def lines_of_interest(line_names, emission, line_rest, z):
    
    '''
    This function will try to give us measurements regarding lines that we are interested in
    
    '''
    
    helium_filt = [True if 'He' in x else False for x in line_name]
    oxygen_filt = [True if 'OII' in x else False for x in line_name]
    alpha_filt = np.array([True if 'Halpha' in x else False for x in line_name])
    beta_filt = np.array([True if 'Hbeta' in x else False for x in line_name]) 
    NII_filt = np.array([True if 'NII' in x else False for x in line_name])
    
    master_filter = np.ones(len(oxygen_filt), dtype = bool)
    
    for i in range(len(alpha_filt)): 
        if alpha_filt[i] or beta_filt[i] or NII_filt[i] or helium_filt[i] or oxygen_filt[i]: 
            master_filter[i] = True 
        else: 
            master_filter[i] = False
    
    interest_lines = line_rest[master_filter]
    interest_names = line_names[master_filter]
    
    conv_to_rest = emission/(1+z)
    #print(interest_names)
    #print(interest_lines)
          
    for i in conv_to_rest:
        ind = abs(interest_lines - i).argmin()
        #print(np.delete(interest_lines, ind))
        #print(np.delete(interest_names, ind))
     
    pass
    
def possible_flux(line_name, line_wavelength, z, flux, wavelength):
    
    ind1 = np.where(line_wavelength < wavelength[0]/(1+z))
    ind2 = np.where(line_wavelength > wavelength[-1]/(1+z))
    ind_tot = np.concatenate((ind1, ind2), axis = None)
    
    master_filt = np.ones(len(line_name), dtype = bool)
    
    master_filt[ind_tot] = False
    
    reduced_line_name = line_name[master_filt]
    reduced_line_wave = line_wavelength[master_filt]
    
    plt.plot(wavelength/(1+z), flux)
    
    for i in reduced_line_wave:
        plt.axvline(i, linestyle = '--', color = 'red')
    
    plt.show()
    
def sorting_info(z, line_rest, line_names, emission, flux, ew):
    
    conv_to_rest = emission/(1+z)
    line_nam = []
    line_wave = []
    
    for i in conv_to_rest:
        
        ind = abs(line_rest-i).argmin()
        line_nam.append(line_names[ind])
        line_wave.append(line_rest[ind])
    
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

'''

file = 'median_J014707+135629_cem.fits'
sp1 = file.split('_')[1]
sp2 = sp1.split('+')[0]

match = ID_1 == sp2
#print(match)
z1 = z_redshift[match]
#print(z1)

spec, wvln = spectrum(file)

possible_flux(line_name, line_wavelength, z1, spec, wvln)

spectra, cont_func, filt= fitting_continuum(wvln, spec, z1, line_wavelength, file)
ind = np.where(wvln < 4500)
ind2 = np.where(wvln > 6350)
ind_tot = np.concatenate((ind, ind2), axis=None)
filt[ind_tot] = False
percentile = 98
print(i)
emission_line, flux, EW = analysis(spectra, wvln, filt, line_wavelength, z1, cont_func, percentile, line_name)
lines_of_interest(line_name, emission_line, line_wavelength, z1)

'''



for i in files:
    
    if 'cem.fits' in i:
        
        sp1 = i.split('_')[1]
        sp2 = sp1.split('+')[0]
        
        match = ID_1 == sp2
        #print(match)
        z1 = z_redshift[match]
        
        seeing_t = seeing[match]
        
        conversion = .188
        #print(z1)
        
        spec, wvln = spectrum(i)
        
        spectra, cont_func, filt= fitting_continuum(wvln, spec, z1, line_wavelength, i)
        ind = np.where(wvln < 4500)
        ind2 = np.where(wvln > 6350)
        ind_tot = np.concatenate((ind, ind2), axis=None)
        filt[ind_tot] = False
        percentile = 98
        print(i)
        print('-----------------------')
        print()
        emission_line, flux, EW = analysis(spectra, wvln, filt, line_wavelength, z1, cont_func, percentile, line_name, i, seeing_t, conversion)
        
        #ax1.plot(wvln[filt], spec[filt], label = i)
        
    if 'mods1b' in i:
        
        sp1 = i.split('_')[1]
        sp2 = sp1.split('+')[0]
        
        match = ID_1 == sp2
        
        #print(match)
        z1 = z_redshift[match]
        seeing_t = seeing[match]
        
        conversion = .12
        
        #print(z1)
        spec, wvln = spectrum(i)
        spectra, cont_func, filt= fitting_continuum(wvln, spec, z1, line_wavelength, i)
        percentile = 97
        print(i)
        print('-----------------------')
        print()
        emission_line, flux, EW = analysis(spectra, wvln, filt, line_wavelength, z1, cont_func, percentile, line_name, i, seeing_t, conversion)
        
        #ax2.plot(wvln, spec, label = i)
    
    
    if 'mods1r' in i:
        
        sp1 = i.split('_')[1]
        sp2 = sp1.split('+')[0]
        
        match = ID_1 == sp2
        #print(match)
        z1 = z_redshift[match]
        seeing_t = seeing[match]
        
        conversion = .123
        
        #print(z1)
        spec, wvln = spectrum(i)
        spectra, cont_func, filt= fitting_continuum(wvln, spec, z1, line_wavelength, i)
        percentile = 99
        print(i)
        print('-----------------------')
        print()
        emission_line, flux, EW = analysis(spectra, wvln, filt, line_wavelength, z1, cont_func, percentile, line_name, i, seeing_t, conversion)
        
        #ax3.plot(wvln, spec, label = i)
        

'''
#ax1.set_ylim(-2, 75)
#ax2.set_ylim(-2, 75)
#ax3.set_ylim(-2, 75)
#ax1.legend(loc='best')
#ax2.legend(loc='best')
#ax3.legend(loc='best')
#plt.show()    
           
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
