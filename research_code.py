from astropy.io import fits
import numpy as np
import pylab as plt
import glob as glob
from scipy.stats import mode
from matplotlib.colors import LogNorm
from specutils import get_wcs_solution
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec


#grabbing all of micaela's fits files from the current directory
files = [x for x in glob.glob('*.fits') if 'SDSS' not in x]

#getting all the sdss fits files so we can do flux calibration
sdss_files = [x for x in glob.glob('*.fits') if 'SDSS' in x]

#gettting only good file by getting rid of my box_data.fits file
#good_files = files[1:]

#reading in the dat file
ID, redshift = np.genfromtxt('targets.dat', usecols=(0,1), unpack = True, skip_header=2, dtype = 'str')

z = redshift.astype(float)


'''
    This is some code that worked before and in case the new version of finding the row index does not work iI have this backup

    #Code starts below

    for i in range(len(data[:,0])):
        
        #this counter will be used to count the number of positive numbers in a row. Initialized at zero
        #so that at each row interation we restart the count
        counter = 0
        
        #this part goes through each column in the row at index i
        for j in range(len(data[0,:])):

            #the code below checks to see if the condition for positive number is satisfied
            
            if data[i,j] > 100:
                continue

            if data[i,j] > 1:
                counter +=1
            else:
                continue
        #here im giving the list mentioned above the value of counter
        positive_num.append(counter)
        
        #im also giving it the row index from the for loop
        row.append(i)
'''

def fitting_gaussian(data, x):
    
    '''
    This function will find the gaussian fit to the column data after it is boxed. Meaning we know where the center of the
    spectrum is at and we added plus or minus 50, in our case to it. Then make a spectrum from it by finding the maximum.

    Parameter
    -------------
    data: this is the data that we would like to fit with a gaussian model. This could be anything from emission lines to columns of the boxed
          data sets.
    
    x: this is the x values that will be fitted with the gaussian

    Output
    -------------
        
    
    '''
    
    def f (x, A, x0, sigma):
        
        '''
        Making a Gaussian Function to fit the columns using curve_fit
        '''
        
        return A * np.exp(-(x-x0)**2/(2*sigma**2))
    
    
    #here is the fitting of the data and x values to the gaussian and gives us the optimal paramters of A, x0, sigma 
    popt, covar = curve_fit(f, x, data)
    
    #this part makes the gaussian function with the parameters fit from above
    y = f(x, *popt)
    
    #plt.figure()
    #plt.plot(x, y)    

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
    
    filt = np.ones(data.shape)
    
    if '_cem.fits' in file:
            
        row_min = 220
        row_max = 500
        
        filt[:row_min, :] = np.zeros(len(data[0,:]))
        filt[row_max:, :] = np.zeros(len(data[0,:]))
 
    
    if 'b_ce.fits' in file:

        row_min = 1410
        row_max = 1890

        filt[:row_min, :] = np.zeros(len(data[0,:]))
        filt[row_max:, :] = np.zeros(len(data[0,:]))
    
    if 'r_ce.fits' in file:

        row_min = 1350
        row_max = 1800

        filt[:row_min, :] = np.zeros(len(data[0,:]))
        filt[row_max:, :] = np.zeros(len(data[0,:]))
    
    
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
    #We used a function called finding_row_center to find the index of the simplified data and add it to the respective row_min
    row_spectrum = finding_row_center(data*filt)

    window = 50
    
    #given the row where spectrum is at we are looking at 50 rows above and below it
    boxed_data = data[row_spectrum - window : row_spectrum + window ,:]
    
    p = get_wcs_solution(hdr)
    
    X, Y = np.meshgrid(range(len(data[0,:])), range(len(data[:,0])))

    wvln_arr = p(X, Y)

    boxed_wvln_array = wvln_arr[row_spectrum - window: row_spectrum + window,:]
    
    x = range(len(data[0,:]))
    y = row_spectrum * np.ones(len(x))

    #plt.figure(figsize = (10,10))
    #plt.imshow(filt*data, cmap = 'gray', origin = 'lower', norm = LogNorm())
    #plt.plot(x, y)
    #plt.show()

    adding_target = 0
    N = 0
    wvln_spec = wvln_arr[row_spectrum,:]
    

    for i in range(len(boxed_data[:,0])):
        
        f = interp1d(boxed_wvln_array[i,:], boxed_data[i,:])
        adding_target += f(wvln_spec)
        N += 1
    
    
    target_spec_avg = adding_target/N
    
    
    fig = plt.figure(figsize = (14,10), constrained_layout = True)
    spec = gridspec.GridSpec(ncols=2, nrows=2,figure = fig)

    ax1 = fig.add_subplot(spec[0,0])
    ax2 = fig.add_subplot(spec[-1, 0], sharex = ax1)
    ax3 = fig.add_subplot(spec[:, 1])

    ax1.plot(wvln_spec, adding_target)
    ax2.plot(wvln_spec, target_spec_avg)
    
    ax1.set_title('Not Averaged Spec')
    ax2.set_title('Averaged Spec')

    ax2.set_xlabel(r'Wavelength [$\AA$]')
    
    ax3.set_title(file)
    ax3.imshow(filt*data, cmap = 'gray', origin = 'lower', norm = LogNorm())
    ax3.plot(x, y)

    #.legend(loc='best')
    #fig.tight_layout()
    plt.show()
    
    #straight summin up the columns together
    #spectrum = np.sum(boxed_data, axis = 0)
    

    ############################
    #This is me using another way to get the 1D spectrum by looking at the max value of each column within boxed_data
    ############################

    #getting the max values for each column of boxed_data
    max_values_col = np.amax(boxed_data, axis = 0)
    
    #for i, val in enumerate(max_values_col):
        #if val > 100:
         #   max_values_col[i] = 0
    
    '''    
    x = range(len(data[0,:]))
    y = row_spectrum * np.ones(len(x))

    plt.figure(figsize = (10,10))
    plt.imshow(data*filt, cmap = 'gray', origin = 'lower', norm=LogNorm())
    plt.plot(x, y, 'r--')
    plt.show()
    
    #getting the polynomial that will map (x,y) pixels to wavelength in angstrom, assigningthis polynomial to p
    p = get_wcs_solution(hdr)
    

    #making the x and y data that I will pass into the polynomial
    x = range(len(data[row_spectrum, :]))
    y = row_spectrum * np.ones(len(x))

    wavelength = p(x, y)

    #code that plots it so that i can see what the 1D spectrum looks like
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

for i in files[:2]:
    spectrum(i)

'''
file1 = good_files[0]
file2 = good_files[-1]

spec1 = spectrum(file1)
spec2= spectrum(file2)


fig = plt.figure(figsize = (14,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(spec1)
ax2.plot(spec2)

fig.tight_layout()

plt.show()
'''
