from astropy.io import fits
import numpy as np
import pylab as plt
import glob as glob
from scipy.stats import mode
from matplotlib.colors import LogNorm

files = [x for x in glob.glob('*.fits')]

good_files = files[1:]
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

def fitting_gaussian(data):
    '''
    This function will find the gaussian fit to the column data after it is boxed. Meaning we know where the center of the
    spectrum is at and we added plus or minus 50, in our case to it. Then make a spectrum from it by finding the maximum.

    Parameter
    -------------
    data: this is the boxed data that is passed in.

    Output
    -------------
    spec_1D_gauss: A 1D spectrum using the gaussian method
    '''
    
    def f (x, A, x0, sigma):
        
        '''
        Making a Gaussian Function to fit the columns using curve_fit
        '''
        
        return A * np.exp(-(x-x0)**2/(2*sigma)**2)
    
    #making a list to gather all of the max values of the fitted gaussians
    max_gaussian = []
    
    x = range(len(data[:,0]))
    #fitting columns to gaussians for every column
    for i in range(len(x)):

        col = data[:, i]
        
        popt, covar = curve_fit(f, x, col, p0 = (1, x[len(x)//2], 1))

        y = f(x, *popt)

        max_gaussian.append(np.amax(y))
    
    #plt.figure(
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
    column_min = 500
    column_max = 3720
    
    #this one goes and checks the len of the split file and if its 3 then assigns one row_min and row_max
    #if the array is length of 4 then wit further splits it into red and blue components and assigns 
    #appropriate row_min and row_max value. These were found by using ds9 and were pik as to not include the slit
    
    if len(x) ==3:

        row_min = 220
        row_max = 500

    elif len(x) == 4:

        if x[2][-1] == 'b':

            row_min = 1410
            row_max = 1890

        if x[2][-1] == 'r':

            row_min = 1350
            row_max = 1800


    #declaring the cut_data array below, which is a simplified version of the data gathered above. It should only
    #contain the spectrum with a box excluding any slit effects
    cut_data = data[row_min:row_max, column_min:column_max] 
    
    #plt.figure(figsize = (14, 8))
    #plt.imshow(cut_data, origin = 'lower', cmap = 'gray',norm = LogNorm())
    #plt.show()
    

    #This one calculates where in the original data array the correct row_index corresponding to the center of the spectrum lies
    #We used a function called finding_row_center to find the index of the simplified data and add it to the respective row_min
    row_spectrum = row_min + finding_row_center(cut_data)
    
    window = 50
    
    #given the row where spectrum is at we are looking at 50 rows above and below it
    boxed_data = data[row_spectrum - window : row_spectrum + window ,:]
    
    #straight summin up the columns together
    spectrum = np.sum(boxed_data, axis = 0)
    

    ############################
    #This is me using another way to get the 1D spectrum by looking at the max value of each column within boxed_data
    ############################

    #getting the max values for each column of boxed_data
    max_values_col = np.amax(boxed_data, axis = 0)
    
    #for i, val in enumerate(max_values_col):
        #if val > 100:
         #   max_values_col[i] = 0

    #code that plots it so that i can see what the 1D spectrum looks like
    plt.figure(figsize = (14,8))
    plt.plot(range(len(data[0,:])), spectrum)
    plt.show()

    return spectrum


for i in good_files:
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
