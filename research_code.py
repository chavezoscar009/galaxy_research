from astropy.io import fits
import numpy as np
import pylab as plt
import glob as glob

from matplotlib.colors import LogNorm

files = [x for x in glob.glob('*.fits')]

good_files = files[1:]

def spectrum(file):
    
    #hdu = fits.open(file[0])
    data = fits.getdata(file)

    #plt.imshow(data, origin='lower', cmap ='gray', norm =LogNorm())
    #plt.show()
    
    #This code will try to reduce the 2D array by a set amount as we dont need the boundary
    #the amount that we will do will be assigned to variables
    row_min = len(data[:,0])//10
    row_max = len(data[:,0]) - row_min
    column_min = len(data[0,:])//4
    column_max = len(data[0,:])-column_min
    
    cut_data = data[row_min:row_max, column_min:column_max] 
    row_spectrum = row_min + np.argmax(cut_data.mean(axis = 1))
  
    window = 20
    
    boxed_data = data[row_spectrum - window : row_spectrum + window ,:]

    spectrum = np.sum(boxed_data, axis = 0)
    
    return spectrum

file1 = good_files[0]

file2 = good_files[1]

spec1 = spectrum(file1)
spec2= spectrum(file2)


fig = plt.figure(figsize = (14,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(spec1)
ax2.plot(spec2)

fig.tight_layout()

plt.show()

