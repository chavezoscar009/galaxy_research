from astropy.io import fits
import numpy as np
import pylab as plt
from glob import glob

from matplotlib.colors import LogNorm

file = glob('*.fits')

hdu = fits.open(file[0])

data = fits.getdata(file[0])

plt.imshow(data, origin='lower', cmap ='gray', norm =LogNorm())
plt.show()

