from scipy.optimize import curve_fit
import numpy as np
import pylab as plt

#importing data that i made from a spectrum from sdss
filo = np.load('spectrum_data.npz')

data1 = filo['data1']
x1 = filo['x1']

data2 = filo['data2'][:-100]
x2 = filo['x2'][:-100]



def fitting_gaussian(data, x):
  
    '''
    This function will find the gaussian fit to the column data after it is boxed. Meaning we know where the center of the
    spectrum is at and we added plus or minus 50, in our case to it. Then make a spectrum from it by finding the maximum.

    Parameter
    ------------
    data: this is the data that we would like to fit with a gaussian model. This could be anything from emission lines to columns of boxed data sets

    x: this is the x values where the gaussian will be fitted around
  
    Output
    -------------
    A gaussian fit of the data. ie: returns a gaussian that best fits the x and data passed in the parameters     
      
    '''
  
    def f (x, A, x0, sigma):
  
        '''
        Making a Gaussian Function to fit using scipy's curve_fit function
        '''
  
        return A * np.exp(-(x-x0)**2/(2*sigma**2))
  
  
    #here is the fitting of the data and x values to the gaussian and gives us the optimal paramters of A, x0, si    gma 
    popt, covar = curve_fit(f, x, data, p0 = (np.max(data), x[np.argmax(data)] , 1), bounds = ([0, x[0], 1e-2], [np.amax(data), x[-1], 100] ))
  
    #this part makes the gaussian function with the parameters fit from above
    y = f(x, *popt)
    
    #me plotting to see if the fit did a good job
    plt.figure(figsize = (14, 8))
    plt.plot(x, y, label = 'Gaussian Fit Data')
    plt.plot(x, data, label = 'Original Data')
    plt.axvline(x[np.argmax(data)])
    plt.show()
    
    #return y    




fitting_gaussian(data1, x1)
fitting_gaussian(data2, x2)
