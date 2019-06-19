from scipy.optimize import curve_fit
import numpy as np
import pylab as plt

#importing data that i made from a spectrum from sdss
filo = np.load('test_gaussians.npz')

#assigning them to their respective variables

#data 1 is the spectrum of the gaussian
data1 = filo['gauss1']

#the x values corresponding to the gaussian
x1 = filo['x1']

#spectrum value for the second spec
data2 = filo['gauss2']

#second specs x-values
x2 = filo['x2']

#getting the test_gauss stuff that i made to test my gauss fit code
#filo2= np.load('test_gauss.npz')

#assigning them to their respective variables
data3 = filo['gauss3']
x3 = filo['x3']

data4 = filo['gauss4']
x4 = filo['x4']


filo2 = np.load('spec-0280.npz')

data5 = filo2['data1']
x5 = filo2['x1']

data6 = filo2['data2']
x6 = filo2['x2']

data7 = filo2['data3']
x7 = filo2['x3']
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
  
    def f (x, A, x0, sigma, b):
  
        '''
        Making a Gaussian Function to fit using scipy's curve_fit function
        '''
  
        return A * np.exp(-(x-x0)**2/(2*sigma**2)) + b
  
  
    #here is the fitting of the data and x values to the gaussian and gives us the optimal paramters of A, x0, si    gma 
    popt, covar = curve_fit(f, x, data, bounds = ([0, x[0], 1e-2, 0], [np.amax(data), x[-1], 100, np.max(data)//2]))
  
    #this part makes the gaussian function with the parameters fit from above
    y = f(x, *popt)
    
    #me plotting to see if the fit did a good job
    #plt.figure(figsize = (14, 8))
    #plt.plot(x, y, label = 'Gaussian Fit Data')
    #plt.plot(x, data, label = 'Original Data')
    #plt.axvline(x[np.argmax(data)])
    #plt.show()
    
    perr = np.sqrt(np.diag(covar))

    gauss_fit_values = {'Amp': popt[0], 'Amp_err': perr[0], 'mean': popt[1], 
                        'mean_err': perr[1], 'sigma': popt[2], 'sigma_err':perr[2], 
                        'b': popt[3], 'b_err': perr[3]}
    

    return y, gauss_fit_values    


def evaluate_gaussian(values ):
    
    '''
    This function is going to give me the value of the gaussian integral:

    Parameters
    ---------------
    values: this should be a dictionary with the optimal values for the gaussian fit

    Output
    ---------------
    Value of the gaussian integral from the values passed in

    '''

    A = values['Amp']
    sigma = values['sigma']

    return A * sigma * np.sqrt(2 * np.pi)



#d, n = fitting_gaussian(data1, x1)
#print('flux is: %5.2f' %(evaluate_gaussian(n)) )
#p, q = fitting_gaussian(data2, x2)
#print('flux is: %5.2f' %(evaluate_gaussian(q)) )

correct_values = [177.7, 61.78, 976.3, 3184, 10080]

calculated = []

w, t = fitting_gaussian(data3, x3)
print('flux is: %5.2f' %(evaluate_gaussian(t)) )
calculated.append(evaluate_gaussian(t))

z, a =fitting_gaussian(data4, x4)
print('flux is: %5.2f' %(evaluate_gaussian(a)) )
calculated.append(evaluate_gaussian(a))

m, o = fitting_gaussian(data5, x5)
print('flux is: %5.2f' %(evaluate_gaussian(o)) )
calculated.append(evaluate_gaussian(o))

z, e = fitting_gaussian(data6, x6)
print('flux is: %5.2f' %(evaluate_gaussian(e)) )
calculated.append(evaluate_gaussian(e))

i, u = fitting_gaussian(data7, x7)
print('flux is: %5.2f' %(evaluate_gaussian(u)) )
calculated.append(evaluate_gaussian(u))

def line(x, m, b):
    return m * x + b

popt, covar = curve_fit(line, calculated, correct_values)

x = np.linspace(np.amin(calculated), np.amax(calculated))
y = line(x, *popt)

plt.figure(figsize = (12, 8))
plt.xlabel('Calculated Flux')
plt.ylabel('Observed Flux')
plt.plot(calculated, correct_values, label = 'Values')
plt.plot(x, y, label = 'Best Fit')
plt.show()
#print(t)
#print()
#print(a)
