from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


#opening up the galaxy info file
hdu = fits.open('galSpecInfo-dr8.fits')
table1 = hdu[1].data

#getting the spectral object ID
specObjID = table1['SpecObjID']
spec_z = table1['Z']

#filtering out the galaxies to only include the ones in my redshift regime
filt = np.array([True if (x > .025 and x < .06) else False for x in spec_z])

hdu.close()

#opening up the galaxy line file
hdu2 = fits.open('galSpecLine-dr8.fits')

#getting the table
table2 = hdu2[1].data

#closing the file
hdu2.close()

#getting the OIII line measurements 
OIII5007 = table2['OIII_5007_FLUX'][filt]
OIII5007_err = table2['OIII_5007_FLux_err'][filt]
OIII5007_EW = table2['OIII_5007_EQW'][filt]

#getting the Hbeta measurement
Hbeta = table2['H_BETA_FLUX'][filt]
Hbeta_err = table2['H_BETA_FLUX_err'][filt]
Hbeta_EW = table2['H_BETA_EQW'][filt]

#getting Halpha line measurements
Halpha = table2['H_ALPHA_FLUX'][filt]
Halpha_err = table2['H_ALPHA_FLUX'][filt]
Halpha_EW = table2['H_ALPHA_EQW'][filt]

#getting NII line measurements
NII6584 = table2['NII_6584_FLUX'][filt]
NII6584_err = table2['NII_6584_FLUX_err'][filt]
NII6584_EW = table2['NII_6584_EQW'][filt]

Obj_ID = specObjID[filt]

negative_num_filt = np.ones(len(Hbeta), dtype = bool)

for i in range(len(Hbeta)):
    if OIII5007[i] < 0 or Hbeta[i] < 0 or Halpha[i] < 0 or NII6584[i] < 0:
        negative_num_filt[i] = False

    if OIII5007[i] ==  0 or Hbeta[i] == 0 or Halpha[i] == 0 or NII6584[i] ==  0:
        negative_num_filt[i] = False

OIII_Hbeta = OIII5007[negative_num_filt]/Hbeta[negative_num_filt]
NII_halpha = NII6584[negative_num_filt]/Halpha[negative_num_filt]

def plotting_BPT(Table):
    
    filt = np.zeros(len(Table), dtype = bool)

    for i, val in enumerate(Table['NII_Halpha_ratio']):
        if val > 0:
            filt[i] = True
    
    new_table = Table[filt]

    OIII_HB = new_table['OIII_Hbeta_ratio']
    NII_HA = new_table['NII_Halpha_ratio']
    obj = new_table['ObjectID']
    
    x = np.logspace(-2, 1.5, 1000)
    y = .61/(np.log10(x) - .47) + 1.19
    
    
    plt.figure(figsize = (10,10))
    plt.title('BPT Diagram')
    plt.xlabel(r'$log(\frac{[NII]6854}{H\alpha}) $ ', fontsize = 15)
    plt.ylabel(r'$log(\frac{[OIII]5007}{H\beta}) $ ', fontsize = 15)
    plt.xlim(-2, 1.5)
    plt.ylim(-1.5, 2)
    
    plt.plot(np.log10(NII_halpha), np.log10(OIII_Hbeta), '.', alpha = .4)
    plt.plot(np.log10(x), y, 'r--', linewidth = .5)
    
    for i in range(len(OIII_HB)):
        plt.plot(np.log10(NII_HA[i]), np.log10(OIII_HB[i]), '*', markersize=10 ,label = obj[i])
    
    plt.legend(loc = 'best')
    plt.savefig('BPT_Diagram.pdf')
    plt.show()







