from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def gal_info(filename):

    #opening up the galaxy info file
    hdu = fits.open(filename)#'galSpecInfo-dr8.fits')
    table1 = hdu[1].data

    #getting the spectral object ID
    specObjID = table1['SpecObjID']
    spec_z = table1['Z']

    hdu.close()
    return specObjID, spec_z

def z_fitlered(spec_z):
    #filtering out the galaxies to only include the ones in my redshift regime
    filt = np.array([True if (x > .025 and x < .06) else False for x in spec_z])

    return filt

def gal_line_info(filename):
    #opening up the galaxy line file
    hdu2 = fits.open(filename)#'galSpecLine-dr8.fits')

    #getting the table
    table2 = hdu2[1].data

    #closing the file
    hdu2.close()

    return table2

def OIII_vals(table2):

    #getting the OIII line measurements
    OIII5007 = table2['OIII_5007_FLUX']
    OIII5007_err = table2['OIII_5007_FLux_err']
    OIII5007_EW = table2['OIII_5007_EQW']

def Hbeta_vals(table2):

    #getting the Hbeta measurement
    Hbeta = table2['H_BETA_FLUX']
    Hbeta_err = table2['H_BETA_FLUX_err']
    Hbeta_EW = table2['H_BETA_EQW']

def Halpha_vals(table2):

    #getting Halpha line measurements
    Halpha = table2['H_ALPHA_FLUX']
    Halpha_err = table2['H_ALPHA_FLUX']
    Halpha_EW = table2['H_ALPHA_EQW']

def NII_vals(table2):

    #getting NII line measurements
    NII6584 = table2['NII_6584_FLUX']
    NII6584_err = table2['NII_6584_FLUX_err']
    NII6584_EW = table2['NII_6584_EQW']

def master_filt(filt, OIII5007, OIII5007_err ,Hbeta, Halpha, Halpha_err, NII6584):

    ind_neg = np.where((OIII5007 < 0) | (Hbeta < 0) | (Halpha < 0 ) | (NII6584 < 0))
    ind_zero = np.where((OIII5007 == 0) | (Hbeta == 0) | (Halpha == 0 ) | (NII6584 == 0))
    ind_neg_err = np.where(OIII5007_err < 0 | Halpha_err < 0)
    ind_low_SN = np.where(OIII5007/OIII5007_err < 5 | Halpha/Halpha_err < 5)

    master_ind = np.unique(np.concatenate([ind_neg, ind_zero, ind_neg_err, ind_low_SN], axis = None))

    

'''
for i in range(len(Hbeta)):

    if OIII5007[i] < 0 or Hbeta[i] < 0 or Halpha[i] < 0 or NII6584[i] < 0:
        bad_values_filt[i] = False

    if OIII5007[i] ==  0 or Hbeta[i] == 0 or Halpha[i] == 0 or NII6584[i] ==  0:
        bad_values_filt[i] = False

    if OIII5007_err[i] < 0 or Halpha_err[i] < 0:
        bad_values_filt[i] = False

    if OIII5007[i]/OIII5007_err[i] < 5 or Halpha[i]/Halpha_err[i] < 5:
        bad_values_filt[i] = False

OIII_Hbeta = OIII5007[bad_values_filt]/Hbeta[bad_values_filt]
NII_halpha = NII6584[bad_values_filt]/Halpha[bad_values_filt]
'''

def plotting_BPT(Table):

    #making a filter to only get the ones that have the ratios there
    filt = np.zeros(len(Table), dtype = bool)

    #for loop that cjhecks to see if ratio is in table if it is assigns that index to True
    for i, val in enumerate(Table['NII6583/Halpha']):
        if val > 0:
            filt[i] = True

    #makes a sub Tbale with only the objects with ratios
    new_table = Table[filt]

    #getting the ratios of interest and object name
    OIII_HB = new_table['OIII5007/Hbeta']
    NII_HA = new_table['NII6583/Halpha']
    obj = new_table['ObjectID']

    #making an x array that will be used for plotting the cutoff points
    x = np.logspace(-2, 1.5, 1000)

    #got this from Kewely et al
    y = .61/(np.log10(x) - .47) + 1.19

    #got this from Kauffmann et al 2003 for AGN classification
    y_agn = .61/(np.log10(x) - .05) + 1.3

    #making a filter for the Kewely function getting rid of the line
    filt1 = np.ones(len(x), dtype = bool)

    #making a filter for the Kauffmann funciton to get rid of the line
    filt2 = np.ones(len(x), dtype = bool)

    #getting indices where the log(x) are bigger than some number to apply the filtering
    ind = np.where(np.log10(x) > 0.4)
    ind2 = np.where(np.log10(x) > 0)

    #assigning those values to false so that i can mask them out in the respective functions
    filt1[ind] = False
    filt2[ind2] = False

    #plotting the BPT diagram
    plt.figure(figsize = (10,10))
    plt.title('BPT Diagram')
    plt.xlabel(r'$log(\frac{[NII]6854}{H\alpha}) $ ', fontsize = 15)
    plt.ylabel(r'$log(\frac{[OIII]5007}{H\beta}) $ ', fontsize = 15)
    plt.xlim(-2, 1.5)
    plt.ylim(-1.5, 2)

    plt.plot(np.log10(NII_halpha), np.log10(OIII_Hbeta), '.', alpha = .4)
    plt.plot(np.log10(x)[filt1], y[filt1], 'k--', linewidth = 1)
    plt.plot(np.log10(x)[filt2], y_agn[filt2], 'b--')

    for i in range(len(OIII_HB)):
        plt.plot(np.log10(NII_HA[i]), np.log10(OIII_HB[i]), '*', markersize=15 ,label = obj[i])

    plt.legend(loc = 'best', frameon=False)
    #plt.savefig('BPT_Diagram.pdf')
    plt.show()

def SFR_galaxies(Table):

    #making an x array that will be used for plotting the cutoff points
    x = np.logspace(-2, 1.5, 1000)

    #got this from Kewely et al
    y = .61/(np.log10(x) - .47) + 1.19

    #making a filter to get only the star forming galaxies
    SFR_filt = np.ones(len(OIII_Hbeta), dtpye = bool)

    #getting the indices where the log(OIII/Hbeta) is bigger than the cutoff
    sfr = np.where(np.log10(OIII_Hbeta) > y[filt1])

    #Setting the indices where they are above the cutoff to false
    SFR_filt[sfr] = False
    pass

#getting OIII4959
OIII4959 = table2['OIII_4959_FLUX'][filt]
OIII4959_err = table2['OIII_4959_FLux_err'][filt]
OIII4959_EW = table2['OIII_4959_EQW'][filt]

#getting the OII line measurements
OII3726 = table2['OII_3726_FLUX'][filt]
OII3726_err = table2['OII_3726_FLux_err'][filt]
OII3726_EW = table2['OII_3726_EQW'][filt]

negative_num_R32 = np.ones(len(Hbeta), dtype = bool)

for i in range(len(Hbeta)):
    if OIII5007[i] < 0 or Hbeta[i] < 0 or OIII4959[i] < 0 or OII3726[i] < 0:
        negative_num_R32[i] = False

    if OIII5007[i] ==  0 or Hbeta[i] == 0 or OIII4959[i] == 0 or OII3726[i] == 0:
        negative_num_R32[i] = False

R32 = (OII3726[negative_num_R32] + OIII5007[negative_num_R32] + OIII4959[negative_num_R32])/Hbeta[negative_num_R32]

def plotting_R32(Table):
    pass
