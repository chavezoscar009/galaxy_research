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

def z_filtered(spec_z):
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

def OIII_vals(table2, filt):

    #getting the OIII line measurements
    OIII5007 = table2['OIII_5007_FLUX'][filt]
    OIII5007_err = table2['OIII_5007_FLux_err'][filt]
    OIII5007_EW = table2['OIII_5007_EQW'][filt]
    OIII5007_EW_ERR = table2['OIII_5007_EQW_err'][filt]

    return OIII5007, OIII5007_err, OIII5007_EW, OIII5007_EW_ERR

def Hbeta_vals(table2, filt):

    #getting the Hbeta measurement
    Hbeta = table2['H_BETA_FLUX'][filt]
    Hbeta_err = table2['H_BETA_FLUX_err'][filt]
    Hbeta_EW = table2['H_BETA_EQW'][filt]
    Hbeta_EW_ERR = table2['H_BETA_EQW_ERR'][filt]

    return Hbeta, Hbeta_err, Hbeta_EW, Hbeta_EW_ERR

def Halpha_vals(table2, filt):

    #getting Halpha line measurements
    Halpha = table2['H_ALPHA_FLUX'][filt]
    Halpha_err = table2['H_ALPHA_FLUX_err'][filt]
    Halpha_EW = table2['H_ALPHA_EQW'][filt]
    Halpha_EW_ERR = table2['H_ALPHA_EQW_ERR'][filt]

    return Halpha, Halpha_err, Halpha_EW, Halpha_EW_ERR

def NII_vals(table2, filt):

    #getting NII line measurements
    NII6584 = table2['NII_6584_FLUX'][filt]
    NII6584_err = table2['NII_6584_FLUX_err'][filt]
    NII6584_EW = table2['NII_6584_EQW'][filt]
    NII6584_EW_ERR = table2['NII_6584_EQW_ERR'][filt]

    return NII6584, NII6584_err, NII6584_EW, NII6584_EW_ERR

def OII_vals(table2, filt):

    #getting NII line measurements
    OII3726 = table2['OII_3726_FLUX'][filt]
    OII3726_err = table2['OII_3726_FLUX_err'][filt]
    OII3729 = table2['OII_3729_FLUX'][filt]
    OII3729_err = table2['OII_3729_FLUX_ERR'][filt]
    
    return OII3726, OII3726_err, OII3729, OII3729_err

def bad_data_filt(OIII5007, OIII5007_err ,Hbeta, Halpha, Halpha_err, NII6584):

    filt = np.ones(len(OIII5007), dtype = bool)

    for i in range(len(Hbeta)):

        if OIII5007[i] <= 0 or Hbeta[i] <= 0 or Halpha[i] <= 0 or NII6584[i] <= 0:
            filt[i] = False

        if OIII5007_err[i] <= 0 or Halpha_err[i] <= 0:
            filt[i] = False



    return filt

specObjID, spec_z = gal_info('galSpecInfo-dr8.fits')
filt = z_filtered(spec_z)
table = gal_line_info('galSpecLine-dr8.fits')

OIII5007, OIII5007_err, OIII5007_EW, OIII5007_EW_ERR = OIII_vals(table, filt)
Hbeta, Hbeta_err, Hbeta_EW, Hbeta_EW_ERR = Hbeta_vals(table, filt)
Halpha, Halpha_err, Halpha_EW, Halpha_EW_ERR = Halpha_vals(table, filt)
NII6584, NII6584_err, NII6584_EW, NII6584_EW_ERR = NII_vals(table, filt)
OII3726, OII3726_err, OII3729, OII3729_err = OII_vals(table, filt)


bad_data_filter =  bad_data_filt(OIII5007, OIII5007_err ,Hbeta, Halpha, Halpha_err, NII6584)

def apply_filter(data, filt):
    return data[filt]

def line_filter(line, line_err, ew, ew_err, filter):

    good_line = apply_filter(line, filter)
    good_line_err = apply_filter(line_err , filter)
    good_line_EW = apply_filter(ew , filter)
    good_line_EW_ERR = apply_filter(ew_err , filter)

    return good_line, good_line_err, good_line_EW, good_line_EW_ERR

OIII5007_line, OIII5007_line_err, OIII5007_line_EW, OIII5007_line_EW_ERR = line_filter(OIII5007, OIII5007_err, OIII5007_EW, OIII5007_EW_ERR, bad_data_filter)
Hbeta_line, Hbeta_line_err, Hbeta_line_EW, Hbeta_line_EW_ERR = line_filter(Hbeta, Hbeta_err, Hbeta_EW, Hbeta_EW_ERR, bad_data_filter)
Halpha_line, Halpha_line_err, Halpha_line_EW, Halpha_line_EW_ERR = line_filter(Halpha, Halpha_err, Halpha_EW, Halpha_EW_ERR, bad_data_filter)
NII6584_line, NII6584_line_err, NII6584_line_EW, NII6584_line_EW_ERR  = line_filter(NII6584, NII6584_err, NII6584_EW, NII6584_EW_ERR, bad_data_filter)

def low_SN(Halpha_line, Halpha_line_err):

    filt = np.ones(len(Halpha_line), dtype = bool)

    for i in range(len(Halpha_line)):
        if Halpha_line[i]/Halpha_line_err[i] < 3:
            filt[i] = False
    return filt

SN_filt = low_SN(Halpha_line, Halpha_line_err)
#plt.figure()
#plt.title('BPT Diagram')
#plt.plot(np.log10(NII_halpha), np.log10(OIII_Hbeta), '.', alpha = .4)
#plt.show()

OIII_Hbeta = OIII5007_line[SN_filt]/Hbeta_line[SN_filt]
NII_halpha = NII6584_line[SN_filt]/Halpha_line[SN_filt]

def SFR_galaxies():

    def SFR(x):
        return .61/(np.log10(x) - .05) + 1.3

    log_OIII_hbeta = np.log10(OIII_hbeta)
    log_NII_halpha = np.log10(NII_halpha)

    sfr_filt = np.ones(len(OIII_hbeta), dtype = bool)

    for i, val in enumerate(log_NII_halpha):
        y = SFR(val)
        if  log_OIII_hbeta[i] >= y:
            sfr_filt[i] = False

    return sfr_filt


def plotting_BPT(table):

    #making a filter to only get the ones that have the ratios there
    filt = np.zeros(len(table), dtype = bool)

    #for loop that checks to see if ratio is in table if it is assigns that index to True
    for i, val in enumerate(table['NII6583/Halpha']):
        if val > 0:
            filt[i] = True

    #makes a sub Tbale with only the objects with ratios
    new_table = table[filt]

    #getting the ratios of interest and object name
    OIII_HB = new_table['OIII5007/Hbeta']
    NII_HA = new_table['NII6583/Halpha']
    obj = new_table['ObjectID']

    #making an x array that will be used for plotting the cutoff points
    x = np.logspace(-2, 1.5, 1000)

    #got this from Kewely et al early 2000
    y = .61/(np.log10(x) - .47) + 1.19

    #got this from Kewely et al early 2013
    y_new = .61/(np.log10(x) + .08) + 1.1

    #got this from Kauffmann et al 2003 for AGN classification
    y_agn = .61/(np.log10(x) - .05) + 1.3

    #making a filter for the Kewely function getting rid of the line
    filt1 = np.ones(len(x), dtype = bool)

    #making a filter for the Kauffmann funciton to get rid of the line
    filt2 = np.ones(len(x), dtype = bool)

    filt3 = np.ones(len(x), dtype = bool)

    #getting indices where the log(x) are bigger than some number to apply the filtering
    ind = np.where(np.log10(x) > 0.4)
    ind2 = np.where(np.log10(x) > 0)
    ind3 = np.where(np.log10(x) > -.1)

    #assigning those values to false so that i can mask them out in the respective functions
    filt1[ind] = False
    filt2[ind2] = False
    filt3[ind3] = False

    #plotting the BPT diagram
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_title('BPT Diagram', fontsize = 30, weight = 'bold')
    ax1.set_xlabel(r'$log(\frac{[NII]6854}{H\alpha}) $ ', fontsize = 30)
    ax1.set_ylabel(r'$log(\frac{[OIII]5007}{H\beta}) $ ', fontsize = 30)
    ax1.set_xlim(-2, 1.5)
    ax1.set_ylim(-1.5, 2)

    sdss, = ax1.plot(np.log10(NII_halpha), np.log10(OIII_Hbeta), '.', alpha = .4, markersize = .07, label = 'SDSS Galaxies')
    line1, = ax1.plot(np.log10(x)[filt1], y[filt1], 'k--', label = 'Kewley 2003',  linewidth = 3)
    line2, = ax1.plot(np.log10(x)[filt3], y_new[filt3], 'y--', label = 'Kewley 2013',linewidth = 3)
    line3, = ax1.plot(np.log10(x)[filt2], y_agn[filt2], 'b--', label = 'Kauffmann', linewidth=3)


    
    for i in range(len(OIII_HB)):
        if i == 0:
            gal, = ax1.plot(np.log10(NII_HA[i]), np.log10(OIII_HB[i]), '*', color='red', markersize=15 ,label = 'This Work')
        else:
            ax1.plot(np.log10(NII_HA[i]), np.log10(OIII_HB[i]), '*', color='red', markersize=15)
            
    
    legend1 = ax1.legend(handles=[line1, line2, line3], prop={'size': 8}, loc='upper left', frameon=False)
    ax = ax1.add_artist(legend1)
    legend2 = ax1.legend(handles = [sdss], loc = 'lower right', frameon=False, prop={'size': 7}, markerscale=100, bbox_to_anchor=(0.95, 0.08, .05, .1))
    a = ax1.add_artist(legend2)
    ax1.legend(handles = [gal], loc = 'lower right', frameon=False, prop={'size': 8},)                     
    plt.show()

'''
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
'''
