import numpy as np
import pylab as plt
from astropy.io import fits

#this code opens up the galaxy info fits files we need this for the redshift
hdu = fits.open('galSpecInfo-dr8.fits')

#grabbing the table associated with the file
galaxy_info_table = hdu[1].data

#opening up the file with the spectrum lines
hdu1 = fits.open('galSpecLine-dr8.fits') 

#grabbing the table associated with the galaxy line file
galaxy_lines = hdu[1].data   

#grabbing the spec_objID
spec_objID = galaxy_info_table['SpecObjID']

#Grabbing the redshift
redshift = galaxy_info_table['Z']

#making a filter for galaxies within our specific redhsift regime
z_filt = np.array([True if x > .025 and x < .06 else False for x in z]) 

#applying the filter to spec_objID and redshift
spec_objID_valid = spec_objID[z_filt]
valid_z = redshift[z_filt]

#################
#MAKING THE BPT DIAGRAM FROM SDSS GALAXIES
#################

NII_6584 = table1['NII_6584_FLUX'][z_filt]  
Halpha = table1['H_ALPHA_FLUX'][z_filt] 

ratio_NII_HA = []  

OIII_5007 = table1['OIII_5007_FLUX'][z_filt] 
Hbeta = table1['H_BETA_FLUX'][z_filt] 

ratio_NII_HA = [] 
ratio_OIII_HB = [] 

for i in range(len(OIII_5007)): 
     
    if Halpha[i] == 0 or Hbeta[i] == 0: 
        continue 
    
    elif NII_6584[i] < 0 or Halpha[i] < 0 or OIII_5007[i] < 0 or Hbeta[i] < 0: 
        continue     
    
    else: 
        ratio_NII_HA.append(NII_6584[i]/Halpha[i]) 
        ratio_OIII_HB.append(OIII_5007[i]/Hbeta[i])
        
plt.figure(figsize = (10,10))
plt.loglog(ratio_NII_HA, ratio_OIII_HB)
plt.show()


