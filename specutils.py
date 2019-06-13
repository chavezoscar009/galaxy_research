from astropy.modeling import models

def get_wcs_solution(hdr):
    """
    This function converts pixels in (x, y) to wavelengths in Angstroms
    
    Parameter
    ---------------
    hdr: the hdr of the file we want to get the pixel to wavelength solution

    Output
    ---------------
    This gives bck a polynomial p which we then plug in any puxel (x,y) and it gives us wavelength
    """
    

    pdict = {}
    
    
    for k,v in hdr.items():
        if 'P_' in k:
            key = k.split('P_')[1].lower()
            pdict[key] = v

    p = models.Polynomial2D(**pdict)
    return p

