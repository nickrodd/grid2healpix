###############################################################################
# grid2healpix.py
###############################################################################
#
# Function that converts cartesian grid into a healpix map of a specified nside
#
# INPUT:
#  - cmap: 2D grid map, entries must be per pixel, not per sr
#          rows denote position in b, columns position in l
#  - bin0bl: the b and l value of the [0,0] pixel of the map (in degrees). 
#            This should be the smallest b value and largest l (bottom left of 
#            the map). b and l are in degrees and measured b in [-90,90] and
#            l in [180,-180], such that (0,0) is the galactic center
#  - spacing: distance between the centre of adjacent pixels in b or l
#             should be a float in degrees
#  - nside: the nside of the output healpix map
#
###############################################################################

import numpy as np
import healpy as hp
from scipy import interpolate
from prop_mod import pm

def g2h(cmap, bin0bl, spacing, nside):
    
    # Determine array of l and b values associated with cmap
    # NB: using convention where l runs from larger (left) to smaller (right)
    b_cmap = bin0bl[0] + np.arange(np.shape(cmap)[0])*spacing
    l_cmap = bin0bl[1] - np.arange(np.shape(cmap)[1])*spacing

    # Check units have been input correctly, if not stop here
    assert ((l_cmap[0] <= 180.) & (l_cmap[-1] >= -180.) &
            (b_cmap[0] >= -90.) & (b_cmap[-1] <= 90.)), \
            "(l,b) coordinates have not been defined correctly"

    # Embed the map into a cartesian map that covers all of space
    # Filling empty pixels with zeros
    # Start out with a much larger grid than necessary then truncate
    nbgrid = np.around(180./spacing).astype(int)
    nlgrid = np.around(360./spacing).astype(int)
    
    bgrid = np.zeros(2*nbgrid)
    temp = bin0bl[0] - np.arange(nbgrid)*spacing
    bgrid[0:nbgrid] = temp[::-1]
    bgrid[nbgrid:2*nbgrid] = bin0bl[0] + (np.arange(nbgrid)+1.)*spacing
    bgrid = bgrid[np.where((bgrid <= 90.) & (bgrid >= -90.))]

    lgrid = np.zeros(2*nlgrid)
    temp = bin0bl[1] + np.arange(nlgrid)*spacing
    lgrid[0:nlgrid] = temp[::-1]
    lgrid[nlgrid:2*nlgrid] = bin0bl[1] - (np.arange(nlgrid)+1.)*spacing
    lgrid = lgrid[np.where((lgrid <= 180.) & (lgrid > -180.))]

    cmap_full = np.zeros(shape=(len(bgrid),len(lgrid)))
    for bi in range(len(bgrid)):
        for li in range(len(lgrid)):
            if ((lgrid[li] in l_cmap) & (bgrid[bi] in b_cmap)):
                b_loc = np.where(b_cmap == bgrid[bi])[0]
                l_loc = np.where(l_cmap == lgrid[li])[0]
                cmap_full[bi,li] = cmap[b_loc,l_loc]
                # Convert to /sr, more accurate for interpolation
                if (np.abs(bgrid[bi]) != 90.):
                    cmap_full[bi,li] /= (spacing*np.pi/180.)**2 \
                                        *np.cos(bgrid[bi]*np.pi/180.)
                else:
                    cmap_full[bi,li] /= (spacing*np.pi/180.)**2 \
                                *np.cos((bgrid[bi]-spacing/10)*np.pi/180.)

    # Extend the grid to account for periodicity in l
    lgrid_ext = 180. - (np.arange(len(lgrid)+2) - 0.5)*spacing
    cmap_ext = np.zeros(shape=(len(bgrid),len(lgrid)+2))
    cmap_ext[:,1:len(lgrid)+1] = cmap_full
    cmap_ext[:,0] = cmap_full[:,-1]
    cmap_ext[:,-1] = cmap_full[:,0]

    # Now use 2D interpolation onto a healpix map
    cmap_int = interpolate.interp2d(lgrid_ext,bgrid,cmap_ext)
    
    npix = hp.nside2npix(nside)
    hmap = np.zeros(npix)
    for pi in range(npix):
        theta, phi = hp.pix2ang(nside,pi)
        lval = pm(phi*180./np.pi+180.,360.)-180.
        bval = (np.pi/2 - theta)*180./np.pi
        hmap[pi] = cmap_int(lval, -bval)

    # Convert back to /pixel
    hmap *= hp.nside2pixarea(nside)

    return hmap
