import numpy as np
from astropy.io import fits
import astropy.units as u
import astropy.constants as cons
import astropy.table as tabl
# import contextlib
# import io

from pysme.sme import SME_Structure
from pysme.iliffe_vector import Iliffe_vector
from pysme.abund import Abund
from pysme.linelist.vald import ValdFile
from pysme.solve import solve
from pysme.synthesize import synthesize_spectrum

from importlib.resources import files

vald = ValdFile(files(__package__) / 'linelist_solar_0.001_1.1_400-800nm')
vald = vald[vald.depth>=0.001]
vald.element = np.vstack(np.char.split(vald.species)).T[0]

def create_ranges(centers, halfspan=5, join=True):
    'Output ranges are sorted'
    'If join is True, merges overlapping ranges'
    centers = np.sort(np.reshape(centers, -1))
    hs = np.reshape(halfspan, -1)
    ws = centers - hs
    we = centers + hs
    wran =  np.array([ws,we]).T
    if join:
        keepflag = np.ones(len(wran),bool)
        for i in range(1,centers.size):
            if wran[i][0] <= wran[i-1][1]:
                wran[i][0] = wran[i-1][0]
                keepflag[i-1] = False
        wran = wran[keepflag]
    return wran

def combine_ranges(*wran):
    wran = np.vstack(wran)
    wran = wran[np.argsort(wran[:,0])]
    keepflag = np.ones(len(wran),bool)
    for i in range(1,len(wran)):
        if wran[i-1][1] >= wran[i][0]:
            wran[i][0] = wran[i-1][0]
            keepflag[i-1] = False
    return wran[keepflag]    
    
def inranges(values, ranges):
    'ranges should be sorted and non-overlapping.'
    ranges_unraveled = np.asarray(ranges).ravel()
    if len(ranges_unraveled) % 2 != 0:
        raise ValueError('Ranges should be pairs of [start, end].')
    if not np.all(ranges_unraveled[:-1] <= ranges_unraveled[1:]):
        raise ValueError('Ranges should be sorted and non-overlapping.')
    return (np.searchsorted(a=ranges_unraveled, v=values, side='right') % 2).astype(bool)

def findvalleys(arr):
    return np.where(np.diff(np.sign(np.diff(arr))) == +2)[0] + 1

def calc_galah_vmic(teff, logg):
    if teff<=5500 and logg>4.2:
        vmic = 1.1 + 1.6e-4*(teff-5500)
    else:
        vmic = 1.1 + 1e-4*(teff-5500) + 4e-7*(teff-5500)**2
    return vmic
   

# def initsme(wran=None, res=0, nlteelems=['H','Mg','Fe','Ca','Ti','Si','Ba']):
#     sme = SME_Structure()
#     if res>0: sme.ipres = res; sme.iptype = 'gauss'
#     # sme.abund = Abund.solar()
#     for elem in nlteelems:
#         sme.nlte.set_nlte(elem, f'nlte_{elem}_pysme.grd')
#     sme.wran = wran
#     return sme

def initsme(linelist=vald, res=None, teff=5777, logg=4.44, monh=0.0, vsini=0, vmic=1.1, vmac=0, nlteelems=['H','Mg','Fe','Ca','Ti','Si','Ba']):
    sme = SME_Structure()
    sme.ipres = 0 if res is None else res
    sme.iptype = 'gauss'
    sme.normalize_by_continuum = True
    for elem in nlteelems:
        sme.nlte.set_nlte(elem, f'nlte_{elem}_pysme.grd')
    sme.teff = teff
    sme.logg = logg
    sme.monh = monh
    sme.vsini = vsini
    sme.vmic = vmic
    sme.vmac = vmac
    sme.linelist = linelist
    sme.abund['Fe'] = 7.38 # A(Fe) correction from GALAH DR3
    return sme
