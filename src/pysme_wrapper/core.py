import numpy as np
from tqdm.auto import tqdm
from copy import deepcopy
from astropy.stats import sigma_clipped_stats, sigma_clip

from pysme.sme import SME_Structure
from pysme.iliffe_vector import Iliffe_vector
from pysme.solve import solve
from pysme.synthesize import synthesize_spectrum

from pysme.synthesize import Synthesizer

import emcee
from scipy.fft import rfft, irfft, rfftfreq, next_fast_len
from scipy.sparse import csr_matrix
from scipy.special import erf
from scipy.interpolate import BSpline
try:
    from multiprocessing import Pool, get_context, get_all_start_methods
except ImportError:
    pass # Multiprocessing not available, will run in single process mode.

from .utils import *
from PyAstronomy import pyasl
from telfit import Modeler

from scipy.interpolate import make_smoothing_spline, make_interp_spline

# import logging
# logging.basicConfig(level=logging.ERROR)

def _objarray(iterable):
    if isinstance(iterable, float) or isinstance(iterable, int):
        return np.array([iterable], dtype=object)
    arr = np.empty(len(iterable), dtype=object)
    for i, item in enumerate(iterable): # Item-wise, so equal-length arrays aren't broadcast into a 2D array
        arr[i] = item
    return arr

class SMEwrapper(SME_Structure):
    def __init__(self, fulllinelist = vald, teff=5777, logg=4.44, monh=0.0, vsini=0, vmic=1.1, vmac=0):
        '''
        Input parameters are self-explanatory.
        Option to input wave_ranges and resolution(s) is given here for the purpose of synthesizing spectra, not applicable for fitting.
        Iron abundance zero point is set to A(Fe)=7.38 by default, following the GALAH DR3 correction. You can change this by setting `self.abund['Fe']` to your preferred value after initializing the object.
        NOTE: All the methods of SME_Structure _should_ also work with this class, but I haven't stress-tested them, so use those with caution and report any issues you find. 

        Parameters
        ----------
        fulllinelist : ValdFile or Linelist object, default provided
            Linelist object. See PySME documentation on how to get/build it. A Default linelist is provided encompassing all lines an FGK-type optical spectrum is typically likely to need, spanning 4000-8000 Å.

        The others are self-explanatory stellar and broadening parameters.
        '''
        # Parent attributes
        super().__init__()
        self.iptype = 'gauss'
        self.normalize_by_continuum = True
        self.vrad_flag = "none"
        self.cscale_flag = 'none'
        self.teff = teff
        self.logg = logg
        self.monh = monh
        self.vsini = vsini
        self.vmic = vmic
        self.vmac = vmac
        self.fulllinelist = fulllinelist
        self.abund['Fe'] = 7.38 # A(Fe) correction from GALAH DR3

        # Wrapper attributes
        # self.RV = None
        self.obswave = None
        self.obsflux = None
        self.obserr = None
        self.obsres = None
        self.__WRAN = None

    #region Attribute setters and getters
    @property
    def WRAN(self):
        return self.__WRAN
    @WRAN.setter
    def WRAN(self, wave_ranges):
        if wave_ranges is None:
            self.__WRAN = None
            return # Used to quit the function
        try:
            wave_ranges = np.array(wave_ranges).reshape(-1,2)
        except:
            raise ValueError('`wave_ranges` needs to be by Nx2 array-like. The `create_ranges` function may help.')
        if (np.diff(wave_ranges.ravel()) <= 0).any():
            raise ValueError('Wave ranges need to be non-overlapping and sorted in ascending order. The `combine_ranges` function will do this for you.')
        self.__WRAN = wave_ranges
    @property
    def NSEG(self):
        return len(self.WRAN) if self.WRAN is not None else 0

    def __getattribute__(self, name):
        if name == "mask" or name == "uncs":
            if self.wave is None or self.spec is None:
                return None
            elif (self.wave.shape[1] != self.spec.shape[1]).any():
                raise ValueError('The wave and spec attributes must have the same shape. Please check your input.')
            else:
                if name == "mask": return Iliffe_vector([np.isfinite(arr) for arr in self.spec])
                else: return Iliffe_vector([np.ones(size, float) for size in self.spec.shape[1]])
        elif name == "synth" or name == "cont":
            if self.wave is None:
                return None
            var = super().__getattribute__(name)
            if var is None or (var.shape[0] != self.wave.shape[0]) or (var.shape[1] != self.wave.shape[1]).any() or (len(self.linelist)==0):
                return None
            return var
        elif name == "central_depth" or name == "line_range":
            var = super().__getattribute__(name)
            if self.wave is None or var is None or len(self.linelist)==0:
                return None
            if len(var) != len(self.wave) or len(var[0])!=len(self.linelist):
                return None
            return var
        return super().__getattribute__(name)
    #endregion

    # region input
    def input_fit_wave_ranges(self, wave_ranges):
        'Alias to doing `self.WRAN = wave_ranges`'
        'The wave ranges should generally be not much larger than a few Å, i.e. each only encompassing one line or a few very closeby lines.'
        self.WRAN = wave_ranges 

    def input_observed_spectrum(self, wave: np.ndarray, flux: np.ndarray, err: np.ndarray | None = None, res: np.ndarray | None = None):
        '''
        Input 1-D arrays for observed wavelength, normalised flux, relative flux error and resolution solution (optional). Arrays must have same length.\\
        IMPORTANT: Resolution solution should match the other arrays. Option to input single resolution values for each segment is given in the function `make_fit_segments`.\\
        Unless you really trust the instrument supplied error estimates on the flux values, I recommend leaving `err` as None and setting ERR='fit' in `make_fit_segments` to calculate the (rms) error empirically in each fit segment.
        '''
        if len(wave) != len(flux): raise ValueError("All input arrays must have the same length")
        if (res is not None) and (len(res)!=len(wave)): raise ValueError('res array doesn\'t match spectrum length.')
        if err is not None and len(err)!=len(wave): raise ValueError('err array doesn\'t match spectrum length.')
        self.obswave = np.array(wave)
        self.obsflux = np.array(flux)
        self.obserr = np.array(err) if err is not None else None
        self.obsres = np.array(res) if res is not None else None
        # if self.obsres is not None and len(self.obsres) != len(self.obswave):
        #     self.obsres = None
        #     print('WARNING: Resolution array length does not match spectrum length. Resolution solution will be set to None. Single value resolutions should be input in `make_fit_segments` instead.')
        self.obstelluric = None

    def get_telluric_transmission(self, lat, alt, temperature, pressure, humidity, airmass, resolution, inair=True, nan_thresh=0.98, return_transmission=False):
        '''
        See https://telfit.readthedocs.io/en/latest/MakeModel.html#:~:text=None-,MakeModel

        parameters:
        ---
        wave: Wavelength array in Å
        lat: latitude (deg)
        alt: altitude in meters above sea level
        temperature: Centigrade
        pressure: hPa
        humidity: %
        airmass: sec(zenith angle)
        resolution (int): Approximate resolution (=lam/dlam) over the wavelength range
        inair (bool): in air (True) or vacuum (False) wavelengths

        nan_thresh: float or None, default: 0.98
            The threshold below which the telluric transmission is considered too low and the corresponding observed flux values are set to nan.
            If None or 0, no change to `self.obsflux`.

        These parameters are generally in the primary header of the fits files containng the spectra.
        '''
        wavestart = self.obswave[0]/10 -1
        waveend = self.obswave[-1]/10 +1       
        modeler = Modeler(print_lblrtm_output=False)
        model = modeler.MakeModel(
            vac2air=inair,
            pressure=pressure,
            temperature= 273.13 +temperature,
            humidity=humidity,
            angle=np.arccos(1/airmass)*180/np.pi,
            lat=lat,
            alt=alt/1000,
            lowfreq=1e7/waveend,
            highfreq=1e7/wavestart,
            resolution=resolution,
            wavegrid=self.obswave/10
        )
        self.obstelluric = model.y
        if nan_thresh is not None:
            self.obsflux[self.obstelluric<nan_thresh] = np.nan
        if return_transmission: return model.y

    # endregion

    # region fitting segments
    def make_fit_segments(self, wave_ranges=None, RES=None, RV='fit', CS='fit', ERR='fit', linelist=None, make_quality_cuts=True, return_copy=False, fit_RV_kwargs={}, err_cs_kwargs={}):
        '''
        Parameters
        ----
        wave_ranges : Nx2 array-like, optional
            start and end values for each wavelength segment. They should be sorted and non-overlapping. Use `create_ranges` and/or `combine_ranges` to make'em easily.
            Not needed if `self.WRAN` has already been set, and will overwrite `self.WRAN`.
        RES : False or int or array-like, optional 
            Spectral resolution for each segment may be input here, if not given with the input spectrum.
            If False, no resolution is set. If int, assumes same resolution for all segments. If array-like, must have same length as wave_ranges and will set the resolution for each segment accordingly.
            If not None, overrides resolutions input at any prior stage.
        RV : str or float or array-like or None, default: 'fit'
            If 'fit', fits for RV in each segment. If a float or array-like, applies the given RV shift(s). If None, sets RV to zero for all segments.
        CS : 'fit' or float or array-like or None, default: 'fit'
            If 'fit', fits for continuum scaling in each segment. If a float or array-like, uses the given scaling factor(s). If None, sets CS to unity for all segments.
        ERR : 'fit' or 'propagate' or array-like or None, default: 'fit'
            Relative error. If 'fit', calculates relative rms error of the continuum in each segment. If 'propagate', propagates the errors from the input spectrum. If array-like, uses the given errors.
        linelist : str or list, optional
            If you want to pass a custom linelist. Else inherits from `self.fulllinelist`. 
        make_quality_cuts : bool, default: True
            Whether to make default quality cuts based on the fitted RV, ERR and CS values.
            If True, segments with outlying RV values (more than 2σ deviation and more than 3 km/s from the mean RV), very high ERR values (more than 3σ above the mean ERR) or outlying mean-CS values (<0.8 or >1.2) will be removed. 
        return_copy : bool, default: False
            Whether to return a copy of the SMEwrapper object with the segments set instead of setting them to the current object.

        Sets the following attributes:
        ---
        WRAN : Nx2 array
            The wave ranges for each segment.
        WAVE : N arrays
            The observed wavelength grid of each segment, RV shifted to the rest frame of the star.
        FLUX : N arrays
            The observed flux values corresponding to the wavelength grid.
        RV : Nx1 array
            The radial velocity values of each segment from Earth.
        ERR : Nx1 array or N arrays or None
            The error values associated with `self.FLUX`.
        RES : Nx1 array or None
            The intrumental resolution of each segment.
        CS : Nx1 array
            The continuum scaling factor applied to each segment.
        CSEG : N arrays
            The indices into `self.obswave` and `self.obsflux` to build the each of the WAVE and FLUX arrays. 
        '''
        if not hasattr(self, 'obswave') or not hasattr(self, 'obsflux') or self.obswave is None or self.obsflux is None:
            raise AttributeError('Observed spectrum not set. Please use the `input_observed_spectrum` method to input the observed spectrum before making fit segments.')
        if return_copy: obj = deepcopy(self)
        else: obj = self
        if wave_ranges is not None:
            obj.WRAN = wave_ranges
        elif obj.WRAN is None:
            raise ValueError('No wave ranges supplied for fitting.')
        obj.wran = None; obj.wave=None; obj.synth=None
        approx_resolution = np.mean(RES) if RES is not None else None

        # RV
        if isinstance(RV, str) and RV=='fit':
            RV = obj.fit_RV(approx_resolution=approx_resolution,linelist=linelist, **fit_RV_kwargs)
        elif RV is None:
            RV = np.zeros(obj.NSEG)
        else:
            RV = np.array(RV).reshape(-1)
            if len(RV)==1: RV = np.full(obj.NSEG, RV)
        obj.RV = RV

        # Build primary arrays
        obj.CSEG = _objarray([inranges(obj.obswave*(1-obj.RV[i]/299792.5), ran).nonzero()[0] for i,ran in enumerate(obj.WRAN)])
        obj.WAVE = _objarray([obj.obswave[c]*(1-obj.RV[i]/299792.5) for i,c in enumerate(obj.CSEG)])
        obj.FLUX = _objarray([obj.obsflux[c] for c in obj.CSEG])

        # CS and ERR
        if (isinstance(CS, str) and CS=='fit') or (isinstance(ERR, str) and ERR=='fit'):
            _ERR, _CS = obj.get_error_and_cscale(obj.RV, approx_resolution, linelist=linelist, **err_cs_kwargs)
        #---
        if CS is None: CS = np.ones(obj.NSEG)
        elif isinstance(CS, str) and CS=='fit': CS = _CS
        elif isinstance(CS, (int, float, np.number)): CS = np.full(obj.NSEG, CS)
        elif len(CS)==obj.NSEG: CS = _objarray(CS)
        else: raise ValueError('Length of input CS array does not match the number of segments.')
        obj.CS = CS
        #---
        if ERR is None: pass
        elif isinstance(ERR, str):
            if ERR=='fit': ERR = _ERR
            elif ERR=='propagate': ERR = _objarray([obj.obserr[c] for c in obj.CSEG])
            elif ERR=='none': ERR = None
            else: raise ValueError('Invalid string input for ERR. Should be one of "fit", "propagate" or "none".')
        elif len(ERR)==obj.NSEG: ERR = _objarray(ERR)
        else: raise ValueError('Length of input ERR array does not match the number of segments.')
        obj.ERR = ERR

        if RES is False:
            obj.RES = None
        else:
            if RES is not None:
                obj.RES = np.array(RES).reshape(-1)
            elif obj.obsres is not None:
                obj.RES = np.array([obj.obsres[c].mean() for c in obj.CSEG])
            else:
                # At this point if RES is None, then raise error
                raise TypeError('No resolution(s) input. If intentional, pass `RES=False` explicitly')
            if len(obj.RES)!=obj.NSEG:
                if len(obj.RES)==1:
                    obj.RES = np.tile(obj.RES, obj.NSEG)
                    print('WARNING: Only a single resolution is given for all segments. This is not recommended. Proceeding')
                else: raise ValueError('Input resolution array does not match the number of segments.')

        if make_quality_cuts:
            c1 = sigma_clip(obj.RV, sigma=2, maxiters=3).mask | (np.abs(obj.RV-obj.RV.mean()) > 2)
            c2 = sigma_clip(obj.ERR, sigma_upper=3, sigma_lower=15, maxiters=3).mask if obj.ERR.dtype==float else np.zeros_like(c1, dtype=bool)
            meanCS = np.hstack([spl(obj.WAVE[i].mean()) for i,spl in enumerate(obj.CS)]) if obj.CS.dtype==object else np.array([arr.mean() for arr in obj.CS])
            c3 = (meanCS<0.8) | (meanCS>1.2)
            icut = (c1 | c2 | c3).nonzero()[0]
            if len(icut):
                print(f'The median fitted RV is {np.median(obj.RV[~(c1|c2|c3)]):.2f} km/s.')
                if obj.ERR.dtype==float:
                    print(f'The median ERR is {np.nanmedian(obj.ERR[~(c1|c2|c3)]):.2f}.')
                print(f'{len(icut)} segments will be removed due to poor fits to the RV, ERR and/or CS values. See function documentation for details.')
                # Formatted to display aligned neatly in fixed-width font
                print(f'{"iseg":<5} {"WAVE_RANGE":<20} {"RV":<8} {"ERR":<8} {"CS":<8} reason')
                for iseg in icut:
                    reason = f'{"RV " if c1[iseg] else "   "}{"ERR " if c2[iseg] else "    "}{"CS" if c3[iseg] else "  "}'
                    print(f'{iseg:<5} {obj.WRAN[iseg].round(2)!s:<20} {obj.RV[iseg]:<8.2f} {obj.ERR[iseg].mean().round(2)!s:<8} {meanCS[iseg]:<8.2f} {reason}')
            obj.delete_fit_segments(icut)
        if return_copy:
            return obj
        
    def save_fit_segments(self, filename):
        '''
        Saves the fit segments and their associated attributes to a .npz file. The saved attributes are WRAN, CSEG, WAVE, FLUX, RV, CS, ERR and RES.
        '''
        np.savez(filename, WRAN=self.WRAN, CSEG=self.CSEG, WAVE=self.WAVE, FLUX=self.FLUX, RV=self.RV, CS=self.CS, ERR=self.ERR, RES=self.RES)

    def load_fit_segments(self, filename):
        '''
        Loads the fit segments and their associated attributes from a .npz file. The file should contain the attributes WRAN, CSEG, WAVE, FLUX, RV, CS, ERR and RES.
        Overwrites any existing fit segments.
        '''
        data = np.load(filename, allow_pickle=True)
        self.WRAN = data['WRAN']
        self.CSEG = data['CSEG']
        self.WAVE = data['WAVE']
        self.FLUX = data['FLUX']
        self.RV = data['RV']
        self.CS = data['CS']
        self.ERR = data['ERR']; self.ERR = self.ERR if self.ERR.shape != () else self.ERR.item()
        self.RES = data['RES']; self.RES = self.RES if self.RES.shape != () else self.RES.item()

    def delete_fit_segments(self, indices):
        'Deletes the fit segments with the given indices and the associated attributes.'
        keepidx = np.setdiff1d(range(self.NSEG), indices)
        self.WRAN = self.WRAN[keepidx]
        self.CSEG = self.CSEG[keepidx]
        self.WAVE = self.WAVE[keepidx]
        self.FLUX = self.FLUX[keepidx]
        self.RV = self.RV[keepidx]
        self.CS = self.CS[keepidx]
        self.ERR = self.ERR[keepidx] if self.ERR is not None else None
        self.RES = self.RES[keepidx] if self.RES is not None else None

    def add_fit_segments(self, wave_ranges, RES=None, RV=None, CS=None, ERR=None, return_copy=False):
        '''
        Makes and adds segments corresponding to the given wave_ranges. Does not have the option to fit RV, CS or ERR for the new segments, these must be computed a-priori and directly input.
        The new segments are inserted in a sorted manner. If they overlap with existing segments, it shouldn't cause problems as long as it's just the edges overlapping. Otherwise, any problems are on you.
        RES and ERR must be None if and only if they are also None for the existing segments. If RV and CS are None, they will be set to zero and unity respectively for the new segments.
        ''' 
        if (self.RES is None and RES is not None) or (self.RES is not None and RES is None) or (self.ERR is None and ERR is not None) or (self.ERR is not None and ERR is None):
            raise ValueError('RES and ERR must be None if and only if they are also None for the existing segments.')
        if return_copy: obj = deepcopy(self)
        else: obj = self

        wave_ranges = np.array(wave_ranges).reshape(-1,2)
        RV = np.zeros(len(wave_ranges)) if RV is None else np.array(RV).reshape(-1)
        if len(RV)==1: RV = np.tile(RV, len(wave_ranges))
        CS = np.ones(len(wave_ranges)) if CS is None else np.array(CS).reshape(-1)
        if len(CS)==1: CS = np.tile(CS, len(wave_ranges))
        RES = None if RES is None else np.array(RES).reshape(-1)
        if len(RES)==1: RES = np.tile(RES, len(wave_ranges))
        if ERR is not None and len(ERR)!=len(wave_ranges): raise ValueError('Input ERR array does not match the number of new segments.')

        CSEG = _objarray([inranges(obj.obswave*(1-RV[i]/299792.5), ran).nonzero()[0] for i,ran in enumerate(wave_ranges)])
        WAVE = _objarray([obj.obswave[c]*(1-RV[i]/299792.5) for i,c in enumerate(CSEG)])
        FLUX = _objarray([obj.obsflux[c] for c in CSEG])

        insidx = np.searchsorted(obj.WRAN[:,0], wave_ranges[:,0])
        obj.WRAN = np.insert(obj.WRAN, insidx, wave_ranges, axis=0)
        obj.CSEG = np.insert(obj.CSEG, insidx, CSEG, axis=0)
        obj.WAVE = np.insert(obj.WAVE, insidx, WAVE, axis=0)
        obj.FLUX = np.insert(obj.FLUX, insidx, FLUX, axis=0)
        obj.RV = np.insert(obj.RV, insidx, RV)
        obj.CS = np.insert(obj.CS, insidx, CS)
        if obj.ERR is not None: obj.ERR = np.insert(obj.ERR, insidx, ERR)
        if obj.RES is not None: obj.RES = np.insert(obj.RES, insidx, RES)
        if return_copy:
            return obj
        
    def user_defined_segments(self, WAVE, FLUX, RES=None, RV=None, CS=None, ERR=None):
        '''
        Directly set the fit segments and their associated attributes. Use with caution, making sure the attributes are consistent with each other and with the input spectrum. 
        This is for users who already have completely processed fit segments and just want to input them directly. Helpful for simulations.

        Parameters
        ----------  
        WAVE : array-like of arrays
            The observed wavelength grid of each segment, RV shifted to the rest frame of the star. 
        FLUX : array-like of arrays
            The observed flux values corresponding to the wavelength grid.
        RES : array-like or float or None, optional
            The intrumental resolution of each segment. If a single value is provided, it will be applied to all segments. If none, no resolution is set for any segment.
        RV : array-like or float or None, optional
            The radial velocity values of each segment from Earth. If a single value is provided, it will be applied to all segments. If None, sets RV to zero for all segments.
        CS : array-like of floats or BSpline objects or None, optional
            The continuum scaling factor applied to each segment. If floats, should have the correct shape: 1 or len(seg) for each segment. If None, sets CS to unity for all segments.
        ERR : array-like or float or None, optional
            The error values associated with `self.FLUX`. If a single value is provided, it will be applied to all segments. If None, no error values are set for any segment.
        '''
        if len(WAVE) != len(FLUX):
            raise ValueError('Length of WAVE and FLUX arrays must match.')
        try:
            if not np.all([len(WAVE[i])==len(FLUX[i]) for i in range(len(WAVE))]):
                raise ValueError('Length of WAVE and FLUX arrays must match.')
        except TypeError:
            raise ValueError('WAVE and FLUX must be array-like of arrays.')
        self.CSEG = None
        self.WAVE = _objarray(WAVE)
        self.FLUX = _objarray(FLUX)
        self.WRAN = np.array([[arr.min(), arr.max()] for arr in self.WAVE])
        self.RV = _objarray(RV)*np.ones(self.NSEG) if RV is not None else np.zeros(self.NSEG)
        self.CS = _objarray(CS) if CS is not None else np.ones(self.NSEG) #
        self.ERR = _objarray(ERR)*np.ones(self.NSEG) if ERR is not None else None
        self.RES = _objarray(RES)*np.ones(self.NSEG) if RES is not None else None
        
        
    def fit_RV(self, approx_resolution=None, wave_locations=None, window_size=40, linelist=None, segments='all', rot_broad_off=True, return_arrays=False):
        '''
        If you have preferred stellar parameters for this, input them in the SMEwrapper object before calling this function. 
        Remember to check if the fitted RVs for each segment are close to each other (within a few km/s at most). Some segments may give outlying RV values due to various reasons (e.g., low SNR, few lines, etc.) and you should remove or repeat the fit for them. 

        Parameters
        ----------
        approx_resolution : int, optional
            The approximate resolution of the spectrum. Input if `obsres` is not set and you feel brodening the synthetic spectrum to the instrumental resolution will improve the RV estimate.
            Will override (but not overwrite) `obsres`. 

        wave_locations : 1D or Nx2 array-like, optional
            If 1D, should be the central wavelengths of the segments. If Nx2, I take the mean of each pair as the central wavelength. 
            Overrides `self.WRAN` if given.

        window_size : float or array-like (in Å), default: 40
            Window size in Å for the cross-correlation. Should be large enough to encompass multiple strong-ish lines but not too large to be affected by large scale variations like those due to different echelle orders.
            If array-like, must have same length as number of segments and will be applied to each segment accordingly.
        
        linelist : ValdFile or Linelist, optional
            In case you want to pass a custom linelist for this, perhaps a more limited one focused on specific lines. 
            Otherwise uses lines in the window with nominal depth > 0.1.

        segments : 'all' or array-like of ints, default: 'all'
            The segments to fit. If 'all', fits all segments. If array-like, should be a list of segment indices to fit.

        rot_broad_off : bool, default: True
            Whether to set vsini to zero for the RV fit.
            This can help to get sharper lines in the synthetic spectrum, reducing the failure modes of the fit, though it may backfire for highly broadened spectra. 

        return_arrays : bool, default: False
            Whether to return the wavelength, flux and synthetic arrays used for the fit along with the RV values. For plotting and debugging purposes.

        Returns
        -------
        RV : array
            Array of fitted RV values for each selected segment.
        '''
        if wave_locations is not None:
            wave_locations = np.array(wave_locations)
            if wave_locations.ndim == 2: wave_locations = wave_locations.mean(axis=1)
        else: 
            wave_locations = self.WRAN.mean(1)
        wranrv = create_ranges(wave_locations, halfspan=np.array(window_size)/2, join=False)
        if linelist is not None:
            self.linelist = linelist
        else: 
            self.linelist = self.fulllinelist
        self.linelist = self.linelist[inranges(self.linelist.wlcent, combine_ranges(wranrv)) & (self.linelist.depth>0.1)]
        if approx_resolution is not None: self.ipres = approx_resolution
        elif self.obsres is not None: self.ipres = self.obsres.mean()
        else: self.ipres = 0
        
        vsini_cached = self.vsini
        if rot_broad_off:
            # zero-out vsini to get sharper lines for the RV fit - it tends not to work very well if the lines are too broad.
            self.vsini = 0 

        if isinstance(segments,str) and segments=='all':
            segments = range(len(wranrv))
        cseg = [inranges(self.obswave, ran)&(np.isfinite(self.obsflux)) for ran in wranrv[segments]]
        self.wave = [self.obswave[c] for c in cseg]
        self.spec = [self.obsflux[c] for c in cseg]
        self.vrad_flag = "each"

        # with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        self = solve(self, ['vrad'])
        RV = np.array(self.vrad)

        if return_arrays:
            WAVE = list(self.wave)
            FLUX = list(self.spec)
            SYN = list(self.synth)

        # # Cleanup
        self.vsini = vsini_cached
        self.vrad = 0
        self.vrad_flag = "none"
        self.ipres = 0
        self.linelist = None
        self.wave = None
        self.spec = None
        self.wran = None
        self.nlte.grid_data = {}
        
        if return_arrays:
            return RV, WAVE, FLUX, SYN
        return RV

    def get_error_and_cscale(self, RV=None, approx_resolution=None, wave_ranges=None, window_size=60, continuum_threshold=0.98, cscale_mode='spline', smoothing_lambda=10, err_quantile_window=[0.3,0.7], linelist=None, segments='all', debug_mode=False):
        '''
        Masks points affected by absorption lines (and tellurics if given) and fits a smoothing spline to the (unmasked) continuum in each segment.
        Also uses the identified continuum points to calculate the relative error (standard deviation) in each wave range (aka segment).
        If you have preferred stellar parameters for this, input them in the SMEwrapper object before calling this function. 

        Parameters
        ----------
        RV : float or array-like, default: 0
            RV shift(s) in km/s to apply to the segments. Must have same length as `self.WRAN` or `wave_ranges` if array-like.

        approx_resolution : int, optional
            The approximate resolution of the spectrum. Input if `obsres` is not set. Is important to compute the line mask (unbroadened lines have a smaller footprint and the mask will not exclude the lines completely).
            Will override (but not overwrite) `obsres`. 

        wave_ranges : Nx2 array-like, optional
            Start and end values for wavelength segment you want to compute the error and continuum scaling for.
            Will override, but not overwrite, `self.WRAN`. If not given, uses `self.WRAN`. If that is also None, throws an error.

        window_size : float or array-like (in Å), default: 60
            Window size in Å for the cross-correlation. The windows should be considerably larger than the lengths of the wave range as the splines will have large edge effects and you want to avoid that. You also want a decent chuck to compute the error.
            The default 60 Å was set based on trial-and-error and should probably work for segments up to 20 Å wide. However, generally speaking your segments should be less than 5 Å wide, as otherwise you're probably doing something wrong.
            If array-like, must have same length as number of segments and will be applied to each segment accordingly.

        continuum_threshold : float, default: 0.98
            The threshold above which a point in the (flat & continuum-normalized) synthetic spectrum is considered to in the continuum.
            Points below the threshold are masked from the continuum fit.

        cscale_mode : str', default: 'spline'
            If 'spline': fits and returns a smoothing spline to the continuum in the stellar rest frame using the given RV.
            If 'segment_mean': In this case the returned `CS` is simply the mean of the observed flux divided by the model flux within each wave range in `self.WRAN` (i.e. directly within each segment).
            If 'window_quantile_mean': In this case the returned `CS` is simply the mean of the observed continuum flux divided by the model continuum flux within the qunatile window for each segment.
            If 'none' : returned `CS` is unity for all segments.
        
        smoothing_lambda : float, default: 10
            The `lam` parameter in `scipy.interpolate.make_smoothing_spline`.
            I set the default to 10 for the FEROS resolution (48000) and the default 60 Å window size. If your values differ sigificantly you may want to change this. Basically you want to make sure the CS output is neither too wiggly nor just a flat line.
        
        err_window_quantiles : [min_quantile, max_quantile], default: [0.3,0.7]
            The standard deviation is calculated from the continuum regions inside the given quantile range of the window_size for each segment. This is to avoid edge effects in the error estimation.
            
        linelist : ValdFile or Linelist, optional
            In case you want to pass a custom linelist - not recommended here, you want to include all lines ideally.

        segments : 'all' or array-like of ints, default: 'all'
            The segments to fit. If 'all', fits all segments. If array-like, should be a list of segment indices to fit.

        debug_mode : bool, default: False
            If True, resturns a dictionary containing relevant arrays used for the fit for each segment. For debugging purposes.

        Returns
        -------
        ERR : array
            Array of relative errors of each selected segment.
        CS : array of `scipy.interpolate.BSpline` objects or floats
             The continuum scaling factor for each segment *in the stellar rest frame*. The format depends on the `cscale_mode` argument.
        if debug_mode is True:
            A dictionary with keys 'WAVE', 'FLUX', 'SYN', 'CCONT', 'CERR' containing the wavelength, flux, synthetic spectrum, continuum mask and error mask arrays used for the fit for each segment.
        '''
        WRAN_cached = self.WRAN
        if wave_ranges is not None:
            self.WRAN = wave_ranges
        elif self.WRAN is None:
            raise ValueError('No wave ranges supplied for fitting.')
        wrancont = create_ranges(self.WRAN.mean(1), halfspan=np.array(window_size)/2, join=False)
        wrancontpad = create_ranges(self.WRAN.mean(1), halfspan=np.array(window_size)/2+1, join=False)
        # print(wrancont)
        if linelist is not None:
            self.linelist = linelist
        else: 
            self.linelist = self.fulllinelist[inranges(self.fulllinelist.wlcent, combine_ranges(wrancontpad))]
        if approx_resolution is not None: self.ipres = approx_resolution
        elif self.obsres is not None: self.ipres = self.obsres.mean()
        elif self.ipres is None: self.ipres = 0

        if RV is None:
            if self.RV is not None and len(self.RV)==self.NSEG:
                RV = self.RV
            else: RV = np.zeros(self.NSEG)
        else: 
            RV = np.tile(RV, self.NSEG) if np.isscalar(RV) else np.array(RV)
        delta_lambda = np.mean(np.diff(self.obswave))/5
        self.wave = np.arange(wrancontpad[0][0], wrancontpad[-1][-1], delta_lambda)
        self = synthesize_spectrum(self)

        CS = np.empty(len(wrancont),'O')
        ERR = np.zeros(len(wrancont))
        if debug_mode:
            CCONT = np.empty(len(wrancont),'O')
            CERR = np.empty(len(wrancont),'O')
            WAVE = np.empty(len(wrancont),'O')
            SYN = np.empty(len(wrancont),'O')
            FLUX = np.empty(len(wrancont),'O')

        for i in range(len(wrancont)):
            cobs = inranges(self.obswave*(1-RV[i]/299792.5), wrancont[i])
            wave = self.obswave[cobs]*(1-RV[i]/299792.5) # RV shifting observed spectrum to stellar rest frame  
            flux = self.obsflux[cobs]
            telluric = np.ones_like(flux) if self.obstelluric is None else self.obstelluric[cobs]
            csyn = inranges(self.wave[0], wrancontpad[i]) # I use wrancontpad here as it conviniently provides some padding to prevent out-of-bound error in the next interpolation.
            flatsyn = np.interp(wave, self.wave[0][csyn], self.synth[0][csyn]) # interpolate onto the observed grid - no integration as coarse estimate is sufficient for masking lines
            flatsyn = flatsyn*telluric # BTW, I call it flatsyn just to highlight that it is flat, there is no other "syn".

            c = (flatsyn>continuum_threshold) & np.isfinite(flux)
            cerr = c & inranges(wave, np.quantile(wave, err_quantile_window))
            if cscale_mode=='spline':
                spl = make_smoothing_spline(wave[c], flux[c]/flatsyn[c], lam=smoothing_lambda)
                CS[i] = spl
                ERR[i] = np.sqrt(np.var(flux[cerr]/spl(wave[cerr])))
            else:
                ERR[i] = np.std(flux[cerr])
                if cscale_mode=='none':
                    CS[i] = 1
                elif cscale_mode=='segment_mean':
                    csegcont = c & inranges(wave, self.WRAN[i])
                    CS[i] = np.mean(flux[csegcont]/flatsyn[csegcont])
                elif cscale_mode=='window_quantile_mean':
                    CS[i] = np.mean(flux[cerr]/flatsyn[csegcont])
                else:
                    raise ValueError('Invalid `cscale_mode`')
                
            if debug_mode:
                CCONT[i] = c
                CERR[i] = cerr
                WAVE[i] = wave
                SYN[i] = flatsyn
                FLUX[i] = flux
                
        # Cleanup
        self.linelist = None
        self.wave = None
        self.spec = None
        self.wran = None
        self.WRAN = WRAN_cached
        self.nlte.grid_data = {}
        
        if debug_mode:
            debug_dict = {'WAVE': WAVE, 'SYN': SYN, 'CCONT': CCONT, 'CERR': CERR, 'FLUX': FLUX}
            return ERR, CS, debug_dict
        return ERR, CS    
    # endregion  


def fast_synthesize(sme, wave_ranges, resolutions=None, delta_lambda=0.003, linelist=None, normalize_by_continuum=True, reuse_nlte_grid=False):
    '''
    Speedy synthesis for multiple wave ranges. Generated spectrum is in the rest frame.

    Parameters
    ----------
    sme : SMEwrapper or SME_Structure object
        An initialized SMEwrapper or SME_Structure object with the desired stellar parameters set.

    wave_ranges : Nx2 array-like
        start and end values for each wavelength segment. Here, they can be both unsorted and overlapping - handling that is infact an expected use case.

    resolutions : None or float or Nx1 array-like, default: None
        Spectral resolution for each segment. If None, no broadening is applied. If single value, assumes same resolution for all segments.
        `sme.ipres` if pre-set is overwritten by this input. 

    delta_lambda : float, default: 0.003Å
        Delta wavelength for the synthesized grid in Å.

    linelist : str or list, optional
        If you want to pass a custom linelist. Else uses `sme.fulllinelist` or `vald` as applicable.
        NOTE: Predefined `sme.linelist` is NOT directly used to ensure consistent behaviour. Custom linelists **must** be explicitly passed here.

    normalize_by_continuum : bool, default: True
        Whether to continuum-normalize the synthesized spectrum.

    Returns
    -------
    WAVE : object-array of wave-grids for each input wavelength range
    SYN  : object-array of synthesized spectra of the above wave-grids
    '''
    sme.iptype = 'gauss'
    sme.normalize_by_continuum = normalize_by_continuum
    sme.vrad = 0
    sme.vrad_flag = "none"
    sme.cscale_flag = 'none'

    wave_ranges = np.array(wave_ranges).reshape(-1,2)
    addbroadening = False
    edgepad = 0
    sme.ipres = 0
    if resolutions is not None:
        resolutions = np.array(resolutions).reshape(-1)
        if len(resolutions)==1:
            sme.ipres = resolutions
        elif len(resolutions)!=len(wave_ranges):
            raise ValueError('Input resolution array does not match the number of wave ranges.') 
        else:
            addbroadening = True
            sme.ipres = 0
            edgepad = 2 * wave_ranges.max()/resolutions.min()
    sme.linelist = None
    if linelist is not None:
        sme.linelist = linelist
    elif hasattr(sme, 'fulllinelist'):
        sme.linelist = sme.fulllinelist
    elif 'vald' in globals():
        sme.linelist = vald
    else:
        raise ValueError('No linelist found. Please input a linelist or make sure the default linelist is available.')
    if not reuse_nlte_grid:
        sme.nlte.grid_data = {} # Fix to avoid the nlte part from trying to reuse a previous grid and bugging out
    wranpad = wave_ranges + [[-edgepad, +edgepad]]
    sme.linelist = sme.linelist[inranges(sme.linelist.wlcent,combine_ranges(wranpad))]
    sme.wave = np.arange(wranpad.min()-1, wranpad.max()+1, delta_lambda)
    
    # with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    sme = synthesize_spectrum(sme)
    WAVE = np.empty(len(wave_ranges),'O') 
    SYN = np.empty(len(wave_ranges),'O')
    if addbroadening:  
        for i in range(len(wave_ranges)):
            cpad = inranges(sme.wave[0], wranpad[i])
            wavepad = sme.wave[0][cpad]
            synpad = sme.synth[0][cpad]
            synpad = pyasl.instrBroadGaussFast(wavepad, synpad, resolutions[i], 'firstlast', maxsig=5)
            c = inranges(wavepad, wave_ranges[i])
            WAVE[i] = wavepad[c]
            SYN[i] = synpad[c]
    else:
        for i in range(len(wave_ranges)):
            c = inranges(sme.wave[0], wave_ranges[i])
            WAVE[i] = sme.wave[0][c]
            SYN[i] = sme.synth[0][c]

    # Cleanup
    # sme.linelist = None
    sme.wave = None
    sme.spec = None
    sme.wran = None    

    return WAVE, SYN
    

# region mcmc

_CLIGHT = 299792.458 # km/s
_GRID_META = ('syngrid', 'wavegrid', 'mu', 'delta_v') # Non-parameter keys of a grid
_BROADENING = ('vsini', 'vmac') # Applied as convolutions during the mcmc runs, never grid parameters

def _disk_annuli(mu):
    '''
    Annulus geometry of PySME's `Synthesizer.integrate_flux`.
    Returns the order that sorts `mu` from disk centre to limb, the projected radii of the n+1 annulus boundaries `r`, and the annulus weights `wt` (relative areas, summing to 1).
    '''
    mu = np.asarray(mu, dtype=float).reshape(-1)
    order = np.argsort(np.sqrt(1 - mu**2))
    rmu = np.sqrt(1 - mu[order]**2)
    if mu.size > 1:
        r = np.concatenate(([0], np.sqrt(0.5*(rmu[:-1]**2 + rmu[1:]**2)), [1]))
    else:
        r = np.array([0., 1.])
    return order, r, r[1:]**2 - r[:-1]**2

def create_mcmc_grid(sme, paramgrids, wave_ranges=None, delta_v=0.15, approx_resolution=None, max_vbroad=30, derived_params={}, filename=None, nprocesses=1, linelist=None, return_grid=False, existing_grid=None, dtype=np.float16):
    '''
    Synthesizes and optionally saves to file the interpolant grid for subsequent mcmc runs. Recommended to save to file as this is the most time-consuming step and you don't wanna repeat it.
    `MCMCsetup` is able to use a subset of the wave_ranges used here, voiding the need to compute multiple grids for stars with similar parameter ranges but different wavelength ranges of interest.

    What is stored are the continuum-normalized specific intensities at each of the `sme.mu` angles, unbroadened by rotation, macroturbulence and the instrument, on a log-spaced (constant velocity step) wavelength grid.
    The radiative transfer is computed directly on this grid (PySME's adaptive wavelength grid is bypassed, so `sme.accwi` plays no part), so no interpolation in wavelength happens anywhere.
    Rotation (`vsini`) and macroturbulence (`vmac`) are then applied during the mcmc runs by integrating these intensities over the stellar disk exactly the way PySME does, followed by the instrumental profile. So both can be fit without re-synthesizing, and no limb-darkening law is assumed.
    Hence if you want to fit for vsini and/or vmac, specify them via `param_bounds` in `MCMCsetup`.

    Parameters
    ----------
    sme : SMEwrapper object
        An initialized SMEwrapper object with the desired wave_ranges and fixed stellar parameters set.

    paramgrids : dict
        Dict of param_name:grid values. grid must be array-like. See PySME documentation for acceptable parameter names.
        Notes:
        1. `vsini` and `vmac` cannot be included here - they are broadening parameters of `MCMCsetup`.
        2. You can't fit for resolution. If you don't know the resolution, then may Param have mercy mercy on your soul. See what I did there! Param and param, mercy and mcmc... /-)

    wave_ranges : Nx2 array-like, optional
        start and end values for each wavelength segment. They should be sorted and non-overlapping. Use `create_ranges` and/or `combine_ranges` to make'em easily.
        If not given, will use the wave_ranges set with `make_fit_segments`. If those aren't set either, will throw an error.

    delta_v : float, default: 0.15 km/s
        Velocity step of the log-spaced wavelength grid (0.0019 Å at 3800 Å, 0.0033 Å at 6600 Å). It must resolve the unbroadened line profiles and the smallest broadening you will fit.

    approx_resolution : float, optional
        The lowest resolution of the spectra that will be fit with this grid. Only used to pad the wavelength windows against convolution edge effects.
        If not given, it is taken from `sme.RES` or `sme.obsres`. If neither is set, an error is thrown.

    max_vbroad : float, default: 30 km/s
        The largest vsini + 3*vmac (km/s) the grid must support. The windows are padded by this plus 5σ of the instrumental profile. `MCMCsetup` checks the padding against its broadening bounds.

    derived_params : dict, optional
        Dict of param_name:function entries. If you want to tie any parameter to some combination of parameters in paramgrids (except `vsini` and `vmac`).
        The function should be solely a function of the `SME_Structure` object. Example: `function = lambda s: 1e-4*s.teff + 0.3*s.logg`.
        Special Case: If you want to use the empirical vmic relation from GALAH DR3 (see Buder et al. 2021), simply pass `vmic='galah'` and it will be set as a function of teff and logg according to the relation in that paper.
        Note: Be careful not to mix up grids constructed with different derived_params, as the derived parameters will not be explicitly reflected in the saved grid and you might end up using the wrong grid for fitting.

    filename : string or Path-like, optional
        Path to save the synthesized grid as well as paramgrids to. Will be saved in .npz format and be readable by `MCMCsetup`.

    nprocesses : int, default: 1 (No multiprocessing)
        The number of processes (cpu cores) to use for parallel processing. If 1, no parallel processing is used.

    linelist : ValdFile or Linelist, optional
        In case you want to pass a custom linelist at this stage for some reason. Otherwise will use the full linelist that is provided to sme. Linelists are chopped to the wavelength ranges (plus some tolerance) input previously.

    return_grid : bool, default: False
        If False, will not return the synthesized grid and will only save to file if `filename` is given. This is useful for saving on RAM if the grid is large and you don't need it in memory after saving to file.
        If filename is None, return_grid will be forcibly set to True.

    existing_grid : dict or filename, optional
        A previously synthesized grid (the dict returned by this function, or the filename it was saved to) whose spectra should be reused.
        Any point of the new `paramgrids` that also exists in the old grid (matched per-parameter within a relative tolerance of 0.001) is copied over, and only the genuinely new points are synthesized. Useful for extending or refining a grid without recomputing it.
        The existing grid must have the same parameters in the same order, and the same `wavegrid`, `mu` and `delta_v` as implied by the current inputs - otherwise an error is thrown.
        Note: `derived_params` are NOT checked, as they aren't recorded in the saved grid. Reusing a grid built with different `derived_params` will silently give you an inconsistent grid.

    dtype : numpy dtype, default: np.float16
        Storage type of the intensities. float16 keeps ~3 significant digits (~2e-4 relative), and halves the size of float32.

    Returns
    -------
    if return_grid is True:
        Returns a dict of the format `{'wavegrid': wavegrid, 'syngrid': syngrid, 'mu': mu, 'delta_v': delta_v, **paramgrids}` where:
        wavegrid : object array
            The log-spaced wavelength grid of each padded window.
        syngrid : ndarray
            The synthesized grid, of shape `(*[len(grid) for grid in paramgrids], nmu, Nwave)`. syngrid[..., j, :] is annulus j's contribution to the normalized flux, `wt_j * I(mu_j) / F_continuum`, so summing over j gives the unbroadened normalized flux.
        mu : ndarray
            The mu angles of the annuli, from disk centre to limb.
        delta_v : float
            Velocity step of the wavelength grid.
        paramgrids : dict
            The input parameter grids that were used to create the synthesized grid. Returned (and saved) for your express convenience.
    '''
    # Checks and Basic Setup
    if wave_ranges is None:
        if not hasattr(sme, 'WRAN') or sme.WRAN is None:
            raise ValueError("Both `wave_ranges` and `sme.WRAN` are None. Please input wave_ranges or use `make_fit_segments` to set `sme.WRAN`.")
    else:
        sme.WRAN = wave_ranges
    paramgrids, derived_params = dict(paramgrids), dict(derived_params)
    if set(paramgrids.keys()) & set(derived_params.keys()):
        raise ValueError("paramgrids and derived_params share common keys. Parameters cannot be in both.")
    if set(_BROADENING) & (set(paramgrids.keys()) | set(derived_params.keys())):
        raise ValueError("'vsini' and 'vmac' cannot be grid or derived parameters. They are applied as convolutions during the mcmc runs, so set them via `param_bounds` in `MCMCsetup`.")
    for key, grid in paramgrids.items():
        grid = np.array(grid)
        if grid.ndim != 1:
            raise ValueError(f"Parameter grid for '{key}' must be 1-dimensional, got shape {grid.shape}")
        if not np.all(np.diff(grid) > 0):
            raise ValueError(f"Parameter grid for '{key}' must be monotonically increasing")
        paramgrids[key] = grid
    if existing_grid is not None:
        if isinstance(existing_grid, (str, bytes)) or hasattr(existing_grid, '__fspath__'):
            with np.load(existing_grid, allow_pickle=True) as data:
                existing_grid = dict(data)
        if any(k not in existing_grid for k in _GRID_META):
            raise ValueError(f"`existing_grid` does not contain all of {_GRID_META}. Grids made before the specific-intensity format cannot be reused.")
        existing_paramgrids = {k: v for k, v in existing_grid.items() if k not in _GRID_META}
        if list(existing_paramgrids.keys()) != list(paramgrids.keys()):
            raise ValueError(f"Parameters of `existing_grid` {list(existing_paramgrids.keys())} do not match the input paramgrids {list(paramgrids.keys())}. Note that the order matters.")
    # Modifiers & Housekeeping
    if linelist is None: linelist = sme.fulllinelist
    if 'vmic' in derived_params and isinstance(derived_params['vmic'], str) and derived_params['vmic']=='galah':
        derived_params['vmic'] = lambda s: calc_galah_vmic(s.teff,s.logg)
    if approx_resolution is None:
        if getattr(sme, 'RES', None) is not None:
            approx_resolution = np.min(np.hstack([np.ravel(r) for r in np.atleast_1d(np.asarray(sme.RES, dtype=object))]).astype(float))
        elif getattr(sme, 'obsres', None) is not None:
            approx_resolution = np.nanmin(sme.obsres)
        else:
            raise ValueError("Resolution unknown. Please pass `approx_resolution` (only used to pad the wavelength windows).")

    # Log-spaced grid of each padded window
    pad_v = max_vbroad + 5*_CLIGHT/(2.3548*approx_resolution) + 20*delta_v
    wranpad = combine_ranges(sme.WRAN * [1 - pad_v/_CLIGHT, 1 + pad_v/_CLIGHT])
    dlnw = delta_v/_CLIGHT
    wavegrid = _objarray([np.exp(np.log(lo) + dlnw*np.arange(int(np.log(hi/lo)/dlnw) + 1)) for lo, hi in wranpad])
    order, _, _ = _disk_annuli(sme.mu)
    mu = np.asarray(sme.mu, dtype=float)[order]

    # Intialize
    cached = {k: getattr(sme, k) for k in ('vsini', 'vmac', 'ipres', 'normalize_by_continuum', 'specific_intensities_only')}
    sme.vsini = 0; sme.vmac = 0; sme.ipres = 0
    sme.normalize_by_continuum = False; sme.specific_intensities_only = True
    sme.vrad_flag = 'none'; sme.cscale_flag = 'none'
    sme.wave = None; sme.synth = None
    sme.wran = [[wranpad[0][0], wranpad[-1][-1]]]; sme.vrad = np.zeros(1) # One segment spanning all windows (see `_create_spectrum`)
    sme.linelist = linelist[inranges(linelist.wlcent, wranpad)]
    syngrid = np.zeros([len(arr) for arr in paramgrids.values()] + [len(mu), sum(len(w) for w in wavegrid)], dtype=dtype)

    # Reuse whatever the existing grid already covers
    todo = np.ones(syngrid.shape[:-2], dtype=bool)
    if existing_grid is not None:
        EWAVE = existing_grid['wavegrid']
        if len(EWAVE)!=len(wavegrid) or not all(len(EWAVE[i])==len(wavegrid[i]) and np.allclose(EWAVE[i], wavegrid[i]) for i in range(len(wavegrid))) \
                or not np.allclose(existing_grid['mu'], mu) or not np.isclose(float(existing_grid['delta_v']), delta_v):
            raise ValueError("The wavegrid, mu or delta_v of `existing_grid` do not match the ones implied by the current inputs, so its spectra cannot be reused. Either match them or drop `existing_grid`.")
        newpos, oldpos = [], []
        for key, grid in paramgrids.items():
            egrid = np.asarray(existing_paramgrids[key], dtype=float)
            nearest = np.argmin(np.abs(grid[:,None]-egrid[None,:]), axis=1)
            match = np.isclose(grid, egrid[nearest], rtol=1e-3, atol=0)
            newpos.append(match.nonzero()[0]); oldpos.append(nearest[match])
        if all(len(pos) for pos in newpos):
            syngrid[np.ix_(*newpos)] = existing_grid['syngrid'][np.ix_(*oldpos)]
            todo[np.ix_(*newpos)] = False
            print(f"Reusing {(~todo).sum()} of {todo.size} grid points from `existing_grid`.")
        else:
            print("`existing_grid` has no grid points in common with the input paramgrids. Synthesizing from scratch.")

    # Populate syngrid
    multi_idxs = list(zip(*todo.nonzero()))
    try:
        if len(multi_idxs)==0:
            print("Nothing left to synthesize.")
        elif nprocesses==1:
            _init_worker(sme, paramgrids, derived_params, wavegrid)
            for multi_idx in tqdm(multi_idxs, total=len(multi_idxs), desc="Synthesizing grid"):
                    syngrid[multi_idx] =_create_spectrum(multi_idx)
        elif nprocesses > 1:
            with Pool(processes=nprocesses, initializer=_init_worker, initargs=(sme, paramgrids, derived_params, wavegrid), maxtasksperchild=3*nprocesses) as pool:
                for idx, result in tqdm(zip(multi_idxs, pool.imap(_create_spectrum, multi_idxs)), total=len(multi_idxs), desc="Synthesizing grid"):
                    syngrid[idx] = result
    finally:
        # Cleanup
        for k, v in cached.items(): setattr(sme, k, v)
        sme.wran = None
        sme.linelist = None

    if filename is not None:
        np.savez(filename, wavegrid=wavegrid, syngrid=syngrid, mu=mu, delta_v=delta_v, **paramgrids)
    else:
        return_grid = True

    if return_grid:
        return {'wavegrid': wavegrid, 'syngrid': syngrid, 'mu': mu, 'delta_v': delta_v, **paramgrids}

# Wrapper function - can be used with pool.map
def _create_spectrum(multi_idx):
    global _worker_sme, _worker_paramgrids, _worker_derived, _worker_wave
    for key, i in zip(_worker_paramgrids.keys(), multi_idx):
        # Consider correcting the abundances !!! to account for the auto-monh corection quirk.
        _worker_sme[key] = _worker_paramgrids[key][i]
    for key, fn in _worker_derived.items():
        _worker_sme[key] = fn(_worker_sme)
    # Pre-filling the synthesizer's wavelength grid makes PySME do the radiative transfer on exactly these points.
    # All windows go in as ONE gapped segment: PySME 0.4.x redoes its line handling over the whole linelist for every segment.
    # NOTE: this uses PySME 0.4.x internals (`Synthesizer.wint` + `reuse_wavelength_grid`). PySME 1.x exposes `sme.wint` for the same purpose.
    wave = np.concatenate(_worker_wave)
    synthesizer = Synthesizer()
    synthesizer.wint = {0: wave}
    wmod, smod, cmod, _ = synthesizer.synthesize_spectrum(_worker_sme, reuse_wavelength_grid=True)
    if len(wmod[0]) != len(wave) or not np.allclose(wmod[0], wave, rtol=1e-10, atol=0):
        raise RuntimeError("PySME did not synthesize on the supplied wavelength grid. This relies on PySME 0.4.x internals - check your PySME version.")
    order, _, wt = _disk_annuli(_worker_sme.mu)
    I = np.asarray(smod[0], dtype=float)[order]
    C = np.asarray(cmod[0], dtype=float)[order]
    return wt[:,None] * I / (wt @ C) # Annulus contributions to the normalized flux
# Initializer function for pool.map
def _init_worker(sme_template, pgrids, dparams, wave):
    global _worker_sme, _worker_paramgrids, _worker_derived, _worker_wave
    _worker_sme = deepcopy(sme_template)
    _worker_paramgrids = pgrids
    _worker_derived = dparams
    _worker_wave = wave


# MCMCsetup object handed to Pool workers once, so that emcee doesn't pickle it (grid and all) with every task
_mcmc_obj = None
def _init_mcmc_worker(obj):
    global _mcmc_obj
    _mcmc_obj = obj
def _mcmc_log_posterior(params):
    return _mcmc_obj.log_posterior(params)


class MCMCsetup:
    def __init__(self, smewrapper, grids, param_bounds=None, nprocesses=1, log_prior_function=None, create_grid_kwargs={}):
        '''
        At this point fit segments have already been created using `make_fit_segments` or directly in the `sme` object, along with all that is involved (RV, error computation and continuum scaling).
        If you're using a pre-computed grid, the SME object given here can have any combination of wave_ranges that are a subset of the wave_ranges used to create the grid. This enables computation of a common grid for multiple spectra.
        Everything star-specific (wavelength crop, instrumental profile, binning onto the observed pixels, continuum scaling, errors) is precomputed here, so `run_mcmc` only interpolates the grid and broadens.

        Parameters
        ----------
        smewrapper : SMEwrapper object
            An initialized SMEwrapper object with the fit segments set. `sme.RES` may hold one resolution per segment, or one per pixel (an array matching `sme.WAVE[i]` for each segment).
            The values of `sme.vsini` and `sme.vmac` are used if they are not in `param_bounds`.

        grids : Path-like or dict
            The grid(s) to be used for MCMC sampling, as made by `create_mcmc_grid`.
            If a path-like object is given, it is assumed to be a path to a saved .npz grid file.
            If a dict is given, it is assumed to be a dictionary of parameter grids and, optionally, the rest of the output of `create_mcmc_grid` (e.g., `{'teff': [5000, 5500, 6000], 'logg': [3.5, 4.0, 4.5]}`). See PySME documentation for acceptable parameter names.
            If supplied dict doesn't have the `'syngrid'` key, a synthesized grid will be created using `create_mcmc_grid`.
            Notes:
                1. Do NOT include `vsini` or `vmac` here. Use `param_bounds`.
                2. You can't fit for resolution. If you don't know the resolution, then may Param have mercy, mercy, on your soul. See what I did there! Param and param, mercy and mcmc... ¬‿¬.

        param_bounds : dict, optional
            Dict of param_name:spec entries to restrict or fix parameters, without rebuilding the grid. Keys can be any grid parameter, 'vsini' or 'vmac'.
            spec = (lo, hi) : the parameter is fit within [lo, hi]. For grid parameters, either can be None to keep the grid edge, and both must lie within the grid.
            spec = value   : the parameter is fixed to that value. For grid parameters it may lie between grid nodes (the grid is linearly interpolated there).
            Grid parameters not given here are fit over their whole grid. 'vsini' and 'vmac' not given here are fixed to `sme.vsini` and `sme.vmac`.
            Example: `param_bounds={'teff': (6800, 7200), 'monh': 0.25, 'vsini': (10, 26)}`.
            Fixing grid parameters reduces the dimensionality of the interpolation (each one halves its cost). Fixing both 'vsini' and 'vmac' precomputes the whole forward model per grid node, which is much faster.

        nprocesses : int, default: 1 (No multiprocessing)
            The number of processes (cpu cores) to use for parallel processing. If 1, no parallel processing is used.

        log_prior_function : callable, optional
            Function of the vector of fitted parameters, in the order of `self.free_params` (the free grid parameters in grid order, then 'vsini' and 'vmac' if fitted).
            The built-in prior prohibits values outside `self.bounds`; this cannot be overrriden.

        create_grid_kwargs : dict, optional
            Passed on to `create_mcmc_grid` if the grid has to be synthesized.

        Attributes
        ----------
        free_params : list of the fitted parameter names, in the order of the mcmc vector.
        bounds : list of (lo, hi) of each of them.
        fixed_params : dict of the fixed parameters and their values.
        paramgrids : dict of the full parameter grids, as loaded.

        Use `run_mcmc` to get the emcee sampler, and `model_spectrum` to get the model at a parameter vector.
        '''
        sme = smewrapper
        self.nprocesses = nprocesses
        self.log_prior_function = log_prior_function
        param_bounds = {} if param_bounds is None else dict(param_bounds)

        # Checks & Basic Setup
        if sme.FLUX is None or len(sme.FLUX) == 0:
            raise ValueError("The SME structure has no fit segments defined.")
        if sme.NSEG != len(sme.FLUX):
            raise ValueError("The number of fit segments in the SME structure does not match the number of wave ranges, possibly because you overwrote the wave ranges after instancing the fit segments.")
        if sme.RES is None:
            raise ValueError("Resolution MUST be set by this stage, one value per segment or one array per segment matching `sme.WAVE`.")
        if isinstance(grids, (str, bytes)) or hasattr(grids, '__fspath__'):
            with np.load(grids, allow_pickle=True) as data:
                grids = dict(data)
            if any(k not in grids for k in _GRID_META):
                raise ValueError(f"Loaded grid file does not contain all of {_GRID_META}. Grids made before the specific-intensity format must be rebuilt with `create_mcmc_grid`.")
        if 'syngrid' not in grids:
            if 'wavegrid' in grids:
                raise ValueError("`wavegrid` present in supplied grids but `syngrid` not found.")
            print("No synthesized grid found in supplied grids. Creating synthesized grid...")
            grids = create_mcmc_grid(sme, grids, return_grid=True, nprocesses=self.nprocesses, **create_grid_kwargs)
        grids = dict(grids)
        syngrid = grids.pop('syngrid')
        wavegrid = grids.pop('wavegrid')
        mu = np.asarray(grids.pop('mu'), dtype=float)
        self.delta_v = float(grids.pop('delta_v'))
        self.paramgrids = {k: np.asarray(v, dtype=float) for k, v in grids.items()}
        unknown = set(param_bounds) - set(self.paramgrids) - set(_BROADENING)
        if unknown:
            raise ValueError(f"`param_bounds` has parameters {sorted(unknown)} that are neither grid parameters {list(self.paramgrids)} nor {list(_BROADENING)}.")

        # Restricted and fixed grid parameters
        self.free_params, self.bounds, self.fixed_params = [], [], {}
        self._free_grids = []
        index, collapse = [], [] # per grid axis: slice into syngrid, and interpolation weight if the axis is fixed
        for name, grid in self.paramgrids.items():
            spec = param_bounds.get(name)
            if spec is None:
                index.append(slice(None)); collapse.append(None)
                self.free_params.append(name); self.bounds.append((grid[0], grid[-1])); self._free_grids.append(grid)
            elif np.ndim(spec) == 0:
                value = float(spec)
                if not grid[0] <= value <= grid[-1]:
                    raise ValueError(f"Fixed value {value} of '{name}' is outside its grid [{grid[0]}, {grid[-1]}].")
                if len(grid) == 1:
                    i, t = 0, 0.
                else:
                    i = min(np.searchsorted(grid, value, 'right') - 1, len(grid) - 2)
                    t = (value - grid[i])/(grid[i+1] - grid[i])
                    if np.isclose(t, 1, rtol=0, atol=1e-9): i, t = i+1, 0.
                    elif np.isclose(t, 0, rtol=0, atol=1e-9): t = 0.
                index.append(slice(i, i+1) if t == 0 else slice(i, i+2)); collapse.append(float(t))
                self.fixed_params[name] = value
            else:
                lo, hi = spec
                lo = grid[0] if lo is None else float(lo)
                hi = grid[-1] if hi is None else float(hi)
                if not grid[0] <= lo < hi <= grid[-1]:
                    raise ValueError(f"Bounds ({lo}, {hi}) of '{name}' must be increasing and within its grid [{grid[0]}, {grid[-1]}].")
                i0 = max(np.searchsorted(grid, lo, 'right') - 1, 0)
                i1 = max(np.searchsorted(grid, hi, 'left'), i0 + 1)
                index.append(slice(i0, i1+1)); collapse.append(None)
                self.free_params.append(name); self.bounds.append((lo, hi)); self._free_grids.append(grid[i0:i1+1])
        self._ngrid = len(self.free_params)

        # Broadening parameters
        vmax = {}
        for name in _BROADENING:
            spec = param_bounds.get(name, getattr(sme, name))
            if np.ndim(spec) == 0:
                if float(spec) < 0: raise ValueError(f"'{name}' cannot be negative.")
                self.fixed_params[name] = vmax[name] = float(spec)
            else:
                lo, hi = map(float, spec)
                if not 0 <= lo < hi:
                    raise ValueError(f"Bounds ({lo}, {hi}) of '{name}' must be increasing and non-negative.")
                self.free_params.append(name); self.bounds.append((lo, hi))
                vmax[name] = hi
        self._ibroad = {name: (self.free_params.index(name) if name in self.free_params else None) for name in _BROADENING}
        if len(self.free_params) == 0:
            raise ValueError("All parameters are fixed. There is nothing to fit.")
        self._lo = np.array([b[0] for b in self.bounds]); self._hi = np.array([b[1] for b in self.bounds])

        # Star specific arrays, per segment and pixel
        nseg = sme.NSEG
        WAVE = [np.asarray(w, dtype=float) for w in sme.WAVE]
        def perpixel(value, i, what):
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size == 1: return np.full(len(WAVE[i]), arr[0])
            if arr.size == len(WAVE[i]): return arr
            raise ValueError(f"{what} of segment {i} has {arr.size} values, but the segment has {len(WAVE[i])} pixels.")
        def persegment(values, what):
            if values is None: return [1.]*nseg
            if np.ndim(values) == 0: return [values]*nseg
            if len(values) == 1: return [values[0]]*nseg
            if len(values) != nseg: raise ValueError(f"{what} has {len(values)} entries, but there are {nseg} segments.")
            return list(values)
        RES = [perpixel(r, i, 'RES') for i, r in enumerate(persegment(sme.RES, 'RES'))]
        ERR = [perpixel(e, i, 'ERR') for i, e in enumerate(persegment(sme.ERR, 'ERR'))]
        CS = [perpixel(cs(WAVE[i]) if isinstance(cs, BSpline) else cs, i, 'CS') for i, cs in enumerate(persegment(sme.CS, 'CS'))]
        FLUX = [perpixel(f, i, 'FLUX') for i, f in enumerate(sme.FLUX)]
        BINE = []
        for w in WAVE:
            binw = np.diff(w)
            BINE.append(np.concatenate(([w[0]-binw[0]/2], w[:-1]+binw/2, [w[-1]+binw[-1]/2])))
        self._npix = np.array([len(w) for w in WAVE])

        # Crop the grid to what each segment needs: the instrumental profile reaches 5σ, and the disk integration vsini + 3*vmac beyond that
        dlnw = self.delta_v/_CLIGHT
        wstarts = np.array([w[0] for w in wavegrid])
        offsets = np.concatenate(([0], np.cumsum([len(w) for w in wavegrid])))
        cols, lam = [], []
        for i in range(nseg):
            reach = vmax['vsini'] + 3*vmax['vmac'] + 5*_CLIGHT/(2.3548*RES[i].min()) + 2*self.delta_v
            lo, hi = BINE[i][0]*(1 - reach/_CLIGHT), BINE[i][-1]*(1 + reach/_CLIGHT)
            iw = np.searchsorted(wstarts, lo, 'right') - 1
            if iw < 0 or hi > wavegrid[iw][-1]:
                raise ValueError(f"Segment {i} ({BINE[i][0]:.2f}-{BINE[i][-1]:.2f} Å) needs grid coverage over {lo:.2f}-{hi:.2f} Å for vsini<={vmax['vsini']}, vmac<={vmax['vmac']} and R>={RES[i].min():.0f}, which no grid window provides. "
                                 "Rebuild the grid with a larger `max_vbroad` (or wave ranges covering this segment), or lower the vsini/vmac bounds.")
            k0, k1 = np.searchsorted(wavegrid[iw], lo, 'left'), np.searchsorted(wavegrid[iw], hi, 'right')
            cols.append(offsets[iw] + np.arange(k0, k1)); lam.append(wavegrid[iw][k0:k1])
        cstarts = np.concatenate(([0], np.cumsum([len(c) for c in cols])))
        self._ntot = cstarts[-1]
        self._L = next_fast_len(self._ntot) # Circular convolution is fine: wrap-around only reaches the crop edges, which are never used

        # Instrumental profile + binning onto the observed pixels + continuum scaling, as one sparse matrix
        rows, colidx, vals = [], [], []
        for i in range(nseg):
            sig = WAVE[i]/(2.3548*RES[i])
            k0 = np.searchsorted(lam[i], BINE[i][:-1] - 5*sig, 'left')
            k1 = np.searchsorted(lam[i], BINE[i][1:] + 5*sig, 'right')
            K = k0[:,None] + np.arange((k1 - k0).max())[None,:]
            inside = K < k1[:,None]
            K = np.minimum(K, len(lam[i]) - 1)
            x = lam[i][K]
            s2 = np.sqrt(2)*sig[:,None]
            # Gaussian integrated over the observed pixel, at each grid pixel (weighted by its width)
            w = 0.5*(erf((BINE[i][1:,None] - x)/s2) - erf((BINE[i][:-1,None] - x)/s2)) * x*dlnw
            w[~inside] = 0
            w *= CS[i][:,None]/w.sum(axis=1, keepdims=True)
            r = np.broadcast_to(self._npix[:i].sum() + np.arange(len(WAVE[i]))[:,None], K.shape)
            rows.append(r[inside]); colidx.append(cstarts[i] + K[inside]); vals.append(w[inside])
        self._M = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(colidx))), shape=(self._npix.sum(), self._ntot))
        obs, err, cs = np.concatenate(FLUX), np.concatenate(ERR), np.concatenate(CS)
        valid = np.isfinite(obs) & np.isfinite(err) & np.isfinite(cs) & (err > 0)
        self._Mw = csr_matrix(self._M[valid].multiply(1/err[valid][:,None])) # χ² = |Mw·F - y|²
        self._y = obs[valid]/err[valid]

        # Grid: restrict, crop, then collapse the fixed axes
        S = syngrid[tuple(index)][..., np.concatenate(cols)].astype(np.float32)
        for ax in reversed(range(len(index))):
            t = collapse[ax]
            if t is None: continue
            S = np.take(S, 0, axis=ax) if S.shape[ax] == 1 else (1 - t)*np.take(S, 0, axis=ax) + t*np.take(S, 1, axis=ax)
        self._J = S
        _, self._r, _ = _disk_annuli(mu)
        self._mu = mu
        self._freq = rfftfreq(self._L)
        self._rot_cache = (None, None); self._mac_cache = (None, None)

        # Fixed broadening: precompute the whole forward model of every grid node
        self._G = None
        if self._ibroad['vsini'] is None and self._ibroad['vmac'] is None:
            nodes = self._J.reshape(-1, *self._J.shape[-2:])
            G = np.empty((len(nodes), len(self._y)))
            for c in range(0, len(nodes), 64):
                G[c:c+64] = (self._Mw @ self._flux(nodes[c:c+64], self.fixed_params['vsini'], self.fixed_params['vmac']).T).T
            self._G = G.reshape(*self._J.shape[:-2], len(self._y))

    # region forward model
    def _interp(self, S, p):
        'Multilinear interpolation of S along its leading (free grid parameter) axes. p must be within the grids.'
        if not self._free_grids: return S
        idx, ts = [], []
        for g, x in zip(self._free_grids, p):
            i = min(max(np.searchsorted(g, x, 'right') - 1, 0), len(g) - 2)
            idx.append(slice(i, i+2)); ts.append(float((x - g[i])/(g[i+1] - g[i])))
        blk = S[tuple(idx)]
        for t in ts:
            blk = blk[0] + t*(blk[1] - blk[0])
        return blk

    def _rot_ft(self, vsini):
        '''Fourier transform of each annulus's rotation kernel (PySME's annular convolution), built oversampled and binned to grid pixels.'''
        if self._rot_cache[0] == vsini: return self._rot_cache[1]
        if vsini <= 0:
            H = np.ones((len(self._mu), len(self._freq)))
        else:
            os = 9 # odd, so that the binning is centred
            hp = int(vsini/self.delta_v) + 1
            v = (self.delta_v/os)*np.arange(-(hp*os + os//2), hp*os + os//2 + 1)
            a1, a2 = vsini*self._r[:-1,None], vsini*self._r[1:,None]
            k = np.sqrt(np.clip(a2**2 - v**2, 0, None)) - np.sqrt(np.clip(a1**2 - v**2, 0, None))
            k = k.reshape(len(self._mu), 2*hp + 1, os).sum(axis=-1)
            k /= k.sum(axis=-1, keepdims=True)
            kpad = np.zeros((len(self._mu), self._L))
            kpad[:, :hp+1] = k[:, hp:]; kpad[:, -hp:] = k[:, :hp]
            H = rfft(kpad, axis=-1)
        self._rot_cache = (vsini, H)
        return H

    def _mac_ft(self, vmac):
        '''Fourier transform of each annulus's radial-tangential macroturbulence kernel (as in PySME).'''
        if self._mac_cache[0] == vmac: return self._mac_cache[1]
        if vmac <= 0:
            H = np.ones((len(self._mu), len(self._freq)))
        else:
            s = vmac/np.sqrt(2)/self.delta_v # in pixels
            sr, st = s*self._mu[:,None], s*np.sqrt(1 - self._mu[:,None]**2)
            H = 0.5*(np.exp(-2*(np.pi*self._freq*sr)**2) + np.exp(-2*(np.pi*self._freq*st)**2))
        self._mac_cache = (vmac, H)
        return H

    def _flux(self, J, vsini, vmac):
        'Disk-integrated normalized flux from annulus contributions J of shape (..., nmu, Ntot).'
        if vsini <= 0 and vmac <= 0:
            return J.sum(axis=-2)
        H = self._rot_ft(vsini) * self._mac_ft(vmac)
        return irfft((rfft(J, self._L, axis=-1)*H).sum(axis=-2), self._L, axis=-1)[..., :self._ntot]

    def _broadening(self, p):
        return [p[i] if i is not None else self.fixed_params[name] for name, i in self._ibroad.items()]

    def model_spectrum(self, params):
        '''
        The model (continuum-scaled) on the observed pixels at the parameter vector `params` (in the order of `self.free_params`).
        Returns an object array with one array per segment, matching `sme.WAVE`.
        '''
        p = np.asarray(params, dtype=float)
        F = self._flux(self._interp(self._J, p[:self._ngrid]), *self._broadening(p))
        return _objarray(np.split(self._M @ F, np.cumsum(self._npix)[:-1]))
    # endregion

    # region emcee functions
    def chisq_log_likelihood(self, params):
        p = np.asarray(params, dtype=float)
        if self._G is not None:
            model = self._interp(self._G, p[:self._ngrid])
        else:
            model = self._Mw @ self._flux(self._interp(self._J, p[:self._ngrid]), *self._broadening(p))
        return -0.5 * np.sum((model - self._y)**2)

    def log_prior(self, params):
        p = np.asarray(params, dtype=float)
        if np.any(p < self._lo) or np.any(p > self._hi):
            return -np.inf
        # User-defined priors
        if self.log_prior_function is not None:
            return self.log_prior_function(p)
        return 0

    # Log-posterior = log-prior + log-likelihood
    def log_posterior(self, params):
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.chisq_log_likelihood(params)
    # endregion

    def run_mcmc(self, nwalkers=None, nsteps=None, initial_vector=None):
        '''
        Runs emcee and returns the `emcee.EnsembleSampler`. The chains are in the order of `self.free_params`.
        Example: `samples = sampler.get_chain(discard=750, thin=1, flat=True)  # remove burn-in and flatten the chains`
        If `initial_vector` (nwalkers x ndim) isn't given, walkers start uniformly within `self.bounds`.
        '''
        # Initialize and run the sampler
        ndim = len(self.free_params)
        if nwalkers is None:
            nwalkers = 8*ndim if 8*ndim < 64 else 64
        if nsteps is None:
            nsteps = 4500
        if initial_vector is None:
            initial_vector = np.random.uniform(low=self._lo, high=self._hi, size=(nwalkers, ndim))
        if self.nprocesses <= 1:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)
            sampler.run_mcmc(initial_vector, nsteps, progress=True)
        else:
            # Workers get this object once (inherited under fork), instead of emcee pickling it with every task
            ctx = get_context('fork') if 'fork' in get_all_start_methods() else get_context()
            with ctx.Pool(processes=self.nprocesses, initializer=_init_mcmc_worker, initargs=(self,)) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, _mcmc_log_posterior, pool=pool)
                sampler.run_mcmc(initial_vector, nsteps, progress=True)
        return sampler
