import numpy as np
from tqdm.auto import tqdm
from copy import deepcopy
from astropy.stats import sigma_clipped_stats, sigma_clip

import emcee
from scipy.interpolate import RegularGridInterpolator
try:
    from multiprocessing import Pool
except ImportError:
    pass # Multiprocessing not available, will run in single process mode.

from .utils import *
from PyAstronomy import pyasl
from telfit import Modeler

from scipy.interpolate import make_smoothing_spline, make_interp_spline

import logging
logging.basicConfig(level=logging.ERROR)

class SMEwrapper(SME_Structure):
    def __init__(self, fulllinelist=vald, teff=5777, logg=4.44, monh=0.0, vsini=0, vmic=1.1, vmac=0):
        '''
        Input parameters are self-explanatory.
        Option to input wave_ranges and resolution(s) is given here for the purpose of synthesizing spectra, not applicable for fitting.

        Non-obvious Parameters
        ----------
        fulllinelist : ValdFile or Linelist object, default provided
            Linelist object. See PySME documentation on how to get/build it. A Default linelist is provided encompassing all lines an AFGK-type optical spectrum is likely to need, spanning 3000-9000 Å.

        nlteelems : list, optional 
            List of elements to perform NLTE calculations for. Computationally inexpensive, so recommended.
            Elements must be represented as their chemical symbols. See PySME documentation for the list of supported elements. 
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

    def input_observed_spectrum(self, wave, flux, err=None, res=None):
        '''
        Input 1-D arrays for observed wavelength, normalised flux, relative flux error and resolution (optional). Arrays must have same length.
        Option to input single resolution values for each segment is given in the function `make_fit_segments`.
        Unless you really trust the instrument supplied error estimates on the flux values, I recommend leaving `err` as None and setting ERR='fit' in `make_fit_segments` to calculate the (rms) error empirically in each fit segment.
        '''
        if len(wave) != len(flux): raise ValueError("All input arrays must have the same length")
        if (res is not None) and (len(res)!=len(wave)): raise ValueError('res array doesn\'t match spectrum length.')
        if err is not None and len(err)!=len(wave): raise ValueError('err array doesn\'t match spectrum length.')
        self.obswave = np.array(wave)
        self.obsflux = np.array(flux)
        self.obserr = np.array(err) if err is not None else None
        self.obsres = np.array(res).reshape(-1) if res is not None else None
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

        These parameters are generally in the 1st header of the fits files.
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
            The indices from `self.obswave` to build the each of the WAVE and FLUX arrays. 
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
        obj.CSEG = np.array([inranges(obj.obswave*(1-obj.RV[i]/299792.5), ran).nonzero()[0] for i,ran in enumerate(obj.WRAN)],'O')
        obj.WAVE = np.array([obj.obswave[c]*(1-obj.RV[i]/299792.5) for i,c in enumerate(obj.CSEG)], 'O')
        obj.FLUX = np.array([obj.obsflux[c] for c in obj.CSEG], 'O')

        # CS and ERR
        if (isinstance(CS, str) and CS=='fit') or (isinstance(ERR, str) and ERR=='fit'):
            _ERR, _CS = obj.get_error_and_cscale(obj.RV, approx_resolution, linelist=linelist, **err_cs_kwargs)
        #---
        if CS is None: CS = np.ones(obj.NSEG)
        elif isinstance(CS, str) and CS=='fit': CS = _CS
        elif isinstance(CS, (int, float, np.number)): CS = np.full(obj.NSEG, CS)
        elif len(CS)==obj.NSEG: CS = np.array(CS,'O')
        else: raise ValueError('Length of input CS array does not match the number of segments.')
        obj.CS = CS
        #---
        if ERR is None: pass
        elif isinstance(ERR, str):
            if ERR=='fit': ERR = _ERR
            elif ERR=='propagate': ERR = np.array([obj.obserr[c] for c in obj.CSEG],'O')
            elif ERR=='none': ERR = None
            else: raise ValueError('Invalid string input for ERR. Should be one of "fit", "propagate" or "none".')
        elif len(ERR)==obj.NSEG: ERR = np.array(ERR,'O')
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
            c1 = sigma_clip(obj.RV, sigma=2, maxiters=3).mask & (np.abs(obj.RV-obj.RV.mean()) > 2)
            c2 = sigma_clip(obj.ERR, sigma_upper=3, sigma_lower=15, maxiters=3).mask if obj.ERR is not None else np.zeros_like(c1, dtype=bool)
            meanCS = np.hstack([spl(obj.WAVE[i].mean()) for i,spl in enumerate(obj.CS)]) if obj.CS.dtype==object else np.array([arr.mean() for arr in obj.CS])
            c3 = (meanCS<0.8) | (meanCS>1.2)
            icut = (c1 | c2 | c3).nonzero()[0]
            if len(icut):
                print(f'The median fitted RV is {np.median(obj.RV[~(c1|c2|c3)]):.2f} km/s.')
                if obj.ERR is not None:
                    print(f'The median ERR is {np.nanmedian(obj.ERR[~(c1|c2|c3)]):.2f}.')
                print(f'{len(icut)} segments will be removed due to poor fits to the RV, ERR and/or CS values. See function documentation for details.')
                # Formatted to display aligned neatly in fixed-width font
                print(f'{"iseg":<5} {"WAVE_RANGE":<20} {"RV":<8} {"ERR":<8} {"CS":<8} reason')
                for iseg in icut:
                    reason = f'{"RV " if c1[iseg] else "   "}{"ERR " if c2[iseg] else "    "}{"CS" if c3[iseg] else "  "}'
                    print(f'{iseg:<5} {obj.WRAN[iseg].round(2)!s:<20} {obj.RV[iseg]:<8.2f} {obj.ERR[iseg].round(2)!s:<8} {meanCS[iseg]:<8.2f} {reason}')
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

        CSEG = np.array([inranges(obj.obswave*(1-RV[i]/299792.5), ran).nonzero()[0] for i,ran in enumerate(wave_ranges)],'O')
        WAVE = np.array([obj.obswave[c]*(1-RV[i]/299792.5) for i,c in enumerate(CSEG)])
        FLUX = np.array([obj.obsflux[c] for c in CSEG])

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
        
        if debug_mode:
            debug_dict = {'WAVE': WAVE, 'SYN': SYN, 'CCONT': CCONT, 'CERR': CERR, 'FLUX': FLUX}
            return ERR, CS, debug_dict
        return ERR, CS    
    # endregion  


def fast_synthesize(sme, wave_ranges, resolutions=None, delta_lambda=0.003, linelist=None, normalize_by_continuum=True):
    '''
    Speedy synthesis for multiple wave ranges. Generated spectrum is in the rest frame.

    Parameters
    ----------
    sme : SMEwrapper or SME_Structure object
        An initialized SMEwrapper or SME_Structure object with the desired stellar parameters set.

    wave_ranges : Nx2 array-like
        start and end values for each wavelength segment. Here, they can be both unsorted and overlapping - handling that is infact an expected use case.

    resolutions : None or float or Nx1 array-like, default: None
        Spectral resolution for each segment may be input here, if not given with the input spectrum. If None, no broadening is applied. If single value, assumes same resolution for all segments.

    delta_lambda : float, default: 0.003Å
        Delta wavelength for the synthesized grid in Å.

    linelist : str or list, optional
        If you want to pass a custom linelist. Else uses `sme.fulllinelist` or `vald` as applicable.
        NOTE: Predefined `sme.linelist` is NOT directly used to ensure consistent behaviour. Custom linelists **must** be explicitly passed here.

    # edgepad : 'auto' or 'none'/None or float, default: 'auto'
    #     The amount of edge padding to apply to the wavelength ranges to account for convolution edge effects.
    #     If 'auto', the padding is automatically determined based on the input wavelength ranges and resolutions. If 'none' or None, no edge padding is applied. If float, applies the given amount of edge padding in Å to all segments.

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
    wranpad = wave_ranges + [[-edgepad, +edgepad]]
    sme.linelist = sme.linelist[inranges(sme.linelist,combine_ranges(wranpad))]
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

def create_mcmc_grid(sme, paramgrids, wave_ranges=None, delta_lambda=0.003, approx_resolution=None, derived_params={}, filename=None, nprocesses=1, linelist=None, return_grid=False):
    '''
    Synthesizes and optionally saves to file the interpolant grid for subsequent mcmc runs. Recommended to save to file as this is the most time-consuming step and you don't wanna repeat it.
    `run_mcmc` is able to use a subset of the wave_ranges used here, voiding the need to compute multiple grids for stars with similar parameter ranges but different wavelength ranges of interest.
    The synthesized grid is unbroadened by vsini and at infinite resolution. The effects of vsini and resolution are instead treated by convolving the synthesized spectrum with the appropriate kernels during the mcmc runs, which has the advantage of being much faster and memory efficient.
    Hence if you want to fit for vsini, specify it directly in the `run_mcmc` function.

    Parameters
    ----------
    sme : SMEwrapper object
        An initialized SMEwrapper object with the desired wave_ranges and fixed stellar parameters set.

    wave_ranges : Nx2 array-like, optional
        start and end values for each wavelength segment. They should be sorted and non-overlapping. Use `create_ranges` and/or `combine_ranges` to make'em easily.
        If not given, will use the wave_ranges set with `make_fit_segments`. If those aren't set either, will throw an error.

    paramgrids : dict
        Dict of param_name:grid values. grid must be array-like. See PySME documentation for acceptable parameter names. 
        Notes:
        1. Do NOT include `vsini` here. It will be ignored.
        2. You can't fit for resolution. If you don't know the resolution, then may Param have mercy mercy on your soul. See what I did there! Param and param, mercy and mcmc... /-)
    
    delta_lambda : float, default: 0.003Å
        Delta wavelength for the synthesized grid in Å. I recommend `delta_lambda` should oversample the observed spectrum by at least a factor of 5.

    approx_resolution : int, optional
        The approximate resolution of the spectra to be synthesized. This is used to determine the amount of edge padding to apply to the wavelength ranges to account for convolution edge effects during the mcmc runs.
        If not given and `sme.RES` or `sme.obsres` has not been set previously, no edge padding will be applied. This WILL result in edge effects in the synthetic spectra during the mcmc runs.
        
    derived_params : dict, optional
        Dict of param_name:function entries. If you want to tie any parameter to some combination of parameters in paramgrids (except `vsini`).
        The function should be solely a function of the `SME_Structure` object. Example: `function = lambda s: 1e-4*s.teff + 0.3*s.logg`.
        Special Case: If you want to use the empirical vmic relation from GALAH DR3 (see Buder et al. 2021), simply pass `vmic='galah'` and it will be set as a function of teff and logg according to the relation in that paper.
        Note: Be careful not to mix up grids constructed with different derived_params, as the derived parameters will not be explicitly reflected in the saved grid and you might end up using the wrong grid for fitting.
    
    filename : string or Path-like, optional
        Path to save the synthesized grid as well as paramgrids to. Will be saved in .npz format and be readable by the `run_mcmc` method.

    nprocesses : int, default: 1 (No multiprocessing)
        The number of processes (cpu cores) to use for parallel processing. If 1, no parallel processing is used.  
        
    linelist : ValdFile or Linelist, optional
        In case you want to pass a custom linelist at this stage for some reason. Otherwise will use the full linelist that is provided with this package. Linelists are chopped to the wavelength ranges (plus some tolerance) input previously.

    return_grid : bool, default: True
        If False, will not return the synthesized grid and will only save to file if `filename` is given. This is useful for saving on RAM if the grid is large and you don't need it in memory after saving to file.
        If filename is None, return_grid will be forcibly set to True.

    Returns
    -------
    if return_grid is True:
        Returns a dict of the format `{'wavegrid': wavegrid, 'syngrid': syngrid, **paramgrids}` where:
        wavegrid : ndarray
            The wavelength grid for the synthesized spectra. This is the same for all spectra in the grid and is determined by the wave_ranges in `sme` and `delta_lambda`.
        syngrid : ndarray
            The synthesized grid of spectra. The order of the dimensions corresponds to the order of the parameters in `paramgrids` followed by the wavegrid dimension.
        paramgrids : dict
            The input parameter grids that were used to create the synthesized grid. Returned (and saved) for your express convenience.
    '''
    # Checks and Basic Setup
    if wave_ranges is None:
        if not hasattr(sme, 'WRAN') or sme.WRAN is None:
            raise ValueError("Both `wave_ranges` and `sme.WRAN` are None. Please input wave_ranges or use `make_fit_segments` to set `sme.WRAN`.")
    else:
        sme.WRAN = wave_ranges
        sme.wran = None; sme.wave=None; sme.synth=None
    if set(paramgrids.keys()) & set(derived_params.keys()):
        raise ValueError("paramgrids and derived_params share common keys. Parameters cannot be in both.")
    if 'vsini' in paramgrids.keys(): del paramgrids['vsini']; print("WARNING: 'vsini' cannot be a grid parameter and is ignored if included in paramgrids. If you want to fit for vsini, specify it directly in the `run_mcmc` function.")
    for key, grid in paramgrids.items():
        grid = np.array(grid)
        if grid.ndim != 1:
            raise ValueError(f"Parameter grid for '{key}' must be 1-dimensional, got shape {grid.shape}")
        if not np.all(np.diff(grid) > 0):
            raise ValueError(f"Parameter grid for '{key}' must be monotonically increasing")
        paramgrids[key] = grid
    # Modifiers & Housekeeping
    if linelist is not None: sme.linelist = linelist
    else: sme.linelist = sme.fulllinelist
    if 'vmic' in derived_params and derived_params['vmic']=='galah':
        derived_params['vmic'] = lambda s: calc_galah_vmic(s.teff,s.logg)

    # Intialize
    sme.ipres = 0
    vsini_cached = sme.vsini; sme.vsini = 0
    if approx_resolution is not None:
        edgepad = 2 * sme.WRAN.max()/approx_resolution
    elif sme.RES is not None:
        edgepad = 2 * sme.WRAN.max()/sme.RES.min()
    elif sme.obsres is not None:
        edgepad = 2 * sme.WRAN.max()/sme.obsres.min()
    else:
        edgepad = 0.3 
    edgepad = max(edgepad, 20*delta_lambda, 0.3) # Ensure minimal padding to handle vsini and avoid later conflicts in ranindices.
    wranpad = combine_ranges(sme.WRAN + [[-edgepad, +edgepad]])
    sme.linelist = sme.linelist[inranges(sme.linelist,wranpad)]
    sme.wave = np.arange(wranpad[0][0]-1, wranpad[-1][-1]+1, delta_lambda)
    cseg = [inranges(sme.wave[0], ran) for ran in wranpad]      
    PWAVE = np.array([sme.wave[0][c] for c in cseg],'O')
    syngrid = np.zeros([len(arr) for arr in paramgrids.values()]+[np.hstack(PWAVE).size], dtype=np.float16) 
    
    # Populate syngrid
    multi_idxs = list(np.ndindex(*syngrid.shape[:-1]))
    if nprocesses==1:
        _init_worker(sme, paramgrids, derived_params, cseg)
        for multi_idx in tqdm(multi_idxs, total=len(multi_idxs), desc="Synthesizing grid"):
                syngrid[multi_idx] =_create_spectrum(multi_idx)
    elif nprocesses > 1:
        with Pool(processes=nprocesses, initializer=_init_worker, initargs=(sme, paramgrids, derived_params, cseg), maxtasksperchild=3*nprocesses) as pool:
            for idx, result in tqdm(zip(multi_idxs, pool.imap(_create_spectrum, multi_idxs)), total=len(multi_idxs), desc="Synthesizing grid"):
                syngrid[idx] = result

    # Cleanup
    sme.vsini = vsini_cached
    sme.linelist = None
        
    if filename is not None:
        np.savez(filename, wavegrid=PWAVE, syngrid=syngrid, **paramgrids)
    else:
        return_grid = True

    if return_grid:
        return {'wavegrid': PWAVE, 'syngrid': syngrid, **paramgrids}

# Wrapper function - can be used with pool.map
def _create_spectrum(multi_idx):
    global sme, cseg, derived_params, paramgrids
    for key, i in zip(paramgrids.keys(), multi_idx):
        sme[key] = paramgrids[key][i]
    for key, fn in derived_params.items():
        sme[key] = fn(sme)
    sme = synthesize_spectrum(sme)
    PSYN = [sme.synth[0][c] for c in cseg]
    return np.hstack(PSYN)
# Initializer function for pool.map
def _init_worker(sme_template, pgrids, dparams, cs):
    global sme, cseg, derived_params, paramgrids
    sme = deepcopy(sme_template)
    cseg = cs
    derived_params = dparams
    paramgrids = pgrids


class MCMCsetup:
    def __init__(self, smewrapper, grids, vsini_grid=None, nprocesses=1, log_prior_function=None, create_grid_kwargs={}):
        '''
        At this point fit segments have already been created using `make_fit_segments` or directly in the `sme` object, along with all that is involved (RV, error computation and continuum scaling). 
        If you're using a pre-computed grid, the SME object given here can have any combination of wave_ranges that are a subset of the wave_ranges used to create the grid. This enables computation of a common grid for multiple spectra.
        
        Parameters
        ----------
        sme : SMEwrapper object
            An initialized SMEwrapper object with the desired wave_ranges and fixed stellar parameters set.

        grids : Path-like or dict
            The grid(s) to be used for MCMC sampling.
            If a path-like object is given, it is assumed to be a path to a saved .npz grid file.
            If a dict is given, it is assumed to be a dictionary of parameter grids and, optionally, a wavegrid and synthesized grid (e.g., `{'teff': [5000, 5500, 6000], 'logg': [3.5, 4.0, 4.5], 'wavegrid':<size N array>, 'syngrid':<3x3xN array>}`). See PySME documentation for acceptable parameter names.
            If supplied dict doesn't have the `'syngrid'` key, a synthesized grid will be created using `create_mcmc_grid`.
            Notes:
                1. Do NOT include `vsini` here. It will be ignored.
                2. You can't fit for resolution. If you don't know the resolution, then may Param have mercy, mercy, on your soul. See what I did there! Param and param, mercy and mcmc... I'll see myself out.

        vsini_grid : [vsini_min, vsini_max], optional
            If given, will be used to fit vsini. Only the min and max values matter for fitting this as vsini is treated as a convolution parameter during mcmc runs, not an interpolant parameter. Hence has the advantage of being computationally quite cheap.
            If not given, vsini will not be fit for and will be fixed to the value set in `sme`.

        nprocesses : int, default: 1 (No multiprocessing)
            The number of processes (cpu cores) to use for parallel processing. If 1, no parallel processing is used.

        log_prior_function : callable, optional
            Function of a vector of fitting parameters. The order of parameters in the vector should be the same as that in grids, vsini going at the end if applicable.
            The built-in log-prior function prohibits values outside the parameter grids; this cannot be overrriden. 

        Returns
        -------
        sampler : emcee.EnsembleSampler object
            The MCMC sampler object after running the sampling. The chains can be accessed via `sampler.chain` and the log-probability values via `sampler.lnprobability`.
            See emcee documentation for more details on the sampler object and how to analyze the chains.
            Example: `samples = sampler.get_chain(discard=750, thin=1, flat=True)  # remove burn-in and flatten the chains`
        '''
        global sme
        sme = smewrapper
        self.nprocesses = nprocesses

        # Checks & Basic Setup
        if len(sme.FLUX) == 0:
            raise ValueError("The SME structure has no fit segments defined.")
        if sme.obswave is None or sme.obsflux is None:
            raise ValueError("Observed spectrum not set in the SME structure. Please use the `input_observed_spectrum` method to input the observed spectrum before running MCMC.")
        if sme.NSEG != len(sme.FLUX):
            raise ValueError("The number of fit segments in the SME structure does not match the number of wave ranges, probably because you overwrote the wave ranges after instancing the fit segments. You should just start over with a fresh SMEwrapper object.")
        if sme.RES is None:
            raise ValueError("Resolution MUST be by this stage. And make sure it is array-like that matches the number of segments if you're gonna do it this late.")
        elif sme.RES is not None:
            sme.RES = np.array(sme.RES).reshape(-1)
            if len(sme.RES) != sme.NSEG:
                if len(sme.RES) == 1:
                    sme.RES = np.tile(sme.RES, sme.NSEG)
                else:
                    raise ValueError("Resolution array length does not match the number of segments.")
        if vsini_grid is not None and len(vsini_grid)!=2:
            raise ValueError("`vsini_grid` should be a list or array of [vsini_min, vsini_max]")
        if isinstance(grids, (str, bytes)) or hasattr(grids, '__fspath__'):
            with np.load(grids, allow_pickle=True) as data:
                grids = dict(data)
            if 'syngrid' not in grids or 'wavegrid' not in grids:
                raise ValueError("Loaded grid file does not contain 'syngrid' and/or 'wavegrid'. Please check the file.")
        if 'syngrid' not in grids:
            if 'wavegrid' in grids:
                raise ValueError("`wavegrid` present in supplied grids but `syngrid` not found.")
            print("No synthesized grid found in supplied grids. Creating synthesized grid...")
            grids = create_mcmc_grid(sme, grids, return_grid=True, nprocesses=self.nprocesses, **create_grid_kwargs)

        # global syngrid, wavegrid, paramgrids
        self.syngrid = grids.pop('syngrid')
        self.wavegrid = grids.pop('wavegrid')
        self.delta_lambda = self.wavegrid[0][1]-self.wavegrid[0][0]
        if 'vsini' in grids: del grids['vsini']
        self.paramgrids = grids        

        self.vsini_grid = vsini_grid
        self.log_prior_function = log_prior_function

        if sme.CS.dtype==object:
            self.CS = np.array([sme.CS[i](sme.WAVE[i]) for i in range(sme.NSEG)], dtype='O')
        else:
            self.CS = sme.CS.copy()

        # Intialize
        sme.ipres = 0

        wvgridsts = np.array([arr[0] for arr in self.wavegrid])
        wvgridnds = np.array([arr[-1] for arr in self.wavegrid])
        self.ranindices = np.searchsorted(wvgridsts, sme.WRAN[:,0]) - 1
        if np.any(sme.WRAN[:,1] > wvgridnds):
            raise ValueError("The current wavelength ranges are not encompassed by the wavegrid. Please check the wavegrid and the fit segments.")
        self.gridlengths = np.array([len(arr) for arr in self.wavegrid])

        # Star Specific arrays
        self.obspec = np.hstack(sme.FLUX)
        if sme.ERR is None: self.specerr = 1
        else: self.specerr = np.hstack(sme.ERR) if sme.ERR.dtype==object else np.hstack(np.repeat(sme.ERR,[len(arr) for arr in sme.FLUX]))
        self.BINE = np.empty(sme.NSEG, dtype='O')
        for i in range(sme.NSEG):
            binw = np.diff(sme.WAVE[i])
            self.BINE[i] = np.zeros(len(sme.WAVE[i])+1)
            self.BINE[i][1:-1] = sme.WAVE[i][:-1]+binw/2
            self.BINE[i][0] = sme.WAVE[i][0]-binw[0]/2
            self.BINE[i][-1] = sme.WAVE[i][-1]+binw[-1]/2

        # Make Interpolant
        self.spint = RegularGridInterpolator(tuple(self.paramgrids.values()), self.syngrid, bounds_error=False, fill_value=0.0)

    # emcee functions
    def chisq_log_likelihood(self, params):
        if self.vsini_grid is not None:
            vsini = params[-1]
            params = params[:-1]
        else:
            vsini = sme.vsini
        supersyn = self.spint(params).squeeze()
        supersyn = np.split(supersyn, np.cumsum(self.gridlengths)[:-1])
        synspec = np.empty(sme.NSEG, dtype='O')
        for i, ig in enumerate(self.ranindices):
            synspec[i] = pyasl.fastRotBroad(self.wavegrid[ig], supersyn[ig], 0.81, vsini)
            synspec[i] = pyasl.instrBroadGaussFast(self.wavegrid[ig], synspec[i], sme.RES[i], 'firstlast', maxsig=5)
            # Rebinning by integrating the flux and then dividing by the bin widths to get average flux in each bin, which is more accurate than interpolating the flux onto the observed grid directly.
            I = np.zeros_like(synspec[i])
            I[1:] = np.cumsum(0.5 * (synspec[i][:-1] + synspec[i][1:])) * self.delta_lambda
            I_edges = np.interp(self.BINE[i], self.wavegrid[ig], I)
            synspec[i] = np.diff(I_edges)/np.diff(self.BINE[i]) 
            synspec[i] = synspec[i] * self.CS[i] # Continuum scaling
        synspec = np.hstack(synspec)
        chisq = np.nansum(((synspec-self.obspec)/self.specerr)**2)
        return -0.5 * chisq
    
    def log_prior(self, params):
        if self.vsini_grid is not None:
            vsini = params[-1]
            params = params[:-1]
            if not (self.vsini_grid[0] <= vsini < self.vsini_grid[1]):
                return -np.inf
        for p, grid in zip(params, self.paramgrids.values()):
            if p < grid.min() or p > grid.max():
                return -np.inf
        # User-defined priors
        if self.log_prior_function is not None:
            return self.log_prior_function(params)
        return 0
    
    # Log-posterior = log-prior + log-likelihood
    def log_posterior(self, params):
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.chisq_log_likelihood(params)
    
    def run_mcmc(self, nwalkers=None, nsteps=None, initial_vector=None):
        # Initialize and run the sampler
        ndim = len(self.paramgrids) + (1 if self.vsini_grid is not None else 0)
        if nwalkers is None:
            nwalkers = 8*ndim if 8*ndim < 64 else 64
        if nsteps is None:
            nsteps = 4500
        if initial_vector is None:
            initial_vector = np.random.uniform(low=[grid[0] for grid in self.paramgrids.values()] + ([self.vsini_grid[0]] if self.vsini_grid is not None else []), 
                                            high=[grid[-1] for grid in self.paramgrids.values()] + ([self.vsini_grid[1]] if self.vsini_grid is not None else []), 
                                            size=(nwalkers, ndim))
        if self.nprocesses <= 1:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)
            sampler.run_mcmc(initial_vector, nsteps, progress=True)
        else: 
            with Pool(processes=self.nprocesses) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior, pool=pool)
                sampler.run_mcmc(initial_vector, nsteps, progress=True)
        return sampler

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# def run_mcmc(sme, grids, vsini_grid=None, log_prior_function=None, nprocesses=1, nwalkers=None, nsteps=None, initial_vector=None, delta_lambda=0.003, derived_params={}, linelist=None, save_grids_to=None):
#     '''
#     At this point fit segments have already been created using `make_fit_segments` or directly in the `sme` object, along with all that is involved (RV, error computation and continuum scaling). 
#     If you're using a pre-computed grid, the SME object given here can have any combination of wave_ranges that are a subset of the wave_ranges used to create the grid. This enables computation of a common grid for multiple spectra.
    
#     Parameters
#     ----------
#     sme : SMEwrapper object
#         An initialized SMEwrapper object with the desired wave_ranges and fixed stellar parameters set.

#     grids : Path-like or dict
#         The grid(s) to be used for MCMC sampling.
#         If a path-like object is given, it is assumed to be a path to a saved .npz grid file.
#         If a dict is given, it is assumed to be a dictionary of parameter grids and, optionally, a wavegrid and synthesized grid (e.g., `{'teff': [5000, 5500, 6000], 'logg': [3.5, 4.0, 4.5], 'wavegrid':<size N array>, 'syngrid':<3x3xN array>}`). See PySME documentation for acceptable parameter names.
#         If supplied dict doesn't have the `'syngrid'` key, a synthesized grid will be created using `create_mcmc_grid`.
#         Notes:
#             1. Do NOT include `vsini` here. It will be ignored.
#             2. You can't fit for resolution. If you don't know the resolution, then may Param have mercy, mercy, on your soul. See what I did there! Param and param, mercy and mcmc... I'll see myself out.

#     vsini_grid : [vsini_min, vsini_max], optional
#         If given, will be used to fit vsini. Only the min and max values matter for fitting this as vsini is treated as a convolution parameter during mcmc runs, not an interpolant parameter. Hence has the advantage of being computationally quite cheap.
#         If not given, vsini will not be fit for and will be fixed to the value set in `sme`.
    
#     log_prior_function : callable, optional
#         Function of a vector of fitting parameters. The order of parameters in the vector should be the same as that in grids, vsini going at the end if applicable.
#         The built-in log-prior function prohibits values outside the parameter grids; this cannot be overrriden. 

#     nprocesses : int, default: 1 (No multiprocessing)
#         The number of processes (cpu cores) to use for parallel processing. If 1, no parallel processing is used.  

#     nwalkers : int, default: `8*nparams if 8*nparams < 64 else 64`
#         Number of walkers to use in the MCMC sampling. If not given, will default to 8 times the number of parameters being fit, capped at 64.

#     nsteps : int, default: 4500
#         Number of steps to run the MCMC sampling. If not given, will default to 4500.

#     initial_vector : array-like (nwalkers, nparams), optional
#         Initial vector of parameters for the MCMC sampling. If not given, will default to a uniform distribution within the parameter grids. 
#         If given, should be of shape (nwalkers, nparams) and the order of parameters along axis 1should be the same as that in grids; vsini going at the end if applicable.

#     create_mcmc_grid Parameters
#     --------------------------- 

#     delta_lambda : float, default: 0.003Å
#         Wavelength resolution for the synthesized grid in Å. Goes unused if a pre-computed grid is supplied. 

#     derived_params : dict, optional
#         Dict of param_name:function entries. If you want to tie any parameter to some combination of parameters in paramgrids (can't include `vsini` or `res` in any way).
#         The function should be wriiten solely as a function of the `SME_Structure` object. Example: `function = lambda s: 1e-4*s.teff + 0.3*s.logg`.
#         Special Case: If you want to use the empirical vmic relation from GALAH DR3 (see Buder et al. 2021), simply pass `vmic='galah'` and it will be set as a function of teff and logg according to the relation in that paper.
#         Note: Goes unused if a pre-computed grid is supplied.

#     linelist : ValdFile or Linelist, optional
#         In case you want to pass a custom linelist at this stage for some reason. Otherwise will use the full linelist that is provided with this package.
#         Goes unused if a pre-computed grid is supplied.
#         Linelists are chopped to the wavelength ranges (plus some tolerance) input previously.

#     save_grids_to : Path-like, optional
#         If provided, the synthesized and parameter grids will be saved to this path in .npz format for future use. 

#     Returns
#     -------
#     sampler : emcee.EnsembleSampler object
#         The MCMC sampler object after running the sampling. The chains can be accessed via `sampler.chain` and the log-probability values via `sampler.lnprobability`.
#         See emcee documentation for more details on the sampler object and how to analyze the chains.
#         Example: `samples = sampler.get_chain(discard=750, thin=1, flat=True)  # remove burn-in and flatten the chains`
#     '''
#     # Checks & Basic Setup
#     if len(sme.FLUX) == 0:
#         raise ValueError("The SME structure has no fit segments defined.")
#     if sme.obswave is None or sme.obsflux is None:
#         raise ValueError("Observed spectrum not set in the SME structure. Please use the `input_observed_spectrum` method to input the observed spectrum before running MCMC.")
#     if sme.NSEG != len(sme.FLUX):
#         raise ValueError("The number of fit segments in the SME structure does not match the number of wave ranges, probably because you overwrote the wave ranges after instancing the fit segments. You should just start over with a fresh SMEwrapper object.")
#     if sme.RES is None:
#         raise ValueError("Resolution MUST be by this stage. And make sure it is array-like that matches the number of segments if you're gonna do it this late.")
#     elif sme.RES is not None:
#         sme.RES = np.array(sme.RES).reshape(-1)
#         if len(sme.RES) != sme.NSEG:
#             if len(sme.RES) == 1:
#                 sme.RES = np.tile(sme.RES, sme.NSEG)
#             else:
#                 raise ValueError("Resolution array length does not match the number of segments.")
#     if vsini_grid is not None and len(vsini_grid)!=2:
#         raise ValueError("`vsini_grid` should be a list or array of [vsini_min, vsini_max]")
#     if isinstance(grids, (str, bytes)) or hasattr(grids, '__fspath__'):
#         with np.load(grids, allow_pickle=True) as data:
#             grids = dict(data)
#         if 'syngrid' not in grids or 'wavegrid' not in grids:
#             raise ValueError("Loaded grid file does not contain 'syngrid' and/or 'wavegrid'. Please check the file.")
#     if set(grids.keys()) & set(derived_params.keys()):
#         raise ValueError("`grids` and `derived_params` share common keys. Parameters cannot be in both.")
#     if 'syngrid' not in grids:
#         if 'wavegrid' in grids:
#             raise ValueError("`wavegrid` present in supplied grids but `syngrid` not found.")
#         grids = create_mcmc_grid(sme, grids, delta_lambda=delta_lambda, derived_params=derived_params, nprocesses=nprocesses, linelist=linelist, return_grid=True, filename=save_grids_to)
    
#     if save_grids_to is not None:
#         np.savez(save_grids_to, **grids)

#     # Modifiers & Housekeeping       
#     syngrid = grids.pop('syngrid')
#     wavegrid = grids.pop('wavegrid')
#     delta_lambda = wavegrid[0][1]-wavegrid[0][0] # AKA delta_lambda
#     if 'vsini' in grids: del grids['vsini']
#     paramgrids = grids
#     # if 'vmic' in derived_params and derived_params['vmic']=='galah':
#     #     derived_params['vmic'] = lambda s: calc_galah_vmic(s.teff,s.logg)
#     # if sme.CS is None: sme.CS = np.ones(sme.NSEG)

#     # Intialize
#     sme.ipres = 0

#     wvgridsts = np.array([arr[0] for arr in wavegrid])
#     wvgridnds = np.array([arr[-1] for arr in wavegrid])
#     ranindices = np.searchsorted(wvgridsts, sme.WRAN[:,0]) - 1
#     if np.any(sme.WRAN[:,1] > wvgridnds):
#         raise ValueError("The current wavelength ranges are not encompassed by the wavegrid. Please check the wavegrid and the fit segments.")
#     gridlengths = np.array([len(arr) for arr in wavegrid])

#     # Star Specific arrays
#     obspec = np.hstack(sme.FLUX)
#     if sme.ERR is None: specerr = 1
#     else: specerr = np.hstack(sme.ERR) if sme.ERR.dtype==object else np.hstack(np.repeat(sme.ERR,[len(arr) for arr in sme.FLUX]))
#     BINE = np.empty(sme.NSEG, dtype='O')
#     for i in range(sme.NSEG):
#         binw = np.diff(sme.WAVE[i])
#         BINE[i] = np.zeros(len(sme.WAVE[i])+1)
#         BINE[i][1:-1] = sme.WAVE[i][:-1]+binw/2
#         BINE[i][0] = sme.WAVE[i][0]-binw[0]/2
#         BINE[i][-1] = sme.WAVE[i][-1]+binw[-1]/2

#     # Make Interpolant
#     spint = RegularGridInterpolator(tuple(paramgrids.values()), syngrid, bounds_error=False, fill_value=0.0)

#     # emcee functions
#     def chisq_log_likelihood(params):
#         if vsini_grid is not None:
#             vsini = params[-1]
#             params = params[:-1]
#         else:
#             vsini = sme.vsini
#         supersyn = spint(params).squeeze()
#         supersyn = np.split(supersyn, np.cumsum(gridlengths)[:-1])
#         synspec = np.empty(sme.NSEG, dtype='O')
#         for i, ig in enumerate(ranindices):
#             synspec[i] = pyasl.fastRotBroad(wavegrid[ig], supersyn[ig], 0.81, vsini)
#             synspec[i] = pyasl.instrBroadGaussFast(wavegrid[ig], synspec[i], sme.RES[i], 'firstlast', maxsig=5)
#             # Rebinning by integrating the flux and then dividing by the bin widths to get average flux in each bin, which is more accurate than interpolating the flux onto the observed grid directly.
#             I = np.zeros_like(synspec[i])
#             I[1:] = np.cumsum(0.5 * (synspec[i][:-1] + synspec[i][1:])) * delta_lambda
#             I_edges = np.interp(BINE[i], wavegrid[ig], I)
#             synspec[i] = np.diff(I_edges)/np.diff(BINE[i]) 
#             synspec[i] = synspec[i] * sme.CS[i] # Continuum scaling
#         synspec = np.hstack(synspec)
#         chisq = np.nansum(((synspec-obspec)/specerr)**2)
#         return -0.5 * chisq
    
#     def log_prior(params):
#         if vsini_grid is not None:
#             vsini = params[-1]
#             params = params[:-1]
#             if not (vsini_grid[0] <= vsini < vsini_grid[1]):
#                 return -np.inf
#         for p, grid in zip(params, paramgrids.values()):
#             if p < grid.min() or p > grid.max():
#                 return -np.inf
#         # User-defined priors
#         if log_prior_function is not None:
#             return log_prior_function(params)
#         return 0
    
#     # Log-posterior = log-prior + log-likelihood
#     def log_posterior(params):
#         lp = log_prior(params)
#         if not np.isfinite(lp):
#             return -np.inf
#         return lp + chisq_log_likelihood(params)
    
#     # Initialize and run the sampler
#     ndim = len(paramgrids) + (1 if vsini_grid is not None else 0)
#     if nwalkers is None:
#         nwalkers = 8*ndim if 8*ndim < 64 else 64
#     if nsteps is None:
#         nsteps = 4500
#     if initial_vector is None:
#         initial_vector = np.random.uniform(low=[grid[0] for grid in paramgrids.values()] + ([vsini_grid[0]] if vsini_grid is not None else []), 
#                                            high=[grid[-1] for grid in paramgrids.values()] + ([vsini_grid[1]] if vsini_grid is not None else []), 
#                                            size=(nwalkers, ndim))
#     if nprocesses <= 1:
#         sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
#         sampler.run_mcmc(initial_vector, nsteps, progress=True)
#     else: 
#         with Pool(processes=nprocesses) as pool:
#             sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
#             sampler.run_mcmc(initial_vector, nsteps, progress=True)

#     return sampler
# # endregion

