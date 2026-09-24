# pysme_wrapper: design notes (as of 2026-09-25, PySME 0.4.188)

This file records **why** `pysme_wrapper` is built the way it is. It is written for a fresh Claude session that will fork this repo and port it to PySME 1.2. Read it before changing anything.

The main risk in the port is that PySME 1.2 fixes some of the problems below, breaks others, and adds new ones. The last attempt to upgrade PySME was full of bugs and was rolled back. So **treat every claim about PySME behaviour here as something to re-test on 1.2**, not something to assume. Section 7 gives probe recipes for doing that.

- Code: `src/pysme_wrapper/core.py` (almost everything) and `src/pysme_wrapper/utils.py` (range helpers, the bundled linelist, and `initsme`).
- Workflow reference: `pysme_wrapper_tutorial.ipynb`. It was written before the MCMC API change of 2026-09-25, so its MCMC cells use the old calls (see section 4).
- Environments on this machine:

| conda env | PySME | Python | notes |
|---|---|---|---|
| `sme` | 0.4.188 | 3.12 | the working env; has telfit, PyAstronomy 0.22, emcee 3.1.6 |
| `main` | 1.0.1 | 3.12 | no telfit, so `pysme_wrapper` does not import there |
| `pysme` | 1.0.1 | 3.14 | |

---

## 1. The two purposes

1. **MCMC without re-synthesizing at every step.** Running PySME per likelihood evaluation is far too slow for emcee (~10⁵–10⁶ evaluations). Instead:
   - synthesize a grid over the stellar parameters **once**, and save it;
   - during MCMC, only interpolate that grid and apply the cheap parameter-dependent operations (broadening, binning onto the observed pixels);
   - one grid can be reused across stars that share wavelength windows.
2. **Work around the limitations and bugs of PySME 0.4.188**, mainly to make each synthesis faster, and to get control over the "fit segments" (RV, continuum, errors), which PySME handled badly.

Guiding style: the wrapper does the data preparation that PySME does badly (segmenting, RV, continuum, errors). It calls PySME only for what PySME is good at: radiative transfer on a wavelength range you give it, with a linelist already cut to that range.

---

## 2. PySME 0.4.188 limitations, and what the wrapper does about each

| # | Limitation (user's account) | Verified 2026-09-25? | Wrapper workaround | Where |
|---|---|---|---|---|
| L1 | Every segment is processed against the **entire linelist**, so N segments cost almost N full syntheses. | **Yes.** With the full 61k-line list: 1 segment 40 s, 13 segments 190 s. With a linelist cut to the windows (276 lines): 0.34 s and 1.34 s. | (a) **Always cut the linelist** to the (padded) wavelength ranges before synthesizing. (b) **Synthesize as few PySME segments as possible.** Usually that means one segment covering all ranges, from which the pieces are sliced out afterwards. | `fast_synthesize`, `get_error_and_cscale` and `fit_RV` all cut the linelist. `fast_synthesize` and `get_error_and_cscale` use a single `sme.wave` covering everything. `create_mcmc_grid` passes all windows as **one gapped segment** (see section 4). |
| L2 | With one resolution per segment (`ipres` list), only the **first value** was used for every segment. | **Not reproduced** through `synthesize_spectrum` in 0.4.188: segment-wise `ipres=[20000, 80000]` gave different broadening per segment. 0.4.188 does contain `ipres = sme.ipres if size==1 else sme.ipres[segment]`. The bug may be in `solve()` or another path; still to be checked. Also, 0.4.188's Gaussian uses `hwhm = 0.5*x_seg[0]/ipres`, the segment's *first* wavelength rather than its centre. | The wrapper never relies on PySME for per-segment resolution. `fast_synthesize` synthesizes with `ipres=0` and broadens each range itself (`pyasl.instrBroadGaussFast`). The MCMC applies its own instrumental matrix, and now supports per-pixel R. | `fast_synthesize`, `MCMCsetup` |
| L3 | Little freedom in handling fit segments. RV shifts were confusing: `vrad` shifts the *model* onto the observed frame, with `vrad_flag` modes `each`/`whole`/`fix`/`none`. | – | **The observed spectrum is shifted once, per segment, into the stellar rest frame** (`WAVE = obswave*(1-RV/c)`). All synthesis then happens in the rest frame with `vrad=0` and `vrad_flag='none'`. | `make_fit_segments`, `add_fit_segments` |
| L4 | **No routine to measure local errors.** | – | `get_error_and_cscale` measures the rms of flux/continuum over continuum pixels in the central quantile window around each segment. | `get_error_and_cscale` |
| L5 | **Continuum scaling and RV were fitted in one pass** (`match_rv_continuum`). The continuum scaling misbehaved, especially at low S/N. | – | These are separated. RV is fitted first (`fit_RV`, cross-correlation-like `solve` for `vrad` only, over a wide window). Continuum is fitted afterwards on RV-corrected data, as a smoothing spline over continuum pixels. PySME's own continuum and RV fitting is always switched off (`cscale_flag='none'`, `vrad_flag='none'`). | `fit_RV`, `get_error_and_cscale`, `make_fit_segments` |
| L6 | The NLTE part tried to reuse a previous grid and failed. | – | `sme.nlte.grid_data = {}` is reset after or before syntheses (`fast_synthesize(reuse_nlte_grid=False)`, `fit_RV`, `get_error_and_cscale`). | several |
| L7 | PySME's internal arrays go stale (`synth`, `cont`, `mask`, `uncs`, `central_depth`, `line_range` still attached after `wave` changes), causing shape errors. | – | `SMEwrapper.__getattribute__` returns `None` for these when their shapes no longer match `wave`/`linelist`. It builds `mask`/`uncs` on the fly. | `SMEwrapper.__getattribute__` |

The pattern that runs through everything is **transient PySME state**. Each method sets `sme.wave`, `sme.spec`, `sme.linelist`, `sme.ipres`, `sme.vsini` and so on just for its own synthesis, then **restores or clears them** (`linelist=None`, `wave=None`, `wran=None`, cached values put back). Keep this discipline, or methods will silently leak state into each other.

---

## 3. Core concepts and conventions

- **Fit segments.** Small windows, typically ±0.5 Å for FEROS, and generally a few Å at most, each around one line or a blend. Each segment carries its own RV, continuum scale (CS), error (ERR) and resolution (RES). Segments are chosen to be deep and uncontaminated (see the line-selection guide in the tutorial).
- **UPPER vs lower case attributes.**
  - UPPERCASE attributes belong to the wrapper and persist: `WRAN` (N×2 ranges; the setter demands sorted and non-overlapping), `NSEG`, `WAVE`, `FLUX`, `ERR`, `CS`, `RV`, `RES`, `CSEG` (indices into `obswave`).
  - `obswave`, `obsflux`, `obserr`, `obsres` and `obstelluric` hold the full observed spectrum.
  - lowercase attributes (`wran`, `wave`, `spec`, `synth`, `ipres`, `vrad`, …) are PySME's own fields and are only used transiently.
- **Object arrays.** Per-segment quantities are 1-D `dtype=object` arrays of arrays (`_objarray`). `_objarray` assigns item by item, so segments of equal length are not broadcast into a 2-D array.
- **Rest frame everywhere.** `WAVE` is already rest-frame. Grids and models are rest-frame.
- **CS** can be scalars, per-pixel arrays, or `scipy.interpolate.BSpline` objects defined in the rest frame. Consumers evaluate them on `WAVE[i]`.
- **ERR** is a *relative* error (flux is normalized), one value per segment by default.
- **Abundances.** `SMEwrapper.__init__` sets `abund['Fe'] = 7.38` (GALAH DR3 zero point).
  - `_create_spectrum` has a TODO about PySME's "auto-monh correction quirk" in abundances. PySME couples `monh` and `abund`; check how 1.2 treats this before trusting `monh`/`abund X` grids.
- **Linelist.** `utils.vald` is the bundled `linelist_solar_0.001_350-900nm` (VALD, depth ≥ 0.001), with an added `.element` column. It is loaded at import time, which is slow. `SMEwrapper(fulllinelist=vald)` keeps the full list, and methods cut from it.
- **Tellurics.** Handled with telfit (`get_telluric_transmission`). Pixels with transmission below `nan_thresh` are set to NaN in `obsflux`. NaNs are then skipped downstream.

---

## 4. Workflow and module map

The typical pipeline, as in the tutorial:

1. **Setup:** `SMEwrapper(teff, logg, …)`, then `input_observed_spectrum(wave, flux, err, res)`, then optionally `get_telluric_transmission(...)`.
2. **Line selection:** `fast_synthesize` on candidate lines and on the nearby contaminants, keeping deep, clean lines. This uses `create_ranges`, `combine_ranges` and `inranges` from utils.
3. `WRAN = create_ranges(centres, halfspan, join=True)`.
4. **Build the fit segments:** `make_fit_segments(RES, RV='fit', CS='fit', ERR='fit' | 'propagate', make_quality_cuts=True)`.
   - `fit_RV` runs PySME `solve(sme, ['vrad'])` with `vrad_flag='each'` on windows of `window_size` (40 Å) around each segment, using lines of depth > 0.1. vsini is set to 0 by default for sharper lines.
   - `get_error_and_cscale`:
     1. synthesize one flat model over 60 Å windows (a single `sme.wave`);
     2. mask points where model × telluric < 0.98;
     3. fit `make_smoothing_spline(flux/model, lam=10)` to the remaining continuum pixels; the result is the CS BSpline;
     4. ERR = std of flux/CS over continuum pixels in the [0.3, 0.7] quantile of each window.
   - Quality cuts drop segments with outlying RV (2σ or more than 2 km/s from the mean), high ERR (3σ), or a mean CS outside 0.8–1.2.
5. **Manage segments:** `save_fit_segments`, `load_fit_segments`, `delete_fit_segments`, `add_fit_segments`, and `user_defined_segments` (for simulations or externally prepared segments).
6. **MCMC:** `create_mcmc_grid(...)`, then `MCMCsetup(sme, grid, param_bounds=...)`, then `run_mcmc()`.

Other helpers:
- `fast_synthesize(sme, wave_ranges, resolutions, delta_lambda)`: one synthesis over all ranges (a single `sme.wave`, linelist cut to the padded ranges), then slicing. With per-range resolutions it broadens with pyasl itself.
- `utils.initsme(...)`: builds a plain `SME_Structure` with NLTE grids `nlte_{elem}_pysme.grd` for H, Mg, Fe, Ca, Ti, Si and Ba.
- `utils.calc_galah_vmic(teff, logg)`: the GALAH DR3 vmic relation, used by `derived_params={'vmic': 'galah'}`.

### MCMC design (current, rewritten 2026-09-25)

**Grid: `create_mcmc_grid`**
- For each grid node it stores the **specific intensities at each μ** (`sme.mu`, 7 angles), not the flux. Each μ is stored as that annulus's contribution to the normalized flux, `wt_j·I(μ_j)/F_cont`.
- The wavelength grid is **log-spaced** (`delta_v`, default 0.15 km/s), in padded windows: `max_vbroad` + 5σ_instr + margin.
- The radiative transfer is **forced onto exactly these points**: pre-fill `Synthesizer.wint = {0: grid}` and call with `reuse_wavelength_grid=True`, with `specific_intensities_only=True`, `normalize_by_continuum=False`, and vsini = vmac = ipres = 0.
  - This bypasses PySME's adaptive grid, so `accwi` plays no part, and no wavelength interpolation happens anywhere.
  - It costs ~3× a default synthesis per Å.
  - All windows are passed as **one gapped segment** (because of L1). This was verified bit-identical to one segment per window, and ~12% faster with a cut linelist.
- The `.npz` keys are `wavegrid`, `syngrid` (shape `(*param_shape, nmu, Nwave)`, float16), `mu`, `delta_v`, plus the parameter grids.
- `vsini` and `vmac` are **not** grid parameters.

**Why intensities, and why log-λ**
- PySME itself broadens on a constant-velocity-step grid (`Synthesizer.new_wavelength_grid`) by **disk integration over μ** (`integrate_flux`: annulus rotation kernels × radial-tangential macroturbulence). Storing intensities lets the MCMC reproduce that exactly, for any vsini and vmac, without re-synthesizing, and with no limb-darkening law.
- The old design (a flux grid on 0.003 Å linear λ, plus `pyasl.fastRotBroad` with ε = 0.81) was off by up to 1% in flux at vsini 20–26 km/s, compared with disk integration.

**`MCMCsetup`**
- **Restricting and fixing parameters:** `param_bounds={name: (lo, hi) | value}` for grid parameters and for `vsini`/`vmac`.
  - A restricted grid axis is sliced to the bracketing nodes.
  - A fixed grid axis is collapsed by linear interpolation, which gives exactly the same result as multilinear interpolation at that value.
- **Star-specific precomputation at setup:**
  - each fit segment is cropped to the grid pixels it needs (with a coverage check against the vsini/vmac bounds);
  - one sparse matrix does the per-pixel-R Gaussian LSF, integration over each observed bin, and CS, all at once; χ² folds in 1/err.
- **Per call:**
  1. lean multilinear interpolation of the μ-intensities;
  2. FFT disk integration (annulus kernels, oversampled and binned, times the analytic Fourier transform of the macroturbulence Gaussian);
  3. one sparse multiply.
- If vsini and vmac are both fixed, the whole model is precomputed per node, and a call is interpolation only.
- **Parallel:** a Pool initializer hands the object to the workers **once** (with the fork context). emcee used to pickle the whole grid with every task, which made `nprocesses=8` ~35× slower than serial.
- **Regression targets** (synthetic test, scratch script from 2026-09-25):
  - model against PySME `synthesize_spectrum` (vsini 12, vmac 3, R 50k, bin-averaged): max 1.7e-4;
  - unbroadened grid against PySME: max 4.5e-4 (float16 storage);
  - fixed and restricted parameters against free: ≤2e-6 relative;
  - speed: ~1.3 ms per call with 4 free parameters and 17k grid pixels; 0.012 ms with broadening fixed.

---

## 5. Invariants worth keeping in any rewrite

1. Synthesis always gets a linelist **cut** to the padded ranges.
2. Keep the number of PySME segments minimal, unless 1.2 is shown to fix L1.
3. Observations are shifted to the rest frame per segment, and PySME never fits RV or continuum inside a synthesis used for fitting.
4. RV, continuum and error are separate, inspectable steps with debug outputs (`return_arrays`, `debug_mode`).
5. MCMC never calls PySME. Everything parameter-independent is precomputed. The grid on disk is star-independent: the star-specific parts (R, CS, err, pixels) are applied in `MCMCsetup`.
6. Clean up transient PySME state after every method.

---

## 6. Known rough edges in the current code (fix or at least don't copy)

- `get_error_and_cscale(cscale_mode='window_quantile_mean')` uses `csegcont`, which is only defined in the `segment_mean` branch, so it raises a `NameError`.
- `make_fit_segments(..., ERR='none')` then crashes in the quality cuts, because they read `obj.ERR.dtype` on `None`.
- `add_fit_segments(RES=None)` reaches `len(RES)` and crashes.
- `make_fit_segments` reduces a per-pixel `obsres` to **one mean per segment**. `MCMCsetup` accepts per-pixel R (an array matching `WAVE[i]`), but you have to set `sme.RES` yourself.
- The CS BSpline comes from a smoothing spline over 60 Å windows. `lam=10` was tuned for FEROS (R ≈ 48,000); other instruments may need retuning.
- The speed of light is written as 299792.5 in the segment code and 299792.458 in the MCMC code.
- `create_mcmc_grid` relies on PySME 0.4.x internals (`Synthesizer.wint` + `reuse_wavelength_grid`). If PySME ignores the supplied grid, it raises a `RuntimeError`. PySME 1.x exposes `sme.wint` for the same purpose.

---

## 7. Porting to 1.2: what to check before trusting it

For each limitation, run a small probe on 1.2 **before** deleting the corresponding workaround. Probes used on 0.4.188:

- **L1, linelist cost per segment:** time `synthesize_spectrum` on 13 small windows as 1 segment against 13 segments, with the full `vald` list and with a cut one. If 1.2 chunks the linelist per segment (0.4.188 has commented-out `linelist_mode == 'chunk'` code, so this may now exist), the gapped-single-segment and cutting workarounds become optional.
- **L2, per-segment `ipres`:** 2 segments, `ipres=[20000, 80000]`; check that each segment's line depth matches a single-segment run at that R. Test **both** `synthesize_spectrum` and `solve`. Also check whether the kernel width uses the segment start or the centre.
- **L5, continuum at low S/N:** compare PySME's own `cscale` fitting with `get_error_and_cscale` on a noisy FEROS segment (for example tau Ceti in this repo, with added noise).
- **Forced wavelength grid (MCMC):** confirm there is a supported way to make the radiative transfer run exactly on a user grid (`sme.wint` in 1.x), and that `specific_intensities_only=True` still returns `(wmod, smod[nmu, n], cmod[nmu, n], ...)`.
- **Disk integration:** confirm 1.2's `integrate_flux` still uses the same annulus geometry: r from μ midpoints, `wt = Δr²`, radial-tangential macroturbulence with σ = vmac·μ/√2 and vmac·√(1−μ²)/√2, and an overall factor π. `MCMCsetup._rot_ft`/`_mac_ft` mirror it; `_disk_annuli` mirrors the geometry.
- **NLTE reuse (L6)** and **stale arrays (L7):** synthesize twice with different `wave`, and with and without NLTE; see whether the resets and `__getattribute__` guards are still needed.
- **Regression:** re-run the MCMC regression test (build a small grid, compare `MCMCsetup.model_spectrum` with a full PySME synthesis with vsini, vmac and ipres). It should reproduce the targets in section 4.

Parts that might become redundant in 1.2 if the probes pass:
- the stale-array guards (L7);
- the NLTE reset (L6);
- per-range broadening in `fast_synthesize` (L2);
- possibly the single-segment tricks (L1).

The parts that are the wrapper's own value will not become redundant: separate RV, continuum and error preparation; the fit-segment bookkeeping; and the MCMC grid machinery.
