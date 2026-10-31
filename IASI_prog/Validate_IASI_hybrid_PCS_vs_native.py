#!/usr/bin/env python
# coding: utf-8
"""
Validate a merged IASI hybrid PCS orbit file against the IASI L1C native file of the same orbit.

usage: python Validate_IASI_hybrid_PCS_vs_native.py <merged orbit .nc> <native .nat> <EV directory> [diagnostics .nc]

Notation, for each band: obs = native spectrum, E = Eigenvectors (orthonormal, R = E @ Nedr),
y = Nedr^-1 obs the noise normalised spectrum. The PCS processing that the checks reproduce is
   global scores        s = round(2 y.E) / 2                          (unpacked P1, P2, P3)
   global residual      x = s @ E - y = Nedr^-1 (s @ R - obs)          (reconstruction - obs)
   quantised residual  xq = round(2 x) / 2                             (unpacked IPO_PccResidual)
   local scores   p_local = round(2 xq.E_local)                        (not unpacked)
so the global + local reconstruction is s @ R - 0.5 * p_local @ R_local (R_local = E_local @ Nedr).
Pixels with the L1C flagged bad for a band (QFlag bits 1-3) are excluded from the spectral checks.

Checks
 1. scan lines, sensing times, geolocation, angles, cloud and land fractions, earth-satellite distance
 2. PCscores/RadianceSum == sum of the native radiances in each band
 3. global PCs: P1, P2, P3 == round(2 y.E) / 2; PCscores/ResidualRms == rms(x)
 4. local PCs: p_local == round(2 xq.E_local); PClocal/rms_local == rms(xq - 0.5 p_local @ E_local)
 5. outliers: IPO_L1cRadiances == native spectrum, IPO_PccResidual == xq, GQisFlagQual == native
    GQisFlagQual, number of flagged outliers == NbrOutliers
 6. statistics of x (global) and of x - 0.5 p_local @ E_local (global + local), not a check

The optional diagnostics file gets the per-pixel rms of x for the global and the global + local
reconstruction and the mean and standard deviation of x per channel over the orbit.
"""
import os
import sys

import numpy as np
import netCDF4 as nc4
import h5py

from iasi_reader_Carlos import NativeIASI1c

BANDS = [(0, 1997), (1997, 5116), (5116, 8461)] # channels of each band
PIXELS = 120

results = []
def check(name, ok, value):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name:62s} {value}")

def max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))

cdr_file, nat_file, ev_dir = sys.argv[1:4]
diag_file = sys.argv[4] if len(sys.argv) > 4 else None

cdr = nc4.Dataset(cdr_file)
cdr.set_auto_mask(False) # packed variables are still unpacked
nat = NativeIASI1c(nat_file)
mdr = nat._mdrs["Level1Data"] # fields without a property in NativeIASI1c

print('CDR    :', cdr_file)
print('native :', nat_file)

#---------------------------------------
# 1. lines, times, geolocation
#---------------------------------------
n_lines = cdr.dimensions['scan_lines'].size
check('number of scan lines', n_lines == len(mdr), f'{n_lines} CDR / {len(mdr)} native')
if n_lines != len(mdr):
    sys.exit(1)

check('SensingTime_day', np.array_equal(cdr['SensingTime_day'][:], mdr['GEPSDatIasi']['Day'][:, 0]), '')
d = max_abs_diff(cdr['SensingTime_msec'][:], mdr['GEPSDatIasi']['MillisecondsOfDay'][:, 0])
check('SensingTime_msec', d == 0, f'max diff {d} ms')

per_pixel = {
    'latitude': nat.latitude, 'longitude': nat.longitude,
    'SatZenith': nat.viewing_zenith_angle, 'SatAzimuth': nat.viewing_azimuth_angle,
    'SunZenith': nat.solar_zenith_angle, 'SunAzimuth': nat.solar_azimuth_angle,
    'CloudFraction': nat.cloud_fraction, 'LandFraction': nat.land_fraction,
}
for name, values in per_pixel.items():
    # CDR pixel = 4 * ifov + pixel in the native file
    d = max_abs_diff(cdr[name][:], values.reshape(n_lines, PIXELS))
    check(name, d == 0, f'max diff {d}')
d = max_abs_diff(cdr['EarthSatDistance'][:], mdr['EARTH_SATELLITE_DISTANCE'])
check('EarthSatDistance', d == 0, f'max diff {d} m')

#---------------------------------------
# 2-6. spectra, one PDU at a time
#---------------------------------------
E, Ninv = [], []
for b in (1, 2, 3):
    with h5py.File(os.path.join(ev_dir, cdr[f'PCscores/Band{b}'].Eigenvectorfile), 'r') as ev:
        E.append(ev['Eigenvectors'][:])
        Ninv.append(np.linalg.inv(ev['Nedr'][:]))

qflag = cdr['QFlag'][:]
good = np.stack([(qflag & (1 << (b - 1))) == 0 for b in (1, 2, 3)], axis=-1) # L1C valid in the band
for b in (1, 2, 3):
    print(f'band {b}: {np.sum(~good[..., b-1])} pixels with L1C flagged bad, excluded from the spectral checks')

rms_glob = np.full((n_lines, PIXELS, 3), np.nan)
rms_loc  = np.full((n_lines, PIXELS, 3), np.nan)
sum_x    = {k: np.zeros(BANDS[-1][1]) for k in ('glob', 'loc')}
sum_x2   = {k: np.zeros(BANDS[-1][1]) for k in ('glob', 'loc')}
n_good   = np.zeros(3)
stats    = {'radsum': 0., 'scores': np.zeros(3), 'scores_diff': np.zeros(3), 'resrms': 0.,
            'p_local': np.zeros(3), 'rms_local': 0.}
out      = {'n': 0, 'n_attr': 0, 'rad': 0., 'res': 0., 'gq': True}

pdu_first_line = cdr['pdu_first_line'][:]
pdu_num_lines  = cdr['pdu_num_lines'][:]
for i, (l0, n) in enumerate(zip(pdu_first_line, pdu_num_lines)):
    lines = slice(l0, l0 + n)
    print(f'PDU {i+1}/{len(pdu_first_line)}: lines {l0}-{l0 + n - 1}  ', end='\r')
    obs = nat.get_spectra(lines).reshape(n, PIXELS, -1).astype(float)

    rs = cdr['PCscores/RadianceSum'][lines]
    rr = cdr['PCscores/ResidualRms'][lines]
    xq_all = np.empty_like(obs)
    for b in (1, 2, 3):
        c0, c1 = BANDS[b-1]
        ok = good[lines, :, b-1]
        n_good[b-1] += ok.sum()
        stats['radsum'] = max(stats['radsum'], np.max(np.abs(rs[ok][:, b-1] / obs[ok][:, c0:c1].sum(-1) - 1)))

        # global PCs
        y = obs[..., c0:c1] @ Ninv[b-1].T
        scores = np.concatenate([cdr[f'PCscores/Band{b}/{k}'][lines] for k in ('P1', 'P2', 'P3')], axis=2)
        scores_est = np.round(2 * y @ E[b-1].T) / 2
        stats['scores'][b-1] += np.sum(np.all(scores == scores_est, axis=-1)[ok])
        stats['scores_diff'][b-1] = max(stats['scores_diff'][b-1], max_abs_diff(scores[ok], scores_est[ok]))
        x = scores @ E[b-1] - y
        rms_glob[lines, :, b-1] = np.where(ok, np.sqrt(np.mean(x**2, axis=-1)), np.nan)
        stats['resrms'] = max(stats['resrms'], np.max(np.abs(rms_glob[lines, :, b-1][ok] / rr[ok][:, b-1] - 1)))
        xq = np.round(2 * x) / 2
        xq_all[..., c0:c1] = xq

        # local PCs
        E_local = cdr[f'PClocal/Band{b}/E_local'][i]
        p_local = cdr[f'PClocal/Band{b}/p_local'][lines].astype(float)
        stats['p_local'][b-1] += np.sum(np.all(p_local == np.round(2 * xq @ E_local.T), axis=-1)[ok])
        rl = cdr[f'PClocal/Band{b}/rms_local'][lines]
        rms_q = np.sqrt(np.mean((xq - 0.5 * p_local @ E_local)**2, axis=-1))
        stats['rms_local'] = max(stats['rms_local'], np.max(np.abs(rms_q[ok] / rl[ok] - 1)))

        # residual of the global + local reconstruction
        x_loc = x - 0.5 * p_local @ E_local
        rms_loc[lines, :, b-1] = np.where(ok, np.sqrt(np.mean(x_loc**2, axis=-1)), np.nan)
        for k, z in (('glob', x), ('loc', x_loc)):
            sum_x[k][c0:c1]  += z[ok].sum(axis=0)
            sum_x2[k][c0:c1] += (z[ok]**2).sum(axis=0)

    # outliers
    ln, px = np.nonzero(cdr['IPO/outlier_flag'][lines])
    out['n'] += len(ln)
    out['n_attr'] += int(cdr['IPO/NbrOutliers'][i])
    if len(ln):
        rad = cdr['IPO/IPO_L1cRadiances'][lines][ln, px]
        out['rad'] = max(out['rad'], np.max(np.abs(rad - obs[ln, px])) / np.abs(obs).max())
        res = cdr['IPO/IPO_PccResidual'][lines][ln, px]
        out['res'] = max(out['res'], max_abs_diff(res, xq_all[ln, px]))
        gq = cdr['IPO/GQisFlagQual'][lines][ln, px].astype(bool)
        out['gq'] &= np.array_equal(gq, mdr['GQisFlagQual'][lines].reshape(n, PIXELS, 3)[ln, px])
print()

check('RadianceSum == sum of native radiances', stats['radsum'] < 1e-5, f"max rel diff {stats['radsum']:.2e}")
for b in (1, 2, 3):
    frac = stats['scores'][b-1] / n_good[b-1]
    check(f'global band {b}: P1,P2,P3 == round(2 y.E)/2', frac > 0.999,
          f"{frac:.4%} of pixels identical, max diff {stats['scores_diff'][b-1]}")
check('global: ResidualRms == rms(x)', stats['resrms'] < 1e-3, f"max rel diff {stats['resrms']:.2e}")
for b in (1, 2, 3):
    frac = stats['p_local'][b-1] / n_good[b-1]
    check(f'local band {b}: p_local == round(2 xq.E_local)', frac > 0.999, f'{frac:.4%} of pixels identical')
check('local: rms_local == rms(xq - 0.5 p_local E_local)', stats['rms_local'] < 1e-3,
      f"max rel diff {stats['rms_local']:.2e}")

check('outliers: outlier_flag count == NbrOutliers', out['n'] == out['n_attr'], f"{out['n']} / {out['n_attr']}")
if out['n']:
    check('outliers: IPO_L1cRadiances == native spectrum', out['rad'] < 1e-6, f"max rel diff {out['rad']:.2e}")
    check('outliers: IPO_PccResidual == xq', out['res'] == 0, f"max diff {out['res']}")
    check('outliers: GQisFlagQual == native', out['gq'], '')

#---------------------------------------
# statistics of the noise normalised residual
#---------------------------------------
print('\nnoise normalised residual Nedr^-1 (reconstruction - native), valid L1C pixels')
print(f"{'':6s} {'reconstruction':16s} {'mean rms(x)':>12s} {'mean bias':>10s} {'max |bias|':>11s}")
for b in (1, 2, 3):
    c0, c1 = BANDS[b-1]
    for k, rms, name in (('glob', rms_glob, 'global'), ('loc', rms_loc, 'global + local')):
        bias = sum_x[k][c0:c1] / n_good[b-1]
        print(f'band {b} {name:16s} {np.nanmean(rms[..., b-1]):12.4f} {bias.mean():10.4f} {np.abs(bias).max():11.4f}')

if diag_file:
    with nc4.Dataset(diag_file, 'w') as f:
        f.createDimension('scan_lines', n_lines)
        f.createDimension('pixels', PIXELS)
        f.createDimension('band', 3)
        f.createDimension('num_chans', BANDS[-1][1])
        f.description = 'Validation of ' + os.path.basename(cdr_file) + ' against ' + os.path.basename(nat_file) + \
                        ': x = Nedr^-1 (reconstruction - native) is the noise normalised residual'
        for name, data, desc in (('rms_global', rms_glob, 'rms of x, global PCs'),
                                 ('rms_global_local', rms_loc, 'rms of x, global + local PCs')):
            v = f.createVariable(name, 'f4', ['scan_lines', 'pixels', 'band'], zlib=True)
            v.description = desc
            v[:] = data
        n_chan = np.repeat(n_good, [c1 - c0 for c0, c1 in BANDS])
        for k, name in (('glob', 'global'), ('loc', 'global_local')):
            mean = sum_x[k] / n_chan
            v = f.createVariable('mean_x_' + name, 'f4', ['num_chans'])
            v.description = 'mean of x over the orbit, ' + name.replace('_', ' + ') + ' PCs'
            v[:] = mean
            v = f.createVariable('std_x_' + name, 'f4', ['num_chans'])
            v.description = 'standard deviation of x over the orbit, ' + name.replace('_', ' + ') + ' PCs'
            v[:] = np.sqrt(sum_x2[k] / n_chan - mean**2)
    print('diagnostics written to', diag_file)

n_fail = sum(not ok for _, ok in results)
print(f'\n{len(results) - n_fail} checks passed, {n_fail} failed')
sys.exit(1 if n_fail else 0)
