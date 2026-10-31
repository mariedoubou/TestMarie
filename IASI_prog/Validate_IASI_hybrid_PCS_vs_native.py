#!/usr/bin/env python
# coding: utf-8
"""
Validate a merged IASI hybrid PCS orbit file against the IASI L1C native file of the same orbit.

usage: python Validate_IASI_hybrid_PCS_vs_native.py <merged orbit .nc> <native .nat> <EV directory> [diagnostics .nc]
              [--line L] [--ifov F] [--pixel P] [--plot-dir D] [--show]

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
 7. plots of the native, global and global + local spectra of one pixel (scan line L, by default the
    valid pixel closest to the equator, field of view
    F = 0-29, pixel P = 0-3) in radiance and in brightness temperature, saved in D as PNG files

The optional diagnostics file gets the per-pixel rms of x for the global and the global + local
reconstruction and the mean and standard deviation of x per channel over the orbit.
"""
import os
import sys
import argparse

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

parser = argparse.ArgumentParser(description='Validate a merged IASI hybrid PCS orbit file against the native L1C')
parser.add_argument('cdr_file', help='merged orbit netCDF')
parser.add_argument('nat_file', help='IASI L1C native file of the same orbit')
parser.add_argument('ev_dir', help='directory of the EV files')
parser.add_argument('diag_file', nargs='?', default=None, help='diagnostics netCDF to write')
parser.add_argument('--line', type=int, default=None,
                    help='scan line of the plotted spectra (0-based), default: the valid pixel closest to the equator')
parser.add_argument('--ifov', type=int, default=14, help='field of view of the plotted spectra (0-29)')
parser.add_argument('--pixel', type=int, default=0, help='pixel of the plotted spectra (0-3)')
parser.add_argument('--plot-dir', default='.', help='directory of the PNG files')
parser.add_argument('--show', action='store_true', help='also show the plots on screen')
args = parser.parse_args()
cdr_file, nat_file, ev_dir, diag_file = args.cdr_file, args.nat_file, args.ev_dir, args.diag_file

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
E, Ninv, R = [], [], []
for b in (1, 2, 3):
    with h5py.File(os.path.join(ev_dir, cdr[f'PCscores/Band{b}'].Eigenvectorfile), 'r') as ev:
        E.append(ev['Eigenvectors'][:])
        Ninv.append(np.linalg.inv(ev['Nedr'][:]))
        R.append(ev['ReconstructionOperator'][:]) # E @ Nedr

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

#---------------------------------------
# 7. spectra of one pixel: native, global and global + local reconstructions
#---------------------------------------
def rad_to_bt(rad, wn):
    # rad in mW/m2/sr/cm-1, wn in cm-1, BT in K (NaN where the radiance is not positive)
    c1 = 1.1910427e-5
    c2 = 1.4387752
    with np.errstate(divide='ignore', invalid='ignore'):
        return c2 * wn / np.log(c1 * wn**3 / rad + 1.)

pix = 4 * args.ifov + args.pixel # CDR pixel = 4 * ifov + pixel
if args.line is None:
    abs_lat = np.where(good[:, pix].all(axis=-1), np.abs(cdr['latitude'][:, pix]), np.inf)
    line = int(np.argmin(abs_lat))
else:
    line = args.line
ipdu = cdr['pdu_index'][line]
wn = nat.wavenumbers # cm-1
obs = nat.get_spectra(slice(line, line + 1)).reshape(PIXELS, -1)[pix].astype(float)
spec_glob = np.empty_like(obs)
spec_loc  = np.empty_like(obs)
for b in (1, 2, 3):
    c0, c1 = BANDS[b-1]
    scores = np.concatenate([cdr[f'PCscores/Band{b}/{k}'][line, pix] for k in ('P1', 'P2', 'P3')])
    p_local = cdr[f'PClocal/Band{b}/p_local'][line, pix].astype(float)
    R_local = cdr[f'PClocal/Band{b}/R_local'][ipdu]
    spec_glob[c0:c1] = scores @ R[b-1]
    spec_loc[c0:c1]  = spec_glob[c0:c1] - 0.5 * p_local @ R_local # p_local are the scores of reconstruction - obs

lat, lon = cdr['latitude'][line, pix], cdr['longitude'][line, pix]
time = np.datetime64('2000-01-01') + np.timedelta64(int(cdr['SensingTime_day'][line]), 'D') \
       + np.timedelta64(int(cdr['SensingTime_msec'][line]), 'ms')
title = (f'{str(time)[:19]}  line {line}  ifov {args.ifov}  pixel {args.pixel}'
         f'  (lat {lat:.2f}, lon {lon:.2f})')
if not good[line, pix].all():
    title += f'\nL1C flagged bad in band(s) {list(np.nonzero(~good[line, pix])[0] + 1)}'
if cdr['IPO/outlier_flag'][line, pix]:
    title += '  (PCS outlier)'

import matplotlib
if not args.show:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

rad = {k: v * 1.e5 for k, v in (('native', obs), ('global', spec_glob), ('global + local', spec_loc))} # mW/m2/sr/cm-1
for unit, label, spectra in (('rad', 'Radiance (mW/m$^2$/sr/cm$^{-1}$)', rad),
                             ('BT', 'Brightness temperature (K)', {k: rad_to_bt(v, wn) for k, v in rad.items()})):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8), height_ratios=[2, 1])
    ax1.plot(wn, spectra['native'], color='blue', linewidth=0.5, label='native')
    ax1.plot(wn, spectra['global'], color='orange', linewidth=0.5, label='reconstructed, global')
    ax1.plot(wn, spectra['global + local'], color='green', linewidth=0.5, label='reconstructed, global + local')
    ax1.set_ylabel(label)
    ax1.set_title(title)
    ax1.legend(loc='best')
    diffs = [spectra['native'] - spectra[k] for k in ('global', 'global + local')]
    ax2.plot(wn, diffs[0], color='orange', linewidth=1.0, label='native - global')
    ax2.plot(wn, diffs[1], color='green', linewidth=0.5, label='native - (global + local)')
    ax2.plot(wn, spectra['global + local'] - spectra['global'], color='red', linewidth=0.8,
             label='local correction: (global + local) - global')
    lim = 1.1 * np.nanpercentile(np.abs(np.concatenate(diffs)), 99.9) # BT differences explode where the radiance is ~0
    ax2.set_ylim(-lim, lim)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('Difference' + (' (K)' if unit == 'BT' else ''))
    ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax2.set_xlim(wn[0], wn[-1])
    ax2.legend(loc='best')
    for ax in (ax1, ax2):
        for c0, _ in BANDS[1:]:
            ax.axvline(wn[c0], color='grey', linestyle=':', linewidth=0.8)
    fig.tight_layout()
    png = os.path.join(args.plot_dir, f'spectrum_{unit}_line{line}_ifov{args.ifov}_pixel{args.pixel}.png')
    fig.savefig(png, dpi=150)
    print('plot written to', png)
if args.show:
    plt.show()

n_fail = sum(not ok for _, ok in results)
print(f'\n{len(results) - n_fail} checks passed, {n_fail} failed')
sys.exit(1 if n_fail else 0)
