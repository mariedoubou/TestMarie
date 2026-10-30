#!/usr/bin/env python
# coding: utf-8
-----------------------------------------------------------------------------
name:			convert_IASI_PCS_2_CDR
description: 	Convert IASI PCS from processing output to CDR version for release.
project:        cdr_converters

2022-10-13 v1.0		Roger Huckle initial implementation
2023-08-30 v1.1		Roger Huckle corrected file name to L1 and added extension end date
2023-10-05 v1.2		Roger Huckle removed L1C group
2026-07-16          Marie - Jupyter notebook + intro of local PC
----------------------------------------
import os
import sys
import pathlib

import argparse
import datetime as dt
import tempfile
import shutil

import numpy as np
import netCDF4 as nc4
import h5py

import matplotlib.dates as dates
import glob

#import Routines_Convert_IASI_PCS_R2_to_nc as iasi
import Routines_Convert_IASI_hybrid_PCS as iasi # this is in mynotebooks.It contains the routine
import importlib

import xarray as xr

#importlib.reload(iasi) # to reload if needed

# -------------------- DICTIONARIEs -----------------------
sat_dic = {
    'M01': 'METOPB',
    'M02': 'METOPA',
    'M03': 'METOPC'
}

ev_dic = { 'M03_B1': 'IASI_EV1_xx_M03_20180101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx',
           'M03_B2': 'IASI_EV2_xx_M03_20180101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx',
           'M03_B3': 'IASI_EV3_xx_M03_20180101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx'
    
}

# # read IASI orbits for one day

# ### checking the reprocessing directories for a specific date

an    = '2022' #'2019'
mois  = '01'   #'12'
jour  = '15'   #'25'

sat  = 'M03'

repro_dir = '/DSNNAS2/Repro/processing/iasi/hybrid_PCS/r1_output/'+sat+'/'+an+'/'+mois+'/'+jour+'/'

print(repro_dir+'IASI_xxx_1C_'+sat+'_'+an+mois+jour+'*')

dirout = '/DSNNAS2/Repro/users/marieb/IASI_PCS/'+jour+'_'+mois+'_'+an+'_'+sat

### create the output directory if not existing

direc = pathlib.Path(dirout)
direc.mkdir(parents=True, exist_ok=True)

# look for directories of all orbits of the day
dir_orbit = glob.glob(repro_dir+'*Z')
dir_orbit.sort()
print(len(dir_orbit),dir_orbit)
print('dir_orbit:',len(dir_orbit),dir_orbit[0])

for n,oo in enumerate(dir_orbit):
    print(n,os.path.basename(oo))

# ## chose one orbit to process orb = x

# chose one orbut to create the netcdf file
orb = 12
print(dir_orbit[orb])

# create the orbit directory
dirout =  '/DSNNAS2/Repro/users/marieb/IASI_PCS/'+jour+'_'+mois+'_'+an+'_'+sat+'/'+os.path.basename(dir_orbit[orb])

print(dirout)

#create the directory for output if does not exist
direc = pathlib.Path(dirout)
direc.mkdir(parents=True, exist_ok=True)

# look for pdus in one orbit: dir_pdu
dir_pdu = glob.glob(dir_orbit[orb]+'/WO*')
dir_pdu.sort()
print('we have ',len(dir_pdu),'pdus in ',os.path.basename(dir_orbit[orb]))


# ### read directly in Kristina outputs

for n,pdu in enumerate(dir_pdu):
    print(n,os.path.basename(pdu))
print('dirout:',dirout)

# ## create the netcdf for each 3 minute PDUs of one orbit

last_scanline = 0 # to compute which line we are in

#create_IASI_PCS_netCDF(nfile, nc, scan_lines, f_sat)
for n,pdu in enumerate(dir_pdu):
    print(n,os.path.basename(pdu))
    IPO = 'no' # initialise to 'no'
    sat = 'M03'  # use IASI-A
    SAT = {'M02': 'A', 'M01': 'B', 'M03': 'C'}
    
    dir_to_look = pdu
    #day = (dt.datetime.today() - dt.timedelta(days=1)).strftime("%Y%m%d")
    
    prpdir = dir_to_look+'/to_pgf/*PRP*Z'
    ipodir = dir_to_look+'/to_pgf/*IPO*Z'
    pchdir = dir_to_look+'/to_pgf/*PCH*Z'
    
    outputTmpdir = dirout

    prp = glob.glob(prpdir)
    pch = glob.glob(pchdir)
    ipo = glob.glob(ipodir)
    prp.sort()
    pch.sort()
    ipo.sort()
    print('len(prp)=',len(prp)) # should be only one prp, pch and ipo file
    print('len(pch)=',len(pch))
    print('len(ipo=',len(ipo))
    
    
    ifile = 0 
    # choose one PDU from the list of files above
    fprp = prp[ifile]
    fpch = pch[ifile]
    fipo = ipo[ifile]
    
    # check that there are variables in IPO
    if os.path.getsize(fipo) > 76000:
        print('size of IPO=', os.path.getsize(fipo))
        IPO = 'yes'
    print('IPO=',IPO)
        
    ### open the 3 files PRP, PCH and IPO
    nc    = nc4.Dataset(fprp, "r")
    ncpch = nc4.Dataset(fpch, "r")
    ncipo = nc4.Dataset(fipo, "r")
        
    scan_lines = len(nc['L1C/LineNumber'][:])
    print('scan_lines=',scan_lines)
    
    print('dir_pdu[pdu]=',pdu)

    # create a new filename
    fmt  = '%Y%m%d%H%M%SZ'
    basename = os.path.basename(fprp)
    dirname  = os.path.dirname(fprp)
    time_now = dt.datetime.now()
    print(time_now)
    print(dt.datetime.strftime(time_now, fmt))
    newname = dirout + '/IASI_hybrid_PCS_'+basename[9:47]+'_R_O_'+dt.datetime.strftime(time_now, fmt)+'.nc'
    print(newname)
    print(os.path.basename(newname))

    #open the new netcdf file to be created 
    nfile = nc4.Dataset(newname, "w", format='NETCDF4')

    f_sat    = sat
    #numlines = 0
    last_scanline = last_scanline + scan_lines #+ numlines

    print('scan_lines=',scan_lines, ' last_scanline=',last_scanline)
    iasi.create_IASI_hybrid_PCS_netCDF(nfile, nc, ncpch, ncipo, scan_lines, last_scanline, sat, IPO)

    # prepare some variables for the metadata creation
    s_start_t = os.path.basename(newname)[23:37]
    s_end_t   = os.path.basename(newname)[39:53]
    fmt1 = '%Y%m%d%H%M%S'
    proc_time = dt.datetime.strftime(time_now, fmt1)
    fnew      = os.path.basename(newname)

    print('s_start_t :',s_start_t)
    print('s_end_t   :',s_end_t)
    print('proc_time :',proc_time)
    print('fnew      :',fnew)
    print('newname   :',newname)

    parent = nc.groups["L1C"].ParentProduct
    orbit  = nc.groups["L1C"].orbit
    print('parent, orbit : ',parent,orbit)

    iasi.Write_Metadata_NC(nfile, s_start_t, s_end_t, f_sat, 
                      os.path.basename(fnew), int(orbit), int(orbit)+1, 
                      proc_time, parent)
    # close files
    nfile.close()
    nc.close()
    ncpch.close()
    ncipo.close()

# ## merged files that works

print('dir_orbir = ',dir_orbit[orb])

print('dirout=',dirout)

print('sat=',sat)

list_file = glob.glob(dirout+'/IASI_hybrid_PCS_1C_'+os.path.basename(dir_orbit[orb])[12:24]+'*.nc')
list_file.sort()

print('list_file=',list_file)

print(len(list_file))

print(os.path.basename(dir_orbit[orb]))

outfile = dirout+'/merged_'+os.path.basename(dir_orbit[orb])+'.nc'
print(outfile)

#outfile = dir_orbit[orb]+'/merged_'+os.path.basename(dir_orbit[orb])+'.nc'
outfile = dirout+'/merged_'+os.path.basename(dir_orbit[orb])+'.nc'
CONCAT_DIM = 'scan_lines'
CHUNK = 10  # process 10 scan_lines at a time to stay low on memory
#SKIP_GROUPS = {'IPO'}  # groups to exclude from concat


def create_dimensions(src_root, dst_root):
    """Create all dimensions at root level (where they are defined)."""
    for name, dim in src_root.dimensions.items():
        if name not in dst_root.dimensions:
            if name == CONCAT_DIM:
                dst_root.createDimension(name, None)  # unlimited
            else:
                dst_root.createDimension(name, len(dim))

def create_variables_in_group(src_grp, dst_grp):
    """Create all variables in a group (without writing data yet)."""
    for name, var in src_grp.variables.items():
        if name not in dst_grp.variables:
            dst_var = dst_grp.createVariable(
                name, var.dtype, var.dimensions,
                zlib=True, complevel=6
            )
            # Preserve ALL attributes including coordinates
            dst_var.setncatts({a: var.getncattr(a) for a in var.ncattrs()})

def create_groups_recursive(src_grp, dst_grp):
    """Recursively create group structure and variables."""
    create_variables_in_group(src_grp, dst_grp)
    for grp_name, sub_grp in src_grp.groups.items():
        if grp_name not in dst_grp.groups:
            dst_sub = dst_grp.createGroup(grp_name)
        else:
            dst_sub = dst_grp.groups[grp_name]
        # Copy group attributes
        dst_sub.setncatts({a: sub_grp.getncattr(a) for a in sub_grp.ncattrs()})
        create_groups_recursive(sub_grp, dst_sub)

def append_group_data(src_grp, dst_grp, offset):
    """Recursively append data from src to dst along scan_lines, chunk by chunk."""
    for name, var in src_grp.variables.items():
        dst_var = dst_grp.variables[name]
        if CONCAT_DIM in var.dimensions:
            axis = var.dimensions.index(CONCAT_DIM)
            n = var.shape[axis]
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                # source slice
                src_idx = [slice(None)] * var.ndim
                src_idx[axis] = slice(start, end)
                # destination slice
                dst_idx = [slice(None)] * var.ndim
                dst_idx[axis] = slice(offset + start, offset + end)
                dst_var[tuple(dst_idx)] = var[tuple(src_idx)]
        else:
            # no scan_lines dim: write only once (from first file, offset==0)
            if offset == 0:
                dst_var[:] = var[:]

    for grp_name, sub_grp in src_grp.groups.items():
        append_group_data(sub_grp, dst_grp.groups[grp_name], offset)


# --- Main ---
with nc4.Dataset(outfile, 'w') as dst:
    # Copy global attributes from first file
    with nc4.Dataset(list_file[0], 'r') as src:
        dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
        create_dimensions(src, dst)
        create_groups_recursive(src, dst)

    # Append data file by file
    offset = 0
    for i, f in enumerate(list_file):
        print(f"Appending file {i+1}/{len(list_file)}: {f}  (offset={offset})")
        with nc4.Dataset(f, 'r') as src:
            n_lines = src.dimensions[CONCAT_DIM].size
            append_group_data(src, dst, offset)
            offset += n_lines
    # --------------------------------------------------
    # Correct global sensing times
    # --------------------------------------------------

    with nc4.Dataset(list_file[0], 'r') as src_first:
        dst.setncattr(
            'sensing_start_time',
            src_first.getncattr('sensing_start_time')
        )

    with nc4.Dataset(list_file[-1], 'r') as src_last:
        dst.setncattr(
            'sensing_end_time',
            src_last.getncattr('sensing_end_time')
        )
        
print(f"Done → {outfile}  (total scan_lines: {offset})")
