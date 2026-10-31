#!/usr/bin/env python
# coding: utf-8
"""
Local version of Convert_IASI_PCS_R2_to_nc_allPDU_final.py for one orbit:
combine PRP, PCH and IPO of each PDU into one netCDF file, then merge the PDU files into one orbit file.

usage: python Convert_IASI_hybrid_PCS_orbit_local.py <directory with the WO_* PDU directories> <output directory> [sat]
example: python Convert_IASI_hybrid_PCS_orbit_local.py /home/oboucher/IASI/iasi_pcs/hdf5 /tmp/out M03
"""
import os
import sys
import glob
import pathlib
import datetime as dt

import netCDF4 as nc4

import Routines_Convert_IASI_hybrid_PCS as iasi

dir_orbit = sys.argv[1]
dirout    = sys.argv[2]
sat       = sys.argv[3] if len(sys.argv) > 3 else 'M03'

pathlib.Path(dirout).mkdir(parents=True, exist_ok=True)

dir_pdu = sorted(glob.glob(dir_orbit + '/WO*'))
print('we have ', len(dir_pdu), 'pdus in ', dir_orbit)

fmt  = '%Y%m%d%H%M%SZ'
fmt1 = '%Y%m%d%H%M%S'

# ## create the netcdf for each 3 minute PDUs of one orbit
last_scanline = 0 # to compute which line we are in
list_file = []
for n, pdu in enumerate(dir_pdu):
    fprp = glob.glob(pdu + '/to_pgf/*PRP*Z')[0] # should be only one prp, pch and ipo file
    fpch = glob.glob(pdu + '/to_pgf/*PCH*Z')[0]
    fipo = glob.glob(pdu + '/to_pgf/*IPO*Z')[0]

    ### open the 3 files PRP, PCH and IPO
    nc    = nc4.Dataset(fprp, "r")
    ncpch = nc4.Dataset(fpch, "r")
    ncipo = nc4.Dataset(fipo, "r")

    # check that there are outliers in IPO
    IPO = 'yes' if ncipo.NbrOutliers > 0 else 'no'

    scan_lines = len(nc['L1C/LineNumber'][:])
    last_scanline = last_scanline + scan_lines

    # create a new filename
    time_now = dt.datetime.now()
    basename = os.path.basename(fprp)
    newname  = dirout + '/IASI_hybrid_PCS_' + basename[9:47] + '_R_O_' + dt.datetime.strftime(time_now, fmt) + '.nc'
    print(n, os.path.basename(pdu), 'scan_lines=', scan_lines, 'last_scanline=', last_scanline, 'IPO=', IPO)

    nfile = nc4.Dataset(newname, "w", format='NETCDF4')
    iasi.create_IASI_hybrid_PCS_netCDF(nfile, nc, ncpch, ncipo, scan_lines, last_scanline, sat, IPO)

    fnew   = os.path.basename(newname)
    parent = nc.groups["L1C"].ParentProduct
    orbit  = nc.groups["L1C"].orbit
    iasi.Write_Metadata_NC(nfile, fnew[23:37], fnew[39:53], sat, fnew, int(orbit), int(orbit) + 1,
                           dt.datetime.strftime(time_now, fmt1), parent)
    # close files
    nfile.close()
    nc.close()
    ncpch.close()
    ncipo.close()
    list_file.append(newname)

# ## merge the PDU files into one orbit file
outfile = dirout + '/merged_' + os.path.basename(os.path.dirname(os.path.abspath(dir_orbit))) + '.nc'
iasi.merge_pdu_files(list_file, outfile)
