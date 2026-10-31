#!/usr/bin/env python

import os
import sys
import argparse
import datetime as dt
import tempfile
import shutil

import numpy as np
import netCDF4 as nc4
import h5py

import matplotlib.dates as dates
import glob


# -------------------- DICTIONARIEs -----------------------
sat_dic = {
    'M01': 'METOPB',
    'M02': 'METOPA',
    'M03': 'METOPC'
}

#ev_dic = {
#    'M01_B1': 'IASI_EV1_xx_M01_20110125000000Z_xxxxxxxxxxxxxxZ_20110119000104Z_xxxx_xxxxxxxxxx',
#    'M01_B2': 'IASI_EV2_xx_M01_20110125000000Z_xxxxxxxxxxxxxxZ_20110119000104Z_xxxx_xxxxxxxxxx',
#    'M01_B3': 'IASI_EV3_xx_M01_20110125000000Z_xxxxxxxxxxxxxxZ_20110119000104Z_xxxx_xxxxxxxxxx',
#    'M02_B1': 'IASI_EV1_xx_M02_20110125000000Z_xxxxxxxxxxxxxxZ_20110119000104Z_xxxx_xxxxxxxxxx',
#    'M02_B2': 'IASI_EV2_xx_M02_20110125000000Z_xxxxxxxxxxxxxxZ_20110119000104Z_xxxx_xxxxxxxxxx',
#    'M02_B3': 'IASI_EV3_xx_M02_20110125000000Z_xxxxxxxxxxxxxxZ_20110119000104Z_xxxx_xxxxxxxxxx'
#}

ev_dic = { 'M02_B1': 'IASI_EV1_xx_Mxx_20200131000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx',
           'M02_B2': 'IASI_EV2_xx_Mxx_20200131000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx',
           'M02_B3': 'IASI_EV3_xx_Mxx_20200131000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx'
         }

def inspect_EV(evf,what ='ReconstructionOperator'):
    # look into Eigen vectors
    # this is a mini tool create to look into the eigen vector file
    #
    # there are different set for v2.1 and v1.4,
    # what was introduce to check the various dataset.
    # Note the'ReconstructionOperator' only exists for v2.1

    f = h5py.File(evf, 'r')

    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"DATASET: {name}  shape={obj.shape}  dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"GROUP:   {name}")
    print('Structure of ',evf)
    f.visititems(print_structure)

    dset = f[what]
    print()
    print('Shape dset')
    print(dset.shape)
    print(dset.dtype)

    print()
    print('metadata attribute if any:')
    for key, value in dset.attrs.items():
        print(key, value)

def create_arguments_parser():
    parser = argparse.ArgumentParser(description='Convert IASI PCS from processing output to CDR version for release',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('DateStart', help='Starting date in YYYYMMDD', type=str)
    parser.add_argument('DateEnd', help='Ending date in YYYYMMDD', type=str)
    parser.add_argument("Indir", help="Input directory of IASI PCS", type=str)
    # parser.add_argument("Workdir", help="Working base directory of IASI PCS", type=str)
    parser.add_argument("Outdir", help="Output directory of IASI PCS", type=str)

    return parser.parse_args()

def get_date_eps_file(eps_file):
    SensingStart = os.path.basename(eps_file)[16:30]
    # print(SensingStart)
    SensingStart = dt.datetime.strptime(SensingStart, "%Y%m%d%H%M%S")
    return SensingStart

def get_sat_eps_file(eps_file):
    sat_id = os.path.basename(eps_file)[12:15]

    return sat_id

def cleanup_tempdir(tempdir: str):
    if os.path.isdir(tempdir):
        shutil.rmtree(tempdir)

def copy_files(inputf,outputd,key):
    base = os.path.basename(inputf)
    #dirf = os.path.dirname(inputf)
    outdir = outputd+key+'/'
    os.makedirs(outdir, exist_ok=True)
    # outfile = os.path.join(outdir, base)
    shutil.copy2(inputf, outdir)    

def get_pc_scores(filename):
    """Return global PC scores with shape (scanline, field of view, 300)."""
    # input: one prp file
    # output pcs:  shape will be line, fov, PC - total 300 channels
    
    with nc4.Dataset(filename) as nc:
        pcs_group = nc.groups['L1C'].groups['PCscores']

        blocks = []
        for band in ['Band1', 'Band2', 'Band3']:
            for part in ['P1', 'P2', 'P3']:
                arr = pcs_group.groups[band][part][:]
                print(band, part, arr.shape)
                blocks.append(arr)

    # concatenate along PC dimension (last axis!)
    return np.concatenate(blocks, axis=2)  # shape will be line, fov, channel - total 8461 channels

def get_pc_scores_from_CDR(filename):
    """Return global PC scores with shape (scanline, field of view, 300)."""
    # input: one prp file
    # output pcs:  shape will be line, fov, PC - total 300 channels
    
    with nc4.Dataset(filename) as nc:
        pcs_group = nc.groups['PCscores']

        blocks = []
        for band in ['Band1', 'Band2', 'Band3']:
            for part in ['P1', 'P2', 'P3']:
                arr = pcs_group.groups[band][part][:]
                print(band, part, arr.shape)
                blocks.append(arr)

    # concatenate along PC dimension (last axis!)
    return np.concatenate(blocks, axis=2)  # shape will be line, fov, channel - total 8461 channels

def get_hpc_scores(filename):
    """Return hybrid PC scores with shape (scanline, pixel, 15)."""
    with nc4.Dataset(filename) as nc:
        blocks = [
            nc.groups['band_1']['p_local'][:],
            nc.groups['band_2']['p_local'][:],
            nc.groups['band_3']['p_local'][:],
        ]
    return np.concatenate(blocks, axis=2)

def get_hpc_scores_from_CDR(filename):
    """Return hybrid PC scores with shape (scanline, pixel, 15)."""
    with nc4.Dataset(filename) as nc:
        hpcs_groups = nc.groups['PClocal']

        block = []
        for band in ['Band1','Band2','Band3']:
            arr = hpcs_groups[band]['p_local'][:]
            print(band,arr.shape)
            block.append(arr)

    return np.concatenate(block,axis = 2)

def get_hpc_R_from_CDR(filename):
    # R_local is stored once per PDU (pdu, 5, nchan): expand to one per line (scan_lines, 5, nchan)
    with nc4.Dataset(filename) as nc:
        r1 = nc.groups['PClocal']['Band1']['R_local'][:]
        r2 = nc.groups['PClocal']['Band2']['R_local'][:]
        r3 = nc.groups['PClocal']['Band3']['R_local'][:]
        if 'pdu' in nc.groups['PClocal']['Band1']['R_local'].dimensions:
            pdu_index = nc['pdu_index'][:]
            r1, r2, r3 = r1[pdu_index], r2[pdu_index], r3[pdu_index]
    return r1, r2, r3

def get_hpc_R(filename):
    with nc4.Dataset(filename) as nc:
        r1 = nc.groups['band_1']['R_local'][:]
        r2 = nc.groups['band_2']['R_local'][:]
        r3 = nc.groups['band_3']['R_local'][:]
    return r1, r2, r3

def get_lonlat(filename):
    # get the lat and lon from the prp file
    
    with nc4.Dataset(filename) as nc:
        lon = nc.groups["L1C"]["Longitude"][:]
        lat = nc.groups["L1C"]["Latitude"][:]
    return lon, lat

def get_lonlat_from_CDR(filename):
    # get the lat and lon from the prp file
    
    with nc4.Dataset(filename) as nc:
        lon = nc.variables['longitude'][:]
        lat = nc.variables['latitude'][:]
    return lon, lat

def get_landfrac(filename):
    # get the lat and lon from the prp file
    
    with nc4.Dataset(filename) as nc:
        lf = nc.groups["L1C"]["LandFraction"][:]
    return lf

def get_landfrac_from_CDR(filename):
    # get the lat and lon from the prp file
    
    with nc4.Dataset(filename) as nc:
        lf = nc.variables['LandFraction'][:]
    return lf

def reconstruct(pcs, R1, R2, R3):
    # shape of R1, R2, and R3: (num_PC,num_channels) - (band1: 90, 1997)
    #                                                - (band2 : 120, 3119)
    #                                                - (band3 : 90, 3345)
    #
    # shape of pcs: [num_line,num_pixel,num_PC] - 23,120,300
    #
    # tensordot just does this for all pixels at once, very fast.
    #  spectrum = np.zeros(1997)
    #  for k in range(90):
    #     spectrum += pcs_pixel[k] * R1[k, :]
    
    n_pc = pcs.shape[2]
    n2, n3 = pcs.shape[0], pcs.shape[1]

    if n_pc == 300:
        x1 = np.tensordot(pcs[:,:,0:90], R1, axes=([2],[0]))
        x2 = np.tensordot(pcs[:,:,90:210], R2, axes=([2],[0]))
        x3 = np.tensordot(pcs[:,:,210:], R3, axes=([2],[0]))
        

    elif n_pc == 15:
        x1 = np.tensordot(pcs[:,:,0:5], R1, axes=([2],[0]))
        x2 = np.tensordot(pcs[:,:,5:10], R2, axes=([2],[0]))
        x3 = np.tensordot(pcs[:,:,10:], R3, axes=([2],[0]))
        
    else:
        raise ValueError("Unexpected number of PCs")
    #return x1,x2,x3
    #x = np.vstack([x1, x2, x3])
    return np.concatenate([x1, x2, x3], axis=2)


def load_reconstruction_operator(filename):
    with h5py.File(filename, "r") as f:
        return f['ReconstructionOperator'][:]

# ### Write_Metadata_NC
#---------------------
# add attributes to the NetCDF file
#---------------------
def Write_Metadata_NC(f,start_t,end_t, sat, f_name, orbit_start, orbit_end, processing_time, parent):
   
    
    fmt1 = '%Y%m%d%H%M%SZ'
    fmt  = '%Y-%m-%dT%H:%M:%SZ'

    time_now = dt.datetime.now()

    print('f:',f)
    print('fname in write:',f_name)
    print('time_now:',time_now)
    
    f.title                        = 'IASI Principle + local Component Score CDR'
    f.ParentProduct                = parent
    # f.creation_time                = dt.datetime.strftime(dt.datetime.strptime('20210510000000Z',fmt1),fmt)  # could be changed
    f.creation_time                = dt.datetime.strftime(dt.datetime.strptime(processing_time+'Z',fmt1),fmt)
    f.creator_email                = 'ops@eumetsat.int'
    f.creator_url                  = 'www.eumetsat.int' 
    f.id                           = 'DOI: 10.15770/EUM_SEC_CLM_00XX'
    f.institution                  = 'EUMETSAT'
    f.source                       = 'Operational satellite observation: IASI L1c R2 + NRT as input'
    f.history                      = dt.datetime.strftime(time_now, fmt) + ' netCDF generated from native file'
    f.references                   = 'DOI: 10.15770/EUM_SEC_CLM_00XX'
    f.summary                      = 'This is the first release of the global + hybrid IASI Principle Component Score (PCS) from METOP. This dataset was reprocessed using the EUMETSAT operational EUMETSAT algorithm v6.5.4 of the EUMETSAT operational IASI level 2 processing chain. Each file contains one full IASI orbit (Svalbard to Svalbard).'
    f.comment                      = 'For this release data from Metop-A and -B level 1c was used from the release 2 of reprocessed IASI level 1c data (10.15770/EUM_SEC_CLM_0064) and operational level1c data from Metop-C.'
    f.instrument                   = 'IASI'
    f.platform                     = sat_dic[sat]
    f.keywords                     = 'IASI, PCS, PRP, principle component'
    f.orbit_start                  = orbit_start
    f.orbit_end                    = orbit_end
    f.processing_level             = '1C'
    f.processing_algorithm_version = '6.5.4'
    f.product_version              = 'Release 1'
    f.product_name                 = f_name
    f.sensing_start_time           = dt.datetime.strftime(dt.datetime.strptime(start_t + 'Z',fmt1),fmt)
    f.sensing_end_time             = dt.datetime.strftime(dt.datetime.strptime(end_t + 'Z',fmt1),fmt)
    if sat =='M02':
        f.cdr_platform_start_time  = dt.datetime.strftime(dt.datetime.strptime('20070710072358Z',fmt1),fmt)
        f.cdr_platform_stop_time   = dt.datetime.strftime(dt.datetime.strptime('20210905222052Z',fmt1),fmt)
    if sat == 'M01':
        f.cdr_platform_start_time  = dt.datetime.strftime(dt.datetime.strptime('20130220035655Z',fmt1),fmt)
        f.cdr_platform_stop_time   = dt.datetime.strftime(dt.datetime.strptime('20250101011758Z',fmt1),fmt)
    if sat == 'M03':
        f.cdr_platform_start_time  = dt.datetime.strftime(dt.datetime.strptime('20181107hhmmssZ',fmt1),fmt)
        f.cdr_platform_stop_time   = dt.datetime.strftime(dt.datetime.strptime('20260630hhmmssZ',fmt1),fmt)
    f.processing_mode              = 'R'
    f.disposition_mode             = 'O' 
    f.Conventions                  = 'CF-1.6'
    f.metadata_conventions         = 'Unidata Dataset Discovery v1.0'
    f.product_level                = '1C'

# ### create_new_file_name

def create_new_file_name(out_dir, infile, sat):
    """
    Create a new file name based on input file name.
    Use 'r' (for reprocessing) and EUMP (for EPS)
    example output: W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,METOPA+PCS+IASI_C_EUMP_20100609013854Z_20100609032054Z_eps_r_l1_0100.nc
    :param out_dir:
    :param infile:
    :param sat:
    :return: new_file
    """

    new_file = 'W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,' + sat_dic[sat] + '+PCS+IASI_C_EUMP_'
    start = os.path.basename(infile)[16:30]
    stop = os.path.basename(infile)[32:46]
    processing_time = os.path.basename(infile)[52:66]

    new_file = new_file + start + 'Z_' + stop + 'Z_eps_r_l1_0100.nc'

    new_file = os.path.join(out_dir, new_file)

    return new_file, start, stop, processing_time


# ### create_IASI_PCS_netCDF

def create_IASI_PCS_netCDF(nf, of, scan_lines, sat):

    # create dimension
    nf.createDimension('pixels', 120)
    nf.createDimension('scan_lines', scan_lines)
    nf.createDimension('cluster', 3)
    nf.createDimension('AVH_CH', 6)
    nf.createDimension('B1P1', 1)
    nf.createDimension('B1P2', 41)
    nf.createDimension('B1P3', 48)
    nf.createDimension('B2P1', 2)
    nf.createDimension('B2P2', 61)
    nf.createDimension('B2P3', 57)
    nf.createDimension('B3P1', 1)
    nf.createDimension('B3P2', 44)
    nf.createDimension('B3P3', 45)

    # create groups
    #L1C = nf.createGroup('L1C')
    #Avhrr = L1C.createGroup('Avhrr')
    #PCscores = L1C.createGroup('PCscores')
    Avhrr = nf.createGroup('Avhrr')
    PCscores = nf.createGroup('PCscores')
    Band1 = PCscores.createGroup('Band1')
    Band2 = PCscores.createGroup('Band2')
    Band3 = PCscores.createGroup('Band3')

    # create and write variables
    latitude = nf.createVariable('latitude', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                  fill_value=9.969209968386869e+36)
    latitude.units = 'degrees_north'
    latitude.standard_name = 'latitude'
    latitude.long_name = 'latitude north positive'
    latitude.valid_min = -90.
    latitude.valid_max = 90.
    latitude.description = 'Latitude of the IASI pixel'
    latitude[:] = of['L1C/Latitude'][:]

    longitude = nf.createVariable('longitude', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                  fill_value=9.969209968386869e+36)
    longitude.units = 'degrees_east'
    longitude.standard_name = 'longitude'
    longitude.long_name = 'longitude west negative'
    longitude.valid_min = -180.
    longitude.valid_max = 180.
    longitude.description = 'Longitude of the IASI pixel'
    longitude[:] = of['L1C/Longitude'][:]

    CloudFraction = nf.createVariable('CloudFraction', 'u1', ['scan_lines', 'pixels'], zlib=True, complevel=6)
    CloudFraction.coordinates = 'latitude longitude'
    CloudFraction.units = 'percent'
    CloudFraction.standard_name = 'cloud_area_fraction'
    CloudFraction.long_name = 'Cloud fraction of IASI pixel'
    CloudFraction.valid_min = 0
    CloudFraction.valid_max = 100
    CloudFraction.description = 'Cloud fraction of the IASI pixel, based on AVHRR cloud flag'
    CloudFraction[:] = of['L1C/CloudFraction'][:]

    EUMQflag = nf.createVariable('EUMQflag', 'u1', ['scan_lines', 'pixels'], zlib=True, complevel=6)
    EUMQflag.coordinates = 'latitude longitude'
    EUMQflag.units = '1'
    EUMQflag.standard_name = 'quality_flag'
    EUMQflag.long_name = 'Eumetsat Quality flag'
    EUMQflag.valid_min = 0
    EUMQflag.valid_max = 255
    EUMQflag.description = 'Eumetsat Quality flag'
    EUMQflag[:] = of['L1C/EUMQflag'][:]

    EarthSatDistance = nf.createVariable('EarthSatDistance', 'u4', ['scan_lines'], zlib=True, complevel=6)
    EarthSatDistance.units = 'm'
    EarthSatDistance.standard_name = 'distance_from_geocenter'
    EarthSatDistance.long_name = 'Earth Satellite distance'
    EarthSatDistance.valid_min = 7000000
    EarthSatDistance.valid_max = 8000000
    EarthSatDistance.description = 'Earth Satellite distance, from earth centre'
    EarthSatDistance[:] = of['L1C/EarthSatDistance'][:]

    FLG_SATMAN = nf.createVariable('FLG_SATMAN', 'u1', ['scan_lines'], zlib=True, complevel=6)
    FLG_SATMAN.units = '1'
    FLG_SATMAN.standard_name = 'quality_flag'
    FLG_SATMAN.long_name = 'Satellite manoeuvre flag'
    FLG_SATMAN.valid_min = 0
    FLG_SATMAN.valid_max = 255
    FLG_SATMAN.description = 'Satellite manoeuvre flag (0=normal, 1=manoeuvre) '
    FLG_SATMAN[:] = of['L1C/FLG_SATMAN'][:]

    InstrumentMode = nf.createVariable('InstrumentMode', 'u1', ['scan_lines'], zlib=True, complevel=6)
    InstrumentMode.units = '1'
    InstrumentMode.standard_name = 'quality_flag'
    InstrumentMode.long_name = 'Instrument Mode'
    InstrumentMode.valid_min = 0
    InstrumentMode.valid_max = 255
    InstrumentMode.description = 'Instrument Mode, 161=normal'
    InstrumentMode[:] = of['L1C/InstrumentMode'][:]

    LandFraction = nf.createVariable('LandFraction', 'u1', ['scan_lines', 'pixels'], zlib=True, complevel=6)
    LandFraction.coordinates = 'latitude longitude'
    LandFraction.units = 'percent'
    LandFraction.standard_name = 'land_area_fraction'
    LandFraction.long_name = 'Land fraction in IASI pixel'
    LandFraction.valid_min = 0
    LandFraction.valid_max = 100
    LandFraction.description = 'Land fraction in IASI pixel from AVHRR 1B '
    LandFraction[:] = of['L1C/LandFraction'][:]

    LineNumber = nf.createVariable('LineNumber', 'u4', ['scan_lines'], zlib=True, complevel=6)
    # LineNumber.coordinates = 'latitude longitude'
    LineNumber.units = '1'
    # LineNumber.standard_name = 'land_area_fraction'
    LineNumber.long_name = 'Line number'
    LineNumber.valid_min = 0
    # LineNumber.valid_max = 100
    LineNumber.description = 'Line number in file'
    LineNumber[:] = of['L1C/LineNumber'][:]

    QFlag = nf.createVariable('QFlag', 'u1', ['scan_lines', 'pixels'], zlib=True, complevel=6)
    QFlag.coordinates = 'latitude longitude'
    QFlag.units = '1'
    QFlag.standard_name = 'quality_flag'
    QFlag.long_name = 'Quality flag of PCS'
    QFlag.valid_min = 0
    QFlag.valid_max = 100
    QFlag.description = 'Quality flag. Bit1: Band 1 L1C, Bit2: Band 2 L1C, Bit3: Band 3 L1C, Bit4: Band 1 PCC,' \
                        ' Bit5: Band 2 PCC, Bit6: Band 3 PCC'
    QFlag[:] = of['L1C/QFlag'][:]

    SatAzimuth = nf.createVariable('SatAzimuth', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                    fill_value=9.969209968386869e+36)
    SatAzimuth.coordinates = 'latitude longitude'
    SatAzimuth.units = '1'
    SatAzimuth.standard_name = 'platform_azimuth_angle'
    SatAzimuth.long_name = 'Platform/satellite azimuth angle'
    SatAzimuth.valid_min = 0.
    SatAzimuth.valid_max = 360.
    SatAzimuth.description = 'Platform/satellite azimuth angle'
    SatAzimuth[:] = of['L1C/SatAzimuth'][:]

    SatZenith = nf.createVariable('SatZenith', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                    fill_value=9.969209968386869e+36)
    SatZenith.coordinates = 'latitude longitude'
    SatZenith.units = '1'
    SatZenith.standard_name = 'platform_zenith_angle'
    SatZenith.long_name = 'Platform/satellite zenith angle'
    SatZenith.valid_min = 0.
    SatZenith.valid_max = 90.
    SatZenith.description = 'Platform/satellite zenith angle'
    SatZenith[:] = of['L1C/SatZenith'][:]

    SensingEndTime_day = nf.createVariable('SensingEndTime_day', 'i8', ['scan_lines'], zlib=True, complevel=6)
    SensingEndTime_day.units = 'day'
    SensingEndTime_day.standard_name = 'time'
    SensingEndTime_day.long_name = 'Sensing End Time in days since 1 January 2000'
    SensingEndTime_day.valid_min = 0
    SensingEndTime_day.valid_max = 32000
    SensingEndTime_day.description = 'Sensing End Time in days since 1 January 2000'
    SensingEndTime_day[:] = of['L1C/SensingEndTime_day'][:]

    SensingEndTime_msec = nf.createVariable('SensingEndTime_msec', 'u4', ['scan_lines'], zlib=True, complevel=6)
    SensingEndTime_msec.units = 'ms'
    SensingEndTime_msec.standard_name = 'time'
    SensingEndTime_msec.long_name = 'Sensing End Time in milliseconds of day'
    SensingEndTime_msec.valid_min = 0
    SensingEndTime_msec.valid_max = 86400000
    SensingEndTime_msec.description = 'Sensing End Time in milliseconds of day'
    SensingEndTime_msec[:] = of['L1C/SensingEndTime_msec'][:]

    SensingTime_day = nf.createVariable('SensingTime_day', 'i8', ['scan_lines'], zlib=True, complevel=6)
    SensingTime_day.units = 'ms'
    SensingTime_day.standard_name = 'time'
    SensingTime_day.long_name = 'Sensing Time in days since 1 January 2000'
    SensingTime_day.valid_min = 0
    SensingTime_day.valid_max = 32000
    SensingTime_day.description = 'Sensing Time in days since 1 January 2000'
    SensingTime_day[:] = of['L1C/SensingTime_day'][:]

    SensingTime_msec = nf.createVariable('SensingTime_msec', 'u4', ['scan_lines'], zlib=True, complevel=6)
    SensingTime_msec.units = 'ms'
    SensingTime_msec.standard_name = 'time'
    SensingTime_msec.long_name = 'Sensing Time in milliseconds of day'
    SensingTime_msec.valid_min = 0
    SensingTime_msec.valid_max = 86400000
    SensingTime_msec.description = 'Sensing Time in milliseconds of day'
    SensingTime_msec[:] = of['L1C/SensingTime_msec'][:]

    SunAzimuth = nf.createVariable('SunAzimuth', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                    fill_value=9.969209968386869e+36)
    SatZenith.coordinates = 'latitude longitude'
    SunAzimuth.units = 'degree'
    SunAzimuth.standard_name = 'solar_azimuth_angle'
    SunAzimuth.long_name = 'solar azimuth angle'
    SunAzimuth.valid_min = 0.
    SunAzimuth.valid_max = 360.
    SunAzimuth.description = 'solar azimuth angle'
    SunAzimuth[:] = of['L1C/SunAzimuth'][:]

    SunZenith = nf.createVariable('SunZenith', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                   fill_value=9.969209968386869e+36)
    SatZenith.coordinates = 'latitude longitude'
    SunZenith.units = 'degree'
    SunZenith.standard_name = 'solar_zenith_angle'
    SunZenith.long_name = 'solar zenith angle'
    SunZenith.valid_min = 0.
    SunZenith.valid_max = 90.
    SunZenith.description = 'solar zenith angle'
    SunZenith[:] = of['L1C/SunZenith'][:]

    # AVHRR group
    RadAnalMean = Avhrr.createVariable('RadAnalMean', 'f4', ['scan_lines', 'pixels', 'cluster', 'AVH_CH'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    RadAnalMean.coordinates = 'latitude longitude'
    RadAnalMean.units = 'W/(m^2*sr*m^-1)'
    RadAnalMean.standard_name = 'mean_toa_outgoing_radiance_per_unit_wavelength'
    RadAnalMean.long_name = 'Mean AVHRR radiance analysis'
    RadAnalMean.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    RadAnalMean.description = 'Mean AVHRR radiance within the three main radiance analysis clusters'
    RadAnalMean[:] = of['L1C/Avhrr/RadAnalMean'][:]

    RadAnalStd = Avhrr.createVariable('RadAnalStd', 'f4', ['scan_lines', 'pixels', 'cluster', 'AVH_CH'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    RadAnalStd.coordinates = 'latitude longitude'
    RadAnalStd.units = 'W/(m^2*sr*m^-1)'
    RadAnalStd.standard_name = 'std_toa_outgoing_radiance_per_unit_wavelength'
    RadAnalStd.long_name = 'Standard deviation AVHRR radiance analysis'
    RadAnalStd.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    RadAnalStd.description = 'Standard deviation of AVHRR radiance within the three main radiance analysis clusters'
    RadAnalStd[:] = of['L1C/Avhrr/RadAnalStd'][:]

    RadAnalWgt = Avhrr.createVariable('RadAnalWgt', 'f4', ['scan_lines', 'pixels', 'cluster'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    RadAnalWgt.coordinates = 'latitude longitude'
    RadAnalWgt.units = 'percent'
    RadAnalWgt.standard_name = 'area_fraction'
    RadAnalWgt.long_name = 'Fraction of IASI FOV covered by the three main AVHRR radiance analysis clusters'
    RadAnalWgt.valid_min = 0.
    RadAnalMean.valid_max = 100.
    RadAnalWgt.description = 'Fraction of IASI FOV covered by the three main AVHRR radiance analysis clusters'
    RadAnalWgt[:] = of['L1C/Avhrr/RadAnalWgt'][:]

    T4_mean = Avhrr.createVariable('T4_mean', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    T4_mean.coordinates = 'latitude longitude'
    T4_mean.units = 'W/(m^2*sr*m^-1)'
    T4_mean.standard_name = 'mean_toa_outgoing_radiance_per_unit_wavelength'
    T4_mean.long_name = 'Mean radiance of AVHRR channel 4'
    T4_mean.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    T4_mean.description = 'Mean radiance of AVHRR channel 4'
    T4_mean[:] = of['L1C/Avhrr/T4_mean'][:]

    T4_std = Avhrr.createVariable('T4_std', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    T4_std.coordinates = 'latitude longitude'
    T4_std.units = 'W/(m^2*sr*m^-1)'
    T4_std.standard_name = 'std_toa_outgoing_radiance_per_unit_wavelength'
    T4_std.long_name = 'Standard deviation of radiance of AVHRR channel 4'
    T4_std.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    T4_std.description = 'Standard deviation of radiance of AVHRR channel 4'
    T4_std[:] = of['L1C/Avhrr/T4_std'][:]

    T5_mean = Avhrr.createVariable('T5_mean', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    T5_mean.coordinates = 'latitude longitude'
    T5_mean.units = 'W/(m^2*sr*m^-1)'
    T5_mean.standard_name = 'mean_toa_outgoing_radiance_per_unit_wavelength'
    T5_mean.long_name = 'Mean radiance of AVHRR channel 5'
    T5_mean.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    T5_mean.description = 'Mean radiance of AVHRR channel 5'
    T5_mean[:] = of['L1C/Avhrr/T5_mean'][:]

    T5_std = Avhrr.createVariable('T5_std', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    T5_std.coordinates = 'latitude longitude'
    T5_std.units = 'W/(m^2*sr*m^-1)'
    T5_std.standard_name = 'std_toa_outgoing_radiance_per_unit_wavelength'
    T5_std.long_name = 'Standard deviation of radiance of AVHRR channel 5'
    T5_std.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    T5_std.description = 'Standard deviation of radiance of AVHRR channel 5'
    T5_std[:] = of['L1C/Avhrr/T5_std'][:]

    # PCscore group
    RadianceSum = PCscores.createVariable('RadianceSum', 'f4', ['scan_lines', 'pixels', 'cluster'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    RadianceSum.coordinates = 'latitude longitude'
    RadianceSum.units = 'W/(m^2*sr*m^-1)'
    RadianceSum.standard_name = 'sum_toa_outgoing_radiance_per_unit_wavelength'
    RadianceSum.long_name = 'Sum of reconstructed radiances in each band'
    RadianceSum.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    RadianceSum.description = 'Sum of reconstructed radiances in each band'
    RadianceSum[:] = of['L1C/PCscores/RadianceSum'][:]

    ResidualRms = PCscores.createVariable('ResidualRms', 'f4', ['scan_lines', 'pixels', 'cluster'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    ResidualRms.coordinates = 'latitude longitude'
    ResidualRms.units = 'N/A'
    ResidualRms.standard_name = 'N/A'
    ResidualRms.long_name = 'Noise normalized residual RMS in each band'
    ResidualRms.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    ResidualRms.description = 'Noise normalized residual RMS in each band'
    ResidualRms[:] = of['L1C/PCscores/ResidualRms'][:]

    # PCscore Band 1 group
    Band1.Eigenvectorfile = ev_dic[sat + '_B1']
    P1_band1 = Band1.createVariable('P1', 'i4', ['scan_lines', 'pixels', 'B1P1'], zlib=True,
                                       complevel=6)
    P1_band1.coordinates = 'latitude longitude'
    P1_band1.units = 'N/A'
    P1_band1.standard_name = 'N/A'
    P1_band1.long_name = 'Principle Component 1 for band 1'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    P1_band1.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P1_band1.description = 'Principle Component 1 for band 1'
    P1_band1[:] = of['L1C/PCscores/Band1/P1'][:] * 0.5

    P2_band1 = Band1.createVariable('P2', 'i4', ['scan_lines', 'pixels', 'B1P2'], zlib=True,
                                       complevel=6)
    P2_band1.coordinates = 'latitude longitude'
    P2_band1.units = 'N/A'
    P2_band1.standard_name = 'N/A'
    P2_band1.long_name = 'Principle Component 2 for band 1'
    # P2_band1.scale_factor = 0.5
    # P2_band1.add_offset = 0.0
    P2_band1.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P2_band1.description = 'Principle Component 2 for band 1'
    P2_band1[:] = of['L1C/PCscores/Band1/P2'][:] * 0.5

    P3_band1 = Band1.createVariable('P3', 'i4', ['scan_lines', 'pixels', 'B1P3'], zlib=True,
                                       complevel=6)
    P3_band1.coordinates = 'latitude longitude'
    P3_band1.units = 'N/A'
    P3_band1.standard_name = 'N/A'
    P3_band1.long_name = 'Principle Component 3 for band 1'
    # P3_band1.scale_factor = 0.5
    # P3_band1.add_offset = 0.0
    P3_band1.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P3_band1.description = 'Principle Component 3 for band 1'
    P3_band1[:] = of['L1C/PCscores/Band1/P3'][:] * 0.5

    # PCscore Band 2 group
    Band2.Eigenvectorfile = ev_dic[sat + '_B2']
    P1_band2 = Band2.createVariable('P1', 'i4', ['scan_lines', 'pixels', 'B2P1'], zlib=True,
                                       complevel=6)
    P1_band2.coordinates = 'latitude longitude'
    P1_band2.units = 'N/A'
    P1_band2.standard_name = 'N/A'
    P1_band2.long_name = 'Principle Component 1 for band 2'
    # P1_band2.scale_factor = 0.5
    # P1_band2.add_offset = 0.0
    P1_band2.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P1_band2.description = 'Principle Component 1 for band 2'
    P1_band2[:] = of['L1C/PCscores/Band2/P1'][:] * 0.5

    P2_band2 = Band2.createVariable('P2', 'i4', ['scan_lines', 'pixels', 'B2P2'], zlib=True,
                                       complevel=6)
    P2_band2.coordinates = 'latitude longitude'
    P2_band2.units = 'N/A'
    P2_band2.standard_name = 'N/A'
    P2_band2.long_name = 'Principle Component 2 for band 2'
    # P2_band2.scale_factor = 0.5
    # P2_band2.add_offset = 0.0
    P2_band2.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P2_band2.description = 'Principle Component 2 for band 2'
    P2_band2[:] = of['L1C/PCscores/Band2/P2'][:] * 0.5

    P3_band2 = Band2.createVariable('P3', 'i4', ['scan_lines', 'pixels', 'B2P3'], zlib=True,
                                       complevel=6)
    P3_band2.coordinates = 'latitude longitude'
    P3_band2.units = 'N/A'
    P3_band2.standard_name = 'N/A'
    P3_band2.long_name = 'Principle Component 3 for band 2'
    # P3_band2.scale_factor = 0.5
    # P3_band2.add_offset = 0.0
    P3_band2.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P3_band2.description = 'Principle Component 3 for band 2'
    P3_band2[:] = of['L1C/PCscores/Band2/P3'][:] * 0.5

    # PCscore Band 3 group
    Band3.Eigenvectorfile = ev_dic[sat + '_B3']
    P1_band3 = Band3.createVariable('P1', 'i4', ['scan_lines', 'pixels', 'B3P1'], zlib=True,
                                       complevel=6)
    P1_band3.coordinates = 'latitude longitude'
    P1_band3.units = 'N/A'
    P1_band3.standard_name = 'N/A'
    P1_band3.long_name = 'Principle Component 1 for band 3'
    # P1_band3.scale_factor = 0.5
    # P1_band3.add_offset = 0.0
    P1_band3.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P1_band3.description = 'Principle Component 1 for band 3'
    P1_band3[:] = of['L1C/PCscores/Band3/P1'][:] * 0.5

    P2_band3 = Band3.createVariable('P2', 'i4', ['scan_lines', 'pixels', 'B3P2'], zlib=True,
                                       complevel=6)
    P2_band3.coordinates = 'latitude longitude'
    P2_band3.units = 'N/A'
    P2_band3.standard_name = 'N/A'
    P2_band3.long_name = 'Principle Component 2 for band 3'
    # P2_band3.scale_factor = 0.5
    # P2_band3.add_offset = 0.0
    P2_band3.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P2_band3.description = 'Principle Component 2 for band 3'
    P2_band3[:] = of['L1C/PCscores/Band3/P2'][:] * 0.5

    P3_band3 = Band3.createVariable('P3', 'i4', ['scan_lines', 'pixels', 'B3P3'], zlib=True,
                                       complevel=6)
    P3_band3.coordinates = 'latitude longitude'
    P3_band3.units = 'N/A'
    P3_band3.standard_name = 'N/A'
    P3_band3.long_name = 'Principle Component 3 for band 3'
    # P3_band3.scale_factor = 0.5
    # P3_band3.add_offset = 0.0
    P3_band3.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P3_band3.description = 'Principle Component 3 for band 3'
    P3_band3[:] = of['L1C/PCscores/Band3/P3'][:] * 0.5

# ### create_IASI_hybrid_PCS_netCDF
def create_IASI_hybrid_PCS_netCDF(nf, of, ofpch,ofipo, scan_lines, last_scanline, sat, IPO):

    pixels    = 120
    num_chans = 8461
    cluster   = 3
    
    # create dimension
    nf.createDimension('pixels', 120)
    nf.createDimension('scan_lines', None) # scan_lines)
    
    nf.createDimension('chans_in_band1',1997)
    nf.createDimension('chans_in_band2',3119)
    nf.createDimension('chans_in_band3',3345)
    nf.createDimension('num_chans',8461)
    
    nf.createDimension('cluster', 3)
    nf.createDimension('AVH_CH', 6)
    nf.createDimension('B1P1', 1)
    nf.createDimension('B1P2', 41)
    nf.createDimension('B1P3', 48)
    nf.createDimension('B2P1', 2)
    nf.createDimension('B2P2', 61)
    nf.createDimension('B2P3', 57)
    nf.createDimension('B3P1', 1)
    nf.createDimension('B3P2', 44)
    nf.createDimension('B3P3', 45)
    
    nf.createDimension('B1PCH', 5)
    nf.createDimension('B2PCH', 5)
    nf.createDimension('B3PCH', 5)
    
    # create groups
    #L1C = nf.createGroup('L1C')
    #Avhrr = L1C.createGroup('Avhrr')
    #PCscores = L1C.createGroup('PCscores')
    Avhrr = nf.createGroup('Avhrr')
    
    PCscores = nf.createGroup('PCscores')
    Band1 = PCscores.createGroup('Band1')
    Band2 = PCscores.createGroup('Band2')
    Band3 = PCscores.createGroup('Band3')

    hybridPC =nf.createGroup('PClocal')
    hybrid_Band1 = hybridPC.createGroup('Band1')
    hybrid_Band2 = hybridPC.createGroup('Band2')
    hybrid_Band3 = hybridPC.createGroup('Band3')

    IPOgroup = nf.createGroup('IPO')
    
    # create and write variables
    latitude = nf.createVariable('latitude', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                  fill_value=9.969209968386869e+36)
    latitude.units = 'degrees_north'
    latitude.standard_name = 'latitude'
    latitude.long_name = 'latitude north positive'
    latitude.valid_min = -90.
    latitude.valid_max = 90.
    latitude.description = 'Latitude of the IASI pixel'
    latitude[:] = of['L1C/Latitude'][:]

    longitude = nf.createVariable('longitude', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                  fill_value=9.969209968386869e+36)
    longitude.units = 'degrees_east'
    longitude.standard_name = 'longitude'
    longitude.long_name = 'longitude west negative'
    longitude.valid_min = -180.
    longitude.valid_max = 180.
    longitude.description = 'Longitude of the IASI pixel'
    longitude[:] = of['L1C/Longitude'][:]

    CloudFraction = nf.createVariable('CloudFraction', 'u1', ['scan_lines', 'pixels'], zlib=True, complevel=6)
    CloudFraction.coordinates = 'latitude longitude'
    CloudFraction.units = 'percent'
    CloudFraction.standard_name = 'cloud_area_fraction'
    CloudFraction.long_name = 'Cloud fraction of IASI pixel'
    CloudFraction.valid_min = 0
    CloudFraction.valid_max = 100
    CloudFraction.description = 'Cloud fraction of the IASI pixel, based on AVHRR cloud flag'
    CloudFraction[:] = of['L1C/CloudFraction'][:]

    EUMQflag = nf.createVariable('EUMQflag', 'u1', ['scan_lines', 'pixels'], zlib=True, complevel=6)
    EUMQflag.coordinates = 'latitude longitude'
    EUMQflag.units = '1'
    EUMQflag.standard_name = 'quality_flag'
    EUMQflag.long_name = 'Eumetsat Quality flag'
    EUMQflag.valid_min = 0
    EUMQflag.valid_max = 255
    EUMQflag.description = 'Eumetsat Quality flag'
    EUMQflag[:] = of['L1C/EUMQflag'][:]

    EarthSatDistance = nf.createVariable('EarthSatDistance', 'u4', ['scan_lines'], zlib=True, complevel=6)
    EarthSatDistance.units = 'm'
    EarthSatDistance.standard_name = 'distance_from_geocenter'
    EarthSatDistance.long_name = 'Earth Satellite distance'
    EarthSatDistance.valid_min = 7000000
    EarthSatDistance.valid_max = 8000000
    EarthSatDistance.description = 'Earth Satellite distance, from earth centre'
    EarthSatDistance[:] = of['L1C/EarthSatDistance'][:]

    FLG_SATMAN = nf.createVariable('FLG_SATMAN', 'u1', ['scan_lines'], zlib=True, complevel=6)
    FLG_SATMAN.units = '1'
    FLG_SATMAN.standard_name = 'quality_flag'
    FLG_SATMAN.long_name = 'Satellite manoeuvre flag'
    FLG_SATMAN.valid_min = 0
    FLG_SATMAN.valid_max = 255
    FLG_SATMAN.description = 'Satellite manoeuvre flag (0=normal, 1=manoeuvre) '
    FLG_SATMAN[:] = of['L1C/FLG_SATMAN'][:]

    InstrumentMode = nf.createVariable('InstrumentMode', 'u1', ['scan_lines'], zlib=True, complevel=6)
    InstrumentMode.units = '1'
    InstrumentMode.standard_name = 'quality_flag'
    InstrumentMode.long_name = 'Instrument Mode'
    InstrumentMode.valid_min = 0
    InstrumentMode.valid_max = 255
    InstrumentMode.description = 'Instrument Mode, 161=normal'
    InstrumentMode[:] = of['L1C/InstrumentMode'][:]

    LandFraction = nf.createVariable('LandFraction', 'u1', ['scan_lines', 'pixels'], zlib=True, complevel=6)
    LandFraction.coordinates = 'latitude longitude'
    LandFraction.units = 'percent'
    LandFraction.standard_name = 'land_area_fraction'
    LandFraction.long_name = 'Land fraction in IASI pixel'
    LandFraction.valid_min = 0
    LandFraction.valid_max = 100
    LandFraction.description = 'Land fraction in IASI pixel from AVHRR 1B '
    LandFraction[:] = of['L1C/LandFraction'][:]

    LineNumber = nf.createVariable('LineNumber', 'u4', ['scan_lines'], zlib=True, complevel=6)
    # LineNumber.coordinates = 'latitude longitude'
    LineNumber.units = '1'
    # LineNumber.standard_name = 'land_area_fraction'
    LineNumber.long_name = 'Line number'
    LineNumber.valid_min = 0
    # LineNumber.valid_max = 100
    LineNumber.description = 'Line number in file'
    LineNumber[:] = of['L1C/LineNumber'][:]

    QFlag = nf.createVariable('QFlag', 'u1', ['scan_lines', 'pixels'], zlib=True, complevel=6)
    QFlag.coordinates = 'latitude longitude'
    QFlag.units = '1'
    QFlag.standard_name = 'quality_flag'
    QFlag.long_name = 'Quality flag of PCS'
    QFlag.valid_min = 0
    QFlag.valid_max = 100
    QFlag.description = 'Quality flag. Bit1: Band 1 L1C, Bit2: Band 2 L1C, Bit3: Band 3 L1C, Bit4: Band 1 PCC,' \
                        ' Bit5: Band 2 PCC, Bit6: Band 3 PCC'
    QFlag[:] = of['L1C/QFlag'][:]

    SatAzimuth = nf.createVariable('SatAzimuth', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                    fill_value=9.969209968386869e+36)
    SatAzimuth.coordinates = 'latitude longitude'
    SatAzimuth.units = '1'
    SatAzimuth.standard_name = 'platform_azimuth_angle'
    SatAzimuth.long_name = 'Platform/satellite azimuth angle'
    SatAzimuth.valid_min = 0.
    SatAzimuth.valid_max = 360.
    SatAzimuth.description = 'Platform/satellite azimuth angle'
    SatAzimuth[:] = of['L1C/SatAzimuth'][:]

    SatZenith = nf.createVariable('SatZenith', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                    fill_value=9.969209968386869e+36)
    SatZenith.coordinates = 'latitude longitude'
    SatZenith.units = '1'
    SatZenith.standard_name = 'platform_zenith_angle'
    SatZenith.long_name = 'Platform/satellite zenith angle'
    SatZenith.valid_min = 0.
    SatZenith.valid_max = 90.
    SatZenith.description = 'Platform/satellite zenith angle'
    SatZenith[:] = of['L1C/SatZenith'][:]

    SensingEndTime_day = nf.createVariable('SensingEndTime_day', 'i8', ['scan_lines'], zlib=True, complevel=6)
    SensingEndTime_day.units = 'day'
    SensingEndTime_day.standard_name = 'time'
    SensingEndTime_day.long_name = 'Sensing End Time in days since 1 January 2000'
    SensingEndTime_day.valid_min = 0
    SensingEndTime_day.valid_max = 32000
    SensingEndTime_day.description = 'Sensing End Time in days since 1 January 2000'
    SensingEndTime_day[:] = of['L1C/SensingEndTime_day'][:]

    SensingEndTime_msec = nf.createVariable('SensingEndTime_msec', 'u4', ['scan_lines'], zlib=True, complevel=6)
    SensingEndTime_msec.units = 'ms'
    SensingEndTime_msec.standard_name = 'time'
    SensingEndTime_msec.long_name = 'Sensing End Time in milliseconds of day'
    SensingEndTime_msec.valid_min = 0
    SensingEndTime_msec.valid_max = 86400000
    SensingEndTime_msec.description = 'Sensing End Time in milliseconds of day'
    SensingEndTime_msec[:] = of['L1C/SensingEndTime_msec'][:]

    SensingTime_day = nf.createVariable('SensingTime_day', 'i8', ['scan_lines'], zlib=True, complevel=6)
    SensingTime_day.units = 'ms'
    SensingTime_day.standard_name = 'time'
    SensingTime_day.long_name = 'Sensing Time in days since 1 January 2000'
    SensingTime_day.valid_min = 0
    SensingTime_day.valid_max = 32000
    SensingTime_day.description = 'Sensing Time in days since 1 January 2000'
    SensingTime_day[:] = of['L1C/SensingTime_day'][:]

    SensingTime_msec = nf.createVariable('SensingTime_msec', 'u4', ['scan_lines'], zlib=True, complevel=6)
    SensingTime_msec.units = 'ms'
    SensingTime_msec.standard_name = 'time'
    SensingTime_msec.long_name = 'Sensing Time in milliseconds of day'
    SensingTime_msec.valid_min = 0
    SensingTime_msec.valid_max = 86400000
    SensingTime_msec.description = 'Sensing Time in milliseconds of day'
    SensingTime_msec[:] = of['L1C/SensingTime_msec'][:]

    SunAzimuth = nf.createVariable('SunAzimuth', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                    fill_value=9.969209968386869e+36)
    SatZenith.coordinates = 'latitude longitude'
    SunAzimuth.units = 'degree'
    SunAzimuth.standard_name = 'solar_azimuth_angle'
    SunAzimuth.long_name = 'solar azimuth angle'
    SunAzimuth.valid_min = 0.
    SunAzimuth.valid_max = 360.
    SunAzimuth.description = 'solar azimuth angle'
    SunAzimuth[:] = of['L1C/SunAzimuth'][:]

    SunZenith = nf.createVariable('SunZenith', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                   fill_value=9.969209968386869e+36)
    SatZenith.coordinates = 'latitude longitude'
    SunZenith.units = 'degree'
    SunZenith.standard_name = 'solar_zenith_angle'
    SunZenith.long_name = 'solar zenith angle'
    SunZenith.valid_min = 0.
    SunZenith.valid_max = 90.
    SunZenith.description = 'solar zenith angle'
    SunZenith[:] = of['L1C/SunZenith'][:]

    # AVHRR group
    RadAnalMean = Avhrr.createVariable('RadAnalMean', 'f4', ['scan_lines', 'pixels', 'cluster', 'AVH_CH'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    RadAnalMean.coordinates = 'latitude longitude'
    RadAnalMean.units = 'W/(m^2*sr*m^-1)'
    RadAnalMean.standard_name = 'mean_toa_outgoing_radiance_per_unit_wavelength'
    RadAnalMean.long_name = 'Mean AVHRR radiance analysis'
    RadAnalMean.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    RadAnalMean.description = 'Mean AVHRR radiance within the three main radiance analysis clusters'
    RadAnalMean[:] = of['L1C/Avhrr/RadAnalMean'][:]

    RadAnalStd = Avhrr.createVariable('RadAnalStd', 'f4', ['scan_lines', 'pixels', 'cluster', 'AVH_CH'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    RadAnalStd.coordinates = 'latitude longitude'
    RadAnalStd.units = 'W/(m^2*sr*m^-1)'
    RadAnalStd.standard_name = 'std_toa_outgoing_radiance_per_unit_wavelength'
    RadAnalStd.long_name = 'Standard deviation AVHRR radiance analysis'
    RadAnalStd.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    RadAnalStd.description = 'Standard deviation of AVHRR radiance within the three main radiance analysis clusters'
    RadAnalStd[:] = of['L1C/Avhrr/RadAnalStd'][:]

    RadAnalWgt = Avhrr.createVariable('RadAnalWgt', 'f4', ['scan_lines', 'pixels', 'cluster'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    RadAnalWgt.coordinates = 'latitude longitude'
    RadAnalWgt.units = 'percent'
    RadAnalWgt.standard_name = 'area_fraction'
    RadAnalWgt.long_name = 'Fraction of IASI FOV covered by the three main AVHRR radiance analysis clusters'
    RadAnalWgt.valid_min = 0.
    RadAnalMean.valid_max = 100.
    RadAnalWgt.description = 'Fraction of IASI FOV covered by the three main AVHRR radiance analysis clusters'
    RadAnalWgt[:] = of['L1C/Avhrr/RadAnalWgt'][:]

    T4_mean = Avhrr.createVariable('T4_mean', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    T4_mean.coordinates = 'latitude longitude'
    T4_mean.units = 'W/(m^2*sr*m^-1)'
    T4_mean.standard_name = 'mean_toa_outgoing_radiance_per_unit_wavelength'
    T4_mean.long_name = 'Mean radiance of AVHRR channel 4'
    T4_mean.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    T4_mean.description = 'Mean radiance of AVHRR channel 4'
    T4_mean[:] = of['L1C/Avhrr/T4_mean'][:]

    T4_std = Avhrr.createVariable('T4_std', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    T4_std.coordinates = 'latitude longitude'
    T4_std.units = 'W/(m^2*sr*m^-1)'
    T4_std.standard_name = 'std_toa_outgoing_radiance_per_unit_wavelength'
    T4_std.long_name = 'Standard deviation of radiance of AVHRR channel 4'
    T4_std.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    T4_std.description = 'Standard deviation of radiance of AVHRR channel 4'
    T4_std[:] = of['L1C/Avhrr/T4_std'][:]

    T5_mean = Avhrr.createVariable('T5_mean', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    T5_mean.coordinates = 'latitude longitude'
    T5_mean.units = 'W/(m^2*sr*m^-1)'
    T5_mean.standard_name = 'mean_toa_outgoing_radiance_per_unit_wavelength'
    T5_mean.long_name = 'Mean radiance of AVHRR channel 5'
    T5_mean.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    T5_mean.description = 'Mean radiance of AVHRR channel 5'
    T5_mean[:] = of['L1C/Avhrr/T5_mean'][:]

    T5_std = Avhrr.createVariable('T5_std', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    T5_std.coordinates = 'latitude longitude'
    T5_std.units = 'W/(m^2*sr*m^-1)'
    T5_std.standard_name = 'std_toa_outgoing_radiance_per_unit_wavelength'
    T5_std.long_name = 'Standard deviation of radiance of AVHRR channel 5'
    T5_std.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    T5_std.description = 'Standard deviation of radiance of AVHRR channel 5'
    T5_std[:] = of['L1C/Avhrr/T5_std'][:]

    # PCscore group
    RadianceSum = PCscores.createVariable('RadianceSum', 'f4', ['scan_lines', 'pixels', 'cluster'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    RadianceSum.coordinates = 'latitude longitude'
    RadianceSum.units = 'W/(m^2*sr*m^-1)'
    RadianceSum.standard_name = 'sum_toa_outgoing_radiance_per_unit_wavelength'
    RadianceSum.long_name = 'Sum of reconstructed radiances in each band'
    RadianceSum.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    RadianceSum.description = 'Sum of reconstructed radiances in each band'
    RadianceSum[:] = of['L1C/PCscores/RadianceSum'][:]

    ResidualRms = PCscores.createVariable('ResidualRms', 'f4', ['scan_lines', 'pixels', 'cluster'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    ResidualRms.coordinates = 'latitude longitude'
    ResidualRms.units = 'N/A'
    ResidualRms.standard_name = 'N/A'
    ResidualRms.long_name = 'Noise normalized residual RMS in each band'
    ResidualRms.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    ResidualRms.description = 'Noise normalized residual RMS in each band'
    ResidualRms[:] = of['L1C/PCscores/ResidualRms'][:]

    # PCscore Band 1 group
    Band1.Eigenvectorfile = ev_dic[sat + '_B1']
    P1_band1 = Band1.createVariable('P1', 'i4', ['scan_lines', 'pixels', 'B1P1'], zlib=True,
                                       complevel=6)
    P1_band1.coordinates = 'latitude longitude'
    P1_band1.units = 'N/A'
    P1_band1.standard_name = 'N/A'
    P1_band1.long_name = 'Principle Component 1 for band 1'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    P1_band1.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P1_band1.description = 'Principle Component 1 for band 1'
    P1_band1[:] = of['L1C/PCscores/Band1/P1'][:] * 0.5

    P2_band1 = Band1.createVariable('P2', 'i4', ['scan_lines', 'pixels', 'B1P2'], zlib=True,
                                       complevel=6)
    P2_band1.coordinates = 'latitude longitude'
    P2_band1.units = 'N/A'
    P2_band1.standard_name = 'N/A'
    P2_band1.long_name = 'Principle Component 2 for band 1'
    # P2_band1.scale_factor = 0.5
    # P2_band1.add_offset = 0.0
    P2_band1.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P2_band1.description = 'Principle Component 2 for band 1'
    P2_band1[:] = of['L1C/PCscores/Band1/P2'][:] * 0.5

    P3_band1 = Band1.createVariable('P3', 'i4', ['scan_lines', 'pixels', 'B1P3'], zlib=True,
                                       complevel=6)
    P3_band1.coordinates = 'latitude longitude'
    P3_band1.units = 'N/A'
    P3_band1.standard_name = 'N/A'
    P3_band1.long_name = 'Principle Component 3 for band 1'
    # P3_band1.scale_factor = 0.5
    # P3_band1.add_offset = 0.0
    P3_band1.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P3_band1.description = 'Principle Component 3 for band 1'
    P3_band1[:] = of['L1C/PCscores/Band1/P3'][:] * 0.5

    # PCscore Band 2 group
    Band2.Eigenvectorfile = ev_dic[sat + '_B2']
    P1_band2 = Band2.createVariable('P1', 'i4', ['scan_lines', 'pixels', 'B2P1'], zlib=True,
                                       complevel=6)
    P1_band2.coordinates = 'latitude longitude'
    P1_band2.units = 'N/A'
    P1_band2.standard_name = 'N/A'
    P1_band2.long_name = 'Principle Component 1 for band 2'
    # P1_band2.scale_factor = 0.5
    # P1_band2.add_offset = 0.0
    P1_band2.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P1_band2.description = 'Principle Component 1 for band 2'
    P1_band2[:] = of['L1C/PCscores/Band2/P1'][:] * 0.5

    P2_band2 = Band2.createVariable('P2', 'i4', ['scan_lines', 'pixels', 'B2P2'], zlib=True,
                                       complevel=6)
    P2_band2.coordinates = 'latitude longitude'
    P2_band2.units = 'N/A'
    P2_band2.standard_name = 'N/A'
    P2_band2.long_name = 'Principle Component 2 for band 2'
    # P2_band2.scale_factor = 0.5
    # P2_band2.add_offset = 0.0
    P2_band2.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P2_band2.description = 'Principle Component 2 for band 2'
    P2_band2[:] = of['L1C/PCscores/Band2/P2'][:] * 0.5

    P3_band2 = Band2.createVariable('P3', 'i4', ['scan_lines', 'pixels', 'B2P3'], zlib=True,
                                       complevel=6)
    P3_band2.coordinates = 'latitude longitude'
    P3_band2.units = 'N/A'
    P3_band2.standard_name = 'N/A'
    P3_band2.long_name = 'Principle Component 3 for band 2'
    # P3_band2.scale_factor = 0.5
    # P3_band2.add_offset = 0.0
    P3_band2.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P3_band2.description = 'Principle Component 3 for band 2'
    P3_band2[:] = of['L1C/PCscores/Band2/P3'][:] * 0.5

    # PCscore Band 3 group
    Band3.Eigenvectorfile = ev_dic[sat + '_B3']
    P1_band3 = Band3.createVariable('P1', 'i4', ['scan_lines', 'pixels', 'B3P1'], zlib=True,
                                       complevel=6)
    P1_band3.coordinates = 'latitude longitude'
    P1_band3.units = 'N/A'
    P1_band3.standard_name = 'N/A'
    P1_band3.long_name = 'Principle Component 1 for band 3'
    # P1_band3.scale_factor = 0.5
    # P1_band3.add_offset = 0.0
    P1_band3.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P1_band3.description = 'Principle Component 1 for band 3'
    P1_band3[:] = of['L1C/PCscores/Band3/P1'][:] * 0.5

    P2_band3 = Band3.createVariable('P2', 'i4', ['scan_lines', 'pixels', 'B3P2'], zlib=True,
                                       complevel=6)
    P2_band3.coordinates = 'latitude longitude'
    P2_band3.units = 'N/A'
    P2_band3.standard_name = 'N/A'
    P2_band3.long_name = 'Principle Component 2 for band 3'
    # P2_band3.scale_factor = 0.5
    # P2_band3.add_offset = 0.0
    P2_band3.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P2_band3.description = 'Principle Component 2 for band 3'
    P2_band3[:] = of['L1C/PCscores/Band3/P2'][:] * 0.5

    P3_band3 = Band3.createVariable('P3', 'i4', ['scan_lines', 'pixels', 'B3P3'], zlib=True,
                                       complevel=6)
    P3_band3.coordinates = 'latitude longitude'
    P3_band3.units = 'N/A'
    P3_band3.standard_name = 'N/A'
    P3_band3.long_name = 'Principle Component 3 for band 3'
    # P3_band3.scale_factor = 0.5
    # P3_band3.add_offset = 0.0
    P3_band3.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    P3_band3.description = 'Principle Component 3 for band 3'
    P3_band3[:] = of['L1C/PCscores/Band3/P3'][:] * 0.5

    #---------------------------------------
    # add hybrid PCS
    #
    # hybrid PCscore Band 1 group
    #---------------------------------------
    #Band1.Eigenvectorfile = ev_dic[sat + '_B1']
    Elocal_band1_hyb = hybrid_Band1.createVariable('E_local', 'f4', ['scan_lines','pixels','B1PCH', 'chans_in_band1'], zlib=True,
                                       complevel=6)
    #P1_band1_hyb.coordinates = 'latitude longitude'
    Elocal_band1_hyb.units = 'N/A'
    Elocal_band1_hyb.standard_name = 'N/A'
    Elocal_band1_hyb.long_name = 'E_local for band 1'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    # P1_band1_hyb.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    Elocal_band1_hyb.description = 'E_local for band 1'
    Elocal1 = ofpch['band_1/E_local'][:]    # (5, 1997)
    #print(type(Elocal))
    #print(Elocal.shape)
    #print(type(Elocal_band1_hyb))
    #print(Elocal_band1_hyb.shape)
    Elocal_band1_hyb[:,:,:,:] = Elocal1[np.newaxis, np.newaxis,:, :]
    
    plocal_band1_hyb = hybrid_Band1.createVariable('p_local', 'i4', ['scan_lines', 'pixels', 'B1PCH'], zlib=True,
                                       complevel=6)
    plocal_band1_hyb.coordinates = 'latitude longitude'
    plocal_band1_hyb.units = 'N/A'
    plocal_band1_hyb.standard_name = 'N/A'
    plocal_band1_hyb.long_name = 'p_local for band 1'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    # P1_band1_hyb.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    plocal_band1_hyb.description = 'p_local for band 1'
    plocal_band1_hyb[:] = ofpch['band_1/p_local'][:] 

    Rlocal_band1_hyb = hybrid_Band1.createVariable('R_local', 'f4', ['scan_lines','pixels','B1PCH', 'chans_in_band1'], zlib=True,
                                       complevel=6)
    #P1_band1_hyb.coordinates = 'latitude longitude'
    Rlocal_band1_hyb.units = 'N/A'
    Rlocal_band1_hyb.standard_name = 'N/A'
    Rlocal_band1_hyb.long_name = 'R_local for band 1'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    # P1_band1_hyb.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    Rlocal_band1_hyb.description = 'R_local for band 1'
    Rlocal1 = ofpch['band_1/R_local'][:]    # (5, 1997)
    Rlocal_band1_hyb[:,:,:,:] = Rlocal1[np.newaxis, np.newaxis,:, :]
    
    rmslocal_band1_hyb = hybrid_Band1.createVariable('rms_local', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6)
    rmslocal_band1_hyb.coordinates = 'latitude longitude'
    rmslocal_band1_hyb.units = 'N/A'
    rmslocal_band1_hyb.standard_name = 'N/A'
    rmslocal_band1_hyb.long_name = 'rms_local for band 1'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    # P1_band1_hyb.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    rmslocal_band1_hyb.description = 'rms_local for band 1'
    rmslocal_band1_hyb[:] = ofpch['band_1/rms_local'][:]

    Vlocal_band1_hyb = hybrid_Band1.createVariable('V_local', 'f4', ['scan_lines','pixels','B1PCH'], zlib=True,
                                       complevel=6)
    #rmslocal_band1_hyb.coordinates = 'latitude longitude'
    Vlocal_band1_hyb.units = 'N/A'
    Vlocal_band1_hyb.standard_name = 'N/A'
    Vlocal_band1_hyb.long_name = 'rms_local for band 1'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    # P1_band1_hyb.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    Vlocal_band1_hyb.description = 'V_local for band 1'
    Vlocal1 = ofpch['band_1/V_local'][:]    # (5, 1997)
    Vlocal_band1_hyb[:,:,:] = Vlocal1[np.newaxis, np.newaxis,:]

    #
    # hybrid PCscore Band 2 group
    #---------------------------------------
    #Band1.Eigenvectorfile = ev_dic[sat + '_B1']
    Elocal_band2_hyb = hybrid_Band2.createVariable('E_local', 'f4', ['scan_lines','pixels','B2PCH', 'chans_in_band2'], zlib=True,
                                       complevel=6)
    #P1_band1_hyb.coordinates = 'latitude longitude'
    Elocal_band2_hyb.units = 'N/A'
    Elocal_band2_hyb.standard_name = 'N/A'
    Elocal_band2_hyb.long_name = 'E_local for band 2'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    # P1_band1_hyb.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    Elocal_band2_hyb.description = 'E_local for band 2'
    Elocal2 = ofpch['band_2/E_local'][:]    # (5, 3119)
    Elocal_band2_hyb[:,:,:,:] = Elocal2[np.newaxis, np.newaxis,:, :] 

    plocal_band2_hyb = hybrid_Band2.createVariable('p_local', 'i4', ['scan_lines', 'pixels', 'B2PCH'], zlib=True,
                                       complevel=6)
    plocal_band2_hyb.coordinates = 'latitude longitude'
    plocal_band2_hyb.units = 'N/A'
    plocal_band2_hyb.standard_name = 'N/A'
    plocal_band2_hyb.long_name = 'p_local for band 2'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    # P1_band1_hyb.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    plocal_band2_hyb.description = 'p_local for band 2'
    plocal_band2_hyb[:] = ofpch['band_2/p_local'][:] 

    Rlocal_band2_hyb = hybrid_Band2.createVariable('R_local', 'f4', ['scan_lines','pixels','B1PCH', 'chans_in_band2'], zlib=True,
                                       complevel=6)
    #P1_band1_hyb.coordinates = 'latitude longitude'
    Rlocal_band2_hyb.units = 'N/A'
    Rlocal_band2_hyb.standard_name = 'N/A'
    Rlocal_band2_hyb.long_name = 'R_local for band 2'
    Rlocal2 = ofpch['band_2/R_local'][:]    # (5, 3119 )
    Rlocal_band2_hyb[:,:,:,:] = Rlocal2[np.newaxis, np.newaxis,:, :]

    rmslocal_band2_hyb = hybrid_Band2.createVariable('rms_local', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6)
    rmslocal_band2_hyb.coordinates = 'latitude longitude'
    rmslocal_band2_hyb.units = 'N/A'
    rmslocal_band2_hyb.standard_name = 'N/A'
    rmslocal_band2_hyb.long_name = 'rms_local for band 2'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    # P1_band1_hyb.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    rmslocal_band2_hyb.description = 'rms_local for band 2'
    rmslocal_band2_hyb[:] = ofpch['band_2/rms_local'][:]

    Vlocal_band2_hyb = hybrid_Band2.createVariable('V_local', 'f4', ['scan_lines','pixels','B2PCH'], zlib=True,
                                       complevel=6)
    #rmslocal_band1_hyb.coordinates = 'latitude longitude'
    Vlocal_band2_hyb.units = 'N/A'
    Vlocal_band2_hyb.standard_name = 'N/A'
    Vlocal_band2_hyb.long_name = 'rms_local for band 2'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    # P1_band1_hyb.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    Vlocal_band2_hyb.description = 'V_local for band 2'  
    Vlocal2= ofpch['band_2/V_local'][:]    # (5, 3119)
    Vlocal_band2_hyb[:,:,:] = Vlocal2[np.newaxis, np.newaxis,:]

    #
    # hybrid PCscore Band 3 group
    #---------------------------------------
    #Band1.Eigenvectorfile = ev_dic[sat + '_B1']
    Elocal_band3_hyb = hybrid_Band3.createVariable('E_local', 'f4', ['scan_lines','pixels','B3PCH', 'chans_in_band3'], zlib=True,
                                       complevel=6)
    #P1_band1_hyb.coordinates = 'latitude longitude'
    Elocal_band3_hyb.units = 'N/A'
    Elocal_band3_hyb.standard_name = 'N/A'
    Elocal_band3_hyb.long_name = 'E_local for band 3'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    # P1_band1_hyb.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    Elocal_band3_hyb.description = 'E_local for band 3'
    Elocal3 = ofpch['band_3/E_local'][:]    # (5, 3345)
    Elocal_band3_hyb[:,:,:,:] = Elocal3[np.newaxis, np.newaxis,:, :] 
 
    plocal_band3_hyb = hybrid_Band3.createVariable('p_local', 'i4', ['scan_lines', 'pixels', 'B3PCH'], zlib=True,
                                       complevel=6)
    plocal_band3_hyb.coordinates = 'latitude longitude'
    plocal_band3_hyb.units = 'N/A'
    plocal_band3_hyb.standard_name = 'N/A'
    plocal_band3_hyb.long_name = 'p_local for band 3'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    # P1_band1_hyb.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    plocal_band3_hyb.description = 'p_local for band 3'
    plocal_band3_hyb[:] = ofpch['band_3/p_local'][:] 

    Rlocal_band3_hyb = hybrid_Band3.createVariable('R_local', 'f4', ['scan_lines','pixels','B1PCH', 'chans_in_band3'], zlib=True,
                                       complevel=6)
    #P1_band1_hyb.coordinates = 'latitude longitude'
    Rlocal_band3_hyb.units = 'N/A'
    Rlocal_band3_hyb.standard_name = 'N/A'
    Rlocal_band3_hyb.long_name = 'R_local for band 3'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    # P1_band1_hyb.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    Rlocal_band3_hyb.description = 'R_local for band 3'
    Rlocal3 = ofpch['band_3/R_local'][:]    # (5, 3345 )
    Rlocal_band3_hyb[:,:,:,:] = Rlocal3[np.newaxis, np.newaxis,:, :]

    rmslocal_band3_hyb = hybrid_Band3.createVariable('rms_local', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6)
    rmslocal_band3_hyb.coordinates = 'latitude longitude'
    rmslocal_band3_hyb.units = 'N/A'
    rmslocal_band3_hyb.standard_name = 'N/A'
    rmslocal_band3_hyb.long_name = 'rms_local for band 3'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    # P1_band1_hyb.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    rmslocal_band3_hyb.description = 'rms_local for band 3'
    rmslocal_band3_hyb[:] = ofpch['band_3/rms_local'][:]

    Vlocal_band3_hyb = hybrid_Band3.createVariable('V_local', 'f4', ['scan_lines','pixels','B3PCH'], zlib=True,
                                       complevel=6)
    #rmslocal_band1_hyb.coordinates = 'latitude longitude'
    Vlocal_band3_hyb.units = 'N/A'
    Vlocal_band3_hyb.standard_name = 'N/A'
    Vlocal_band3_hyb.long_name = 'rms_local for band 3'
    # P1_band1.scale_factor = 0.5
    # P1_band1.add_offset = 0.0
    # P1_band1_hyb.valid_min = 0.
    # RadAnalMean.valid_max = 99.
    Vlocal_band3_hyb.description = 'V_local for band 3'
    Vlocal3= ofpch['band_3/V_local'][:]    # (5, 3345)
    Vlocal_band3_hyb[:,:,:] = Vlocal3[np.newaxis, np.newaxis,:]

    # add statistics
    #---------------------
    stat_res_mean = IPOgroup.createVariable('ResidualMean', 'f4', ['scan_lines','pixels','num_chans'], zlib=True, complevel=6)
    #stat_res_mean.coordinates = 'latitude'
    stat_res_mean.units = '1'
    stat_res_mean.standard_name = 'IPOresmean'
    stat_res_mean.long_name = 'IPO Statistics ResidualMean'
    stat_res_mean.valid_min = -1000.
    stat_res_mean.valid_max = 1000.
    stat_res_mean.description = 'IPO Statistics Residual Mean'

    res_mean = ofipo['Statistics']['ResidualMean'][:]    # (8461)
    stat_res_mean[:,:,:] = res_mean[np.newaxis, np.newaxis,:]

    # need to recopy res_mean to each line of the pdu: not big in the file because of zlub=True
    #res_mean = ofipo['Statistics']['ResidualMean'][:]
    #for i in range(stat_res_mean.shape[0]):  
    #    stat_res_mean[i, :] = res_mean

    #---------------------
    stat_res_stdv = IPOgroup.createVariable('ResidualStdv', 'f4', ['scan_lines', 'pixels', 'num_chans'], zlib=True, complevel=6)
    #stat_res_mean.coordinates = 'latitude'
    stat_res_stdv.units = '1'
    stat_res_stdv.standard_name = 'IPOresstdv'
    stat_res_stdv.long_name = 'IPO Statistics ResidualStdv'
    stat_res_stdv.valid_min = -1000.
    stat_res_stdv.valid_max = 1000.
    stat_res_stdv.description = 'IPO Statistics Residual Standard deviation'

    res_stdv = ofipo['Statistics']['ResidualStdv'][:]    # (8461)
    stat_res_stdv[:,:,:] = res_stdv[np.newaxis, np.newaxis,:]
    
    # need to recopy res_mean to each line of the pdu
    #res_stdv = ofipo['Statistics']['ResidualStdv'][:]
    #for i in range(stat_res_stdv.shape[0]):  
    #    stat_res_stdv[i, :] = res_stdv
    
    L1cRadiances = IPOgroup.createVariable('IPO_L1cRadiances', 'f4', ['scan_lines','pixels','num_chans'], zlib=True, 
                                     complevel=6, fill_value=np.float32(-9999.0),shuffle=True, chunksizes=(1, 120, 1024))
    L1cRadiances.coordinates = 'latitude longitude'
    L1cRadiances.units = '1'
    L1cRadiances.standard_name = 'IPO_L1cRadiances'
    L1cRadiances.long_name = 'IPO L1cRadiances'
    #L1cRadiances.valid_min = -1000.
    #L1cRadiances.valid_max = 1000.
    L1cRadiances.description = 'IPO L1cRadiances'
    # Initialize the NetCDF variable
    L1cRadiances[:] = -9999.0

    #if 'L1cRadiance' in ncipo.variables:
    if IPO == 'yes':
        pix    = ofipo['PixelNumber'][:]
        lin    = ofipo['LineNumber'][:]
        L1cRad = ofipo['L1cRadiance'][:]

        L1cRadiances[lin, pix, :] = L1cRad

    PccResidual = IPOgroup.createVariable('IPO_PccResidual', 'f4', ['scan_lines','pixels','num_chans'], zlib=True, 
                                     complevel=6, fill_value=np.float32(-9999.0),shuffle=True, chunksizes=(1, 120, 1024))
    PccResidual.coordinates = 'latitude longitude'
    PccResidual.units = '1'
    PccResidual.standard_name = 'IPO_PccResidual'
    PccResidual.long_name = 'IPO PccResidual'
    #L1cRadiances.valid_min = -1000.
    #L1cRadiances.valid_max = 1000.
    PccResidual.description = 'IPO PccResidual'
    # Initialize the NetCDF variable
    PccResidual[:] = -9999.0

    #if 'PccResidual' in ncipo.variables:
    if IPO == 'yes':
        pix    = ofipo['PixelNumber'][:]
        lin    = ofipo['LineNumber'][:]
        PccRes = ofipo['PccResidual'][:]

        PccResidual[lin, pix, :] = PccRes
        
    ResidualRMS = IPOgroup.createVariable('IPO_ResidualRMS', 'f4', ['scan_lines','pixels','cluster'], zlib=True, 
                                     complevel=6, fill_value=np.float32(-9999.0),shuffle=True, chunksizes=(1, 120, 3))
    ResidualRMS.coordinates = 'latitude longitude'
    ResidualRMS.units = '1'
    ResidualRMS.standard_name = 'IPO_ResidualRMS'
    ResidualRMS.long_name = 'IPO ResidualRMS'
    #L1cRadiances.valid_min = -1000.
    #L1cRadiances.valid_max = 1000.
    ResidualRMS.description = 'IPO ResidualRMS'
    # Initialize the NetCDF variable
    ResidualRMS[:] = -9999.0

    if IPO == 'yes':
    #if 'ResidualRMS' in ncipo.variables:
        pix    = ofipo['PixelNumber'][:]
        lin    = ofipo['LineNumber'][:]
        ResRMS = ofipo['ResidualRMS'][:]

        ResidualRMS[lin, pix, :] = ResRMS

def plot_one_spect_all_pixels(wn,spec,ifov,mini,maxi,savefi='no',unit='rad',add_vert='no'):
# plot one sprectrum for all 4 pixels
# plot_one_spect_all_pixels(wn,spec*1000.*100.,ifov,0,120,savefi='no',unit='rad')
# for the radiance *1000*100 to get the correct unit mw/m2/sr/cm-1

    wl2freq=lambda t: 1.E4/t
    newlabel = [15,12,10,8,6,4]
    newpos   = [wl2freq(t) for t in newlabel]

    fig, ax1 = plt.subplots(figsize=(14,8))

    ax1.tick_params(axis="x", labelsize=14,rotation=0)
    ax1.tick_params(axis="y", labelsize=14,rotation=0)

    ax1.set_ylim(mini,maxi)
    ax1.set_xlim(wn[0],wn[len(wn)-1])

    ax1.plot(wn,spec[:,ifov,0],label='Pixel 1',color='blue')
    ax1.plot(wn,spec[:,ifov,1],label='Pixel 2',color='yellow')
    ax1.plot(wn,spec[:,ifov,2],label='Pixel 3',color='red')
    ax1.plot(wn,spec[:,ifov,3],label='Pixel 4',color='green')

    if unit=='rad':
        ax1.set_ylabel('Radiance (mW/m$^2$/sr/cm$^{-1}$)',fontsize=18)
    if unit=='BT':
        ax1.set_ylabel('Brigthness Temperature (K)',fontsize=18)
    ax1.set_xlabel('Wavenumber (cm$^{-1}$)',fontsize=18,color='black')
    ax1.legend(loc='best',fontsize=18)

    # add a top axes showing the wavelength
    ax2 = ax1.twiny()
    ax2.tick_params(axis="x", labelsize=14,rotation=0)
    ax2.set_xticks(newpos)
    ax2.set_xticklabels(newlabel)
    ax2.set_xlabel('Wavelength '+u'(\u03bcm)',fontsize=18,color='black')
    ax2.set_xlim(ax1.get_xlim())

    if savefi!='no':
        plt.savefig(savefi,dpi=100,bbox_inches='tight')

