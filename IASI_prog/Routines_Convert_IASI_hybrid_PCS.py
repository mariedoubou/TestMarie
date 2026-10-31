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

# not used anymore: the Eigenvectorfile attributes are now copied from the PRP file
ev_dic = { 'M03_B1': 'IASI_EV1_xx_M03_20180101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx',
           'M03_B2': 'IASI_EV2_xx_M03_20180101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx',
           'M03_B3': 'IASI_EV3_xx_M03_20180101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx'
         }

#-------------------------------------
# compute the number of lines in an orbit that was splitted in PDUs
#-------------------------------------
def count_num_lines_in_several_pdus(directory_of_one_orbit,ope = 'no'):
    num_scan_lines = 0
  
    scan_lines_by_pdu = {}

    # get all directories containing PDUs for the orbit a
    if ope == 'no':
       dir_orb_in_pdu = glob.glob(directory_of_one_orbit+'/WO*')

 
    #create_IASI_PCS_netCDF(nfile, nc, scan_lines, f_sat)
    
    for pdu in dir_orb_in_pdu:
        print('pdu=',pdu)
        
        day = (dt.datetime.today() - dt.timedelta(days=1)).strftime("%Y%m%d")
    
        prpdir = pdu+'/to_pgf/*PRP*Z'
    
        prp = glob.glob(prpdir)
        prp.sort()
      
        #print('len(prp)=',len(prp))
        #print(prp[0])
 
        ### open the PRP file
        nc = nc4.Dataset(prp[0], "r")

        # find the num of scanline of that pdu
        scan_lines = len(nc['L1C/LineNumber'][:])
        print('scan_lines=',scan_lines)

        # add to the previous PDUs
        num_scan_lines = num_scan_lines + scan_lines
    
        print('dir_orb_in_pdu[pdu]=',os.path.basename(pdu),scan_lines)
        
        scan_lines_by_pdu[os.path.basename(pdu)] = scan_lines

        nc.close()
    print('there are:', num_scan_lines, ' in',os.path.basename(directory_of_one_orbit))
    return num_scan_lines, scan_lines_by_pdu


# last sentence of the summary attribute, for a PDU file and for a merged orbit file
summary_pdu   = 'Each file contains one PDU (3 minutes) of an IASI orbit.'
summary_orbit = 'Each file contains one full IASI orbit (Svalbard to Svalbard).'

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
    f.summary                      = 'This is the first release of the global + hybrid IASI Principle Component Score (PCS) from METOP. This dataset was reprocessed using the EUMETSAT operational EUMETSAT algorithm v6.5.4 of the EUMETSAT operational IASI level 2 processing chain. ' + summary_pdu
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
        f.cdr_platform_start_time  = dt.datetime.strftime(dt.datetime.strptime('20181107000000Z',fmt1),fmt)
        f.cdr_platform_stop_time   = dt.datetime.strftime(dt.datetime.strptime('20260630000000Z',fmt1),fmt)
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
    #nf.createDimension('B1P2', 41)
    nf.createDimension('B1P2', 55) #v2.1
    #nf.createDimension('B1P3', 48)
    nf.createDimension('B1P3', 34) #v2.1
    nf.createDimension('B2P1', 2)
    #nf.createDimension('B2P2', 61)
    nf.createDimension('B2P2', 47) #v2.1
    #nf.createDimension('B2P3', 57)
    nf.createDimension('B2P3', 71)
    nf.createDimension('B3P1', 1)
    #nf.createDimension('B3P2', 44)
    nf.createDimension('B3P2', 77) # v2.1
    #nf.createDimension('B3P3', 45)
    nf.createDimension('B3P3', 12) #v2.1
    
    nf.createDimension('B1PCH', 5)
    nf.createDimension('B2PCH', 5)
    nf.createDimension('B3PCH', 5)

    # per-PDU fields (local PCs, IPO statistics) use the 'pdu' dimension instead of 'scan_lines'
    # it is 1 in a PDU file and the number of PDUs in a merged orbit file
    nf.createDimension('pdu', None)
    nf.createDimension('outlier_threshold', 12)

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

    Maps = nf.createGroup('Maps')

    # PDU bookkeeping
    #---------------------------------------
    first_line = last_scanline - scan_lines # first line of this PDU in the orbit

    pdu_first_line = nf.createVariable('pdu_first_line', 'i4', ['pdu'])
    pdu_first_line.units = '1'
    pdu_first_line.long_name = 'First line of the PDU'
    pdu_first_line.description = 'Index (0-based) in the orbit of the first scan line of each PDU'
    pdu_first_line[:] = [first_line]

    pdu_num_lines = nf.createVariable('pdu_num_lines', 'i4', ['pdu'])
    pdu_num_lines.units = '1'
    pdu_num_lines.long_name = 'Number of lines in the PDU'
    pdu_num_lines.description = 'Number of scan lines of each PDU'
    pdu_num_lines[:] = [scan_lines]

    pdu_index = nf.createVariable('pdu_index', 'i4', ['scan_lines'], zlib=True, complevel=6)
    pdu_index.units = '1'
    pdu_index.long_name = 'PDU index of the line'
    pdu_index.description = 'Index (0-based) along the pdu dimension of the PDU each scan line belongs to, ' \
                            'e.g. PClocal/Band1/R_local[pdu_index[line]]'
    pdu_index[:] = np.zeros(scan_lines, dtype='i4') # offset by the PDU number when merging

    orbit_line = nf.createVariable('orbit_line', 'i4', ['scan_lines'], zlib=True, complevel=6)
    orbit_line.units = '1'
    orbit_line.long_name = 'Line number in orbit'
    orbit_line.description = 'Line number (0-based) in the orbit, i.e. pdu_first_line + LineNumber'
    orbit_line[:] = first_line + of['L1C/LineNumber'][:]

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
    SensingTime_day.units = 'day'
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
    SunAzimuth.coordinates = 'latitude longitude'
    SunAzimuth.units = 'degree'
    SunAzimuth.standard_name = 'solar_azimuth_angle'
    SunAzimuth.long_name = 'solar azimuth angle'
    SunAzimuth.valid_min = 0.
    SunAzimuth.valid_max = 360.
    SunAzimuth.description = 'solar azimuth angle'
    SunAzimuth[:] = of['L1C/SunAzimuth'][:]

    SunZenith = nf.createVariable('SunZenith', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                   fill_value=9.969209968386869e+36)
    SunZenith.coordinates = 'latitude longitude'
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
    RadAnalStd.description = 'Standard deviation of AVHRR radiance within the three main radiance analysis clusters'
    RadAnalStd[:] = of['L1C/Avhrr/RadAnalStd'][:]

    RadAnalWgt = Avhrr.createVariable('RadAnalWgt', 'f4', ['scan_lines', 'pixels', 'cluster'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    RadAnalWgt.coordinates = 'latitude longitude'
    RadAnalWgt.units = 'percent'
    RadAnalWgt.standard_name = 'area_fraction'
    RadAnalWgt.long_name = 'Fraction of IASI FOV covered by the three main AVHRR radiance analysis clusters'
    RadAnalWgt.valid_min = 0.
    RadAnalWgt.valid_max = 100.
    RadAnalWgt.description = 'Fraction of IASI FOV covered by the three main AVHRR radiance analysis clusters'
    RadAnalWgt[:] = of['L1C/Avhrr/RadAnalWgt'][:]

    T4_mean = Avhrr.createVariable('T4_mean', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    T4_mean.coordinates = 'latitude longitude'
    T4_mean.units = 'W/(m^2*sr*m^-1)'
    T4_mean.standard_name = 'mean_toa_outgoing_radiance_per_unit_wavelength'
    T4_mean.long_name = 'Mean radiance of AVHRR channel 4'
    T4_mean.valid_min = 0.
    T4_mean.description = 'Mean radiance of AVHRR channel 4'
    T4_mean[:] = of['L1C/Avhrr/T4_mean'][:]

    T4_std = Avhrr.createVariable('T4_std', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    T4_std.coordinates = 'latitude longitude'
    T4_std.units = 'W/(m^2*sr*m^-1)'
    T4_std.standard_name = 'std_toa_outgoing_radiance_per_unit_wavelength'
    T4_std.long_name = 'Standard deviation of radiance of AVHRR channel 4'
    T4_std.valid_min = 0.
    T4_std.description = 'Standard deviation of radiance of AVHRR channel 4'
    T4_std[:] = of['L1C/Avhrr/T4_std'][:]

    T5_mean = Avhrr.createVariable('T5_mean', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    T5_mean.coordinates = 'latitude longitude'
    T5_mean.units = 'W/(m^2*sr*m^-1)'
    T5_mean.standard_name = 'mean_toa_outgoing_radiance_per_unit_wavelength'
    T5_mean.long_name = 'Mean radiance of AVHRR channel 5'
    T5_mean.valid_min = 0.
    T5_mean.description = 'Mean radiance of AVHRR channel 5'
    T5_mean[:] = of['L1C/Avhrr/T5_mean'][:]

    T5_std = Avhrr.createVariable('T5_std', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    T5_std.coordinates = 'latitude longitude'
    T5_std.units = 'W/(m^2*sr*m^-1)'
    T5_std.standard_name = 'std_toa_outgoing_radiance_per_unit_wavelength'
    T5_std.long_name = 'Standard deviation of radiance of AVHRR channel 5'
    T5_std.valid_min = 0.
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
    RadianceSum.description = 'Sum of reconstructed radiances in each band'
    RadianceSum[:] = of['L1C/PCscores/RadianceSum'][:]

    ResidualRms = PCscores.createVariable('ResidualRms', 'f4', ['scan_lines', 'pixels', 'cluster'], zlib=True,
                                       complevel=6, fill_value=9.969209968386869e+36)
    ResidualRms.coordinates = 'latitude longitude'
    ResidualRms.units = 'N/A'
    ResidualRms.standard_name = 'N/A'
    ResidualRms.long_name = 'Noise normalized residual RMS in each band'
    #ResidualRms.valid_min = 0.
    ResidualRms.description = 'Noise normalized residual RMS in each band'
    ResidualRms[:] = of['L1C/PCscores/ResidualRms'][:]

    # PCscore Band 1 group
    Band1.Eigenvectorfile = of['L1C/PCscores/Band1'].Eigenvectorfile
    scale_band1 = of['L1C/PCscores/Band1'].scale_factor # score quantisation factor (0.5)
    P1_band1 = Band1.createVariable('P1', 'i4', ['scan_lines', 'pixels', 'B1P1'], zlib=True,
                                       complevel=6)
    P1_band1.coordinates = 'latitude longitude'
    P1_band1.units = 'N/A'
    P1_band1.standard_name = 'N/A'
    P1_band1.long_name = 'Principle Component 1 for band 1'
    P1_band1.scale_factor = scale_band1
    P1_band1.set_auto_scale(False) # write the raw quantised scores
    ## P1_band1.valid_min = 0.
    P1_band1.description = 'Principle Component 1 for band 1'
    P1_band1[:] = of['L1C/PCscores/Band1/P1'][:] #* 0.5

    P2_band1 = Band1.createVariable('P2', 'i4', ['scan_lines', 'pixels', 'B1P2'], zlib=True,
                                       complevel=6)
    P2_band1.coordinates = 'latitude longitude'
    P2_band1.units = 'N/A'
    P2_band1.standard_name = 'N/A'
    P2_band1.long_name = 'Principle Component 2 for band 1'
    P2_band1.scale_factor = scale_band1
    P2_band1.set_auto_scale(False) # write the raw quantised scores
    ## P2_band1.valid_min = 0.
    P2_band1.description = 'Principle Component 2 for band 1'
    P2_band1[:] = of['L1C/PCscores/Band1/P2'][:] #* 0.5

    P3_band1 = Band1.createVariable('P3', 'i4', ['scan_lines', 'pixels', 'B1P3'], zlib=True,
                                       complevel=6)
    P3_band1.coordinates = 'latitude longitude'
    P3_band1.units = 'N/A'
    P3_band1.standard_name = 'N/A'
    P3_band1.long_name = 'Principle Component 3 for band 1'
    P3_band1.scale_factor = scale_band1
    P3_band1.set_auto_scale(False) # write the raw quantised scores
    ## P3_band1.valid_min = 0.
    P3_band1.description = 'Principle Component 3 for band 1'
    P3_band1[:] = of['L1C/PCscores/Band1/P3'][:] #* 0.5

    # PCscore Band 2 group
    Band2.Eigenvectorfile = of['L1C/PCscores/Band2'].Eigenvectorfile
    scale_band2 = of['L1C/PCscores/Band2'].scale_factor # score quantisation factor (0.5)
    P1_band2 = Band2.createVariable('P1', 'i4', ['scan_lines', 'pixels', 'B2P1'], zlib=True,
                                       complevel=6)
    P1_band2.coordinates = 'latitude longitude'
    P1_band2.units = 'N/A'
    P1_band2.standard_name = 'N/A'
    P1_band2.long_name = 'Principle Component 1 for band 2'
    P1_band2.scale_factor = scale_band2
    P1_band2.set_auto_scale(False) # write the raw quantised scores
    ## P1_band2.valid_min = 0.
    P1_band2.description = 'Principle Component 1 for band 2'
    P1_band2[:] = of['L1C/PCscores/Band2/P1'][:] #* 0.5

    P2_band2 = Band2.createVariable('P2', 'i4', ['scan_lines', 'pixels', 'B2P2'], zlib=True,
                                       complevel=6)
    P2_band2.coordinates = 'latitude longitude'
    P2_band2.units = 'N/A'
    P2_band2.standard_name = 'N/A'
    P2_band2.long_name = 'Principle Component 2 for band 2'
    P2_band2.scale_factor = scale_band2
    P2_band2.set_auto_scale(False) # write the raw quantised scores
    #P2_band2.valid_min = 0.
    P2_band2.description = 'Principle Component 2 for band 2'
    P2_band2[:] = of['L1C/PCscores/Band2/P2'][:] #* 0.5

    P3_band2 = Band2.createVariable('P3', 'i4', ['scan_lines', 'pixels', 'B2P3'], zlib=True,
                                       complevel=6)
    P3_band2.coordinates = 'latitude longitude'
    P3_band2.units = 'N/A'
    P3_band2.standard_name = 'N/A'
    P3_band2.long_name = 'Principle Component 3 for band 2'
    P3_band2.scale_factor = scale_band2
    P3_band2.set_auto_scale(False) # write the raw quantised scores
    ## P3_band2.valid_min = 0.
    P3_band2.description = 'Principle Component 3 for band 2'
    P3_band2[:] = of['L1C/PCscores/Band2/P3'][:] #* 0.5

    # PCscore Band 3 group
    Band3.Eigenvectorfile = of['L1C/PCscores/Band3'].Eigenvectorfile
    scale_band3 = of['L1C/PCscores/Band3'].scale_factor # score quantisation factor (0.5)
    P1_band3 = Band3.createVariable('P1', 'i4', ['scan_lines', 'pixels', 'B3P1'], zlib=True,
                                       complevel=6)
    P1_band3.coordinates = 'latitude longitude'
    P1_band3.units = 'N/A'
    P1_band3.standard_name = 'N/A'
    P1_band3.long_name = 'Principle Component 1 for band 3'
    P1_band3.scale_factor = scale_band3
    P1_band3.set_auto_scale(False) # write the raw quantised scores
    ## P1_band3.valid_min = 0.
    P1_band3.description = 'Principle Component 1 for band 3'
    P1_band3[:] = of['L1C/PCscores/Band3/P1'][:] #* 0.5

    P2_band3 = Band3.createVariable('P2', 'i4', ['scan_lines', 'pixels', 'B3P2'], zlib=True,
                                       complevel=6)
    P2_band3.coordinates = 'latitude longitude'
    P2_band3.units = 'N/A'
    P2_band3.standard_name = 'N/A'
    P2_band3.long_name = 'Principle Component 2 for band 3'
    P2_band3.scale_factor = scale_band3
    P2_band3.set_auto_scale(False) # write the raw quantised scores
    ## P2_band3.valid_min = 0.
    P2_band3.description = 'Principle Component 2 for band 3'
    P2_band3[:] = of['L1C/PCscores/Band3/P2'][:] #* 0.5

    P3_band3 = Band3.createVariable('P3', 'i4', ['scan_lines', 'pixels', 'B3P3'], zlib=True,
                                       complevel=6)
    P3_band3.coordinates = 'latitude longitude'
    P3_band3.units = 'N/A'
    P3_band3.standard_name = 'N/A'
    P3_band3.long_name = 'Principle Component 3 for band 3'
    P3_band3.scale_factor = scale_band3
    P3_band3.set_auto_scale(False) # write the raw quantised scores
    ## P3_band3.valid_min = 0.
    P3_band3.description = 'Principle Component 3 for band 3'
    P3_band3[:] = of['L1C/PCscores/Band3/P3'][:] #* 0.5

    #---------------------------------------
    # add hybrid PCS
    #
    # hybrid PCscore Band 1 group
    #---------------------------------------
    #Band1.Eigenvectorfile = ev_dic[sat + '_B1']
    Elocal_band1_hyb = hybrid_Band1.createVariable('E_local', 'f8', ['pdu','B1PCH', 'chans_in_band1'], zlib=True,
                                       complevel=6)
    #P1_band1_hyb.coordinates = 'latitude longitude'
    Elocal_band1_hyb.units = 'N/A'
    Elocal_band1_hyb.standard_name = 'N/A'
    Elocal_band1_hyb.long_name = 'E_local for band 1'
    Elocal_band1_hyb.description = 'Local eigen vector for band 1'
    Elocal1 = ofpch['band_1/E_local'][:]    # (5, 1997)
    #print(type(Elocal))
    #print(Elocal.shape)
    #print(type(Elocal_band1_hyb))
    #print(Elocal_band1_hyb.shape)
    #Elocal_band1_hyb[:,:,:] = Elocal1[np.newaxis, :, :]
    Elocal_band1_hyb[:] = Elocal1[np.newaxis, :, :] # one per PDU

    
    plocal_band1_hyb = hybrid_Band1.createVariable('p_local', 'i4', ['scan_lines', 'pixels', 'B1PCH'], zlib=True,
                                       complevel=6)
    plocal_band1_hyb.coordinates = 'latitude longitude'
    plocal_band1_hyb.units = 'N/A'
    plocal_band1_hyb.standard_name = 'N/A'
    plocal_band1_hyb.long_name = 'p_local for band 1'
    plocal_band1_hyb.description = 'p_local for band 1'
    plocal_band1_hyb[:] = ofpch['band_1/p_local'][:] 

    Rlocal_band1_hyb = hybrid_Band1.createVariable('R_local', 'f8', ['pdu','B1PCH', 'chans_in_band1'], zlib=True,
                                       complevel=6)
    Rlocal_band1_hyb.units = 'N/A'
    Rlocal_band1_hyb.standard_name = 'N/A'
    Rlocal_band1_hyb.long_name = 'R_local for band 1'
    Rlocal_band1_hyb.description = 'reconstruction factor (R_local = N (normalisation from EV file) *E_local) for band 1'
    Rlocal1 = ofpch['band_1/R_local'][:]    # (5, 1997)
    Rlocal_band1_hyb[:] = Rlocal1[np.newaxis, :, :] # one per PDU
    
    rmslocal_band1_hyb = hybrid_Band1.createVariable('rms_local', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6)
    rmslocal_band1_hyb.coordinates = 'latitude longitude'
    rmslocal_band1_hyb.units = 'N/A'
    rmslocal_band1_hyb.standard_name = 'N/A'
    rmslocal_band1_hyb.long_name = 'rms_local for band 1'
    rmslocal_band1_hyb.description = 'rms_local for band 1'
    rmslocal_band1_hyb[:] = ofpch['band_1/rms_local'][:]

    Vlocal_band1_hyb = hybrid_Band1.createVariable('V_local', 'f8', ['pdu','B1PCH'], zlib=True,
                                       complevel=6)
    #rmslocal_band1_hyb.coordinates = 'latitude longitude'
    Vlocal_band1_hyb.units = 'N/A'
    Vlocal_band1_hyb.standard_name = 'N/A'
    Vlocal_band1_hyb.long_name = 'V_local for band 1'
    Vlocal_band1_hyb.description = 'scaling/eigen value for band 1'
    Vlocal1 = ofpch['band_1/V_local'][:]    # (5, 1997)
    Vlocal_band1_hyb[:] = Vlocal1[np.newaxis, :] # one per PDU

    #
    # hybrid PCscore Band 2 group
    #---------------------------------------
    #Band1.Eigenvectorfile = ev_dic[sat + '_B1']
    Elocal_band2_hyb = hybrid_Band2.createVariable('E_local', 'f8', ['pdu','B2PCH', 'chans_in_band2'], zlib=True,
                                       complevel=6)
    #P1_band1_hyb.coordinates = 'latitude longitude'
    Elocal_band2_hyb.units = 'N/A'
    Elocal_band2_hyb.standard_name = 'N/A'
    Elocal_band2_hyb.long_name = 'E_local for band 2'
    Elocal_band2_hyb.description = 'Local eigen vector for band 2'
    Elocal2 = ofpch['band_2/E_local'][:]    # (5, 3119)
    Elocal_band2_hyb[:] = Elocal2[np.newaxis, :, :] # one per PDU

    plocal_band2_hyb = hybrid_Band2.createVariable('p_local', 'i4', ['scan_lines', 'pixels', 'B2PCH'], zlib=True,
                                       complevel=6)
    plocal_band2_hyb.coordinates = 'latitude longitude'
    plocal_band2_hyb.units = 'N/A'
    plocal_band2_hyb.standard_name = 'N/A'
    plocal_band2_hyb.long_name = 'p_local for band 2'
    plocal_band2_hyb.description = 'p_local for band 2'
    plocal_band2_hyb[:] = ofpch['band_2/p_local'][:] 

    Rlocal_band2_hyb = hybrid_Band2.createVariable('R_local', 'f8', ['pdu','B2PCH', 'chans_in_band2'], zlib=True,
                                       complevel=6)
    Rlocal_band2_hyb.units = 'N/A'
    Rlocal_band2_hyb.standard_name = 'N/A'
    Rlocal_band2_hyb.long_name = 'R_local for band 2'
    Rlocal_band2_hyb.description = 'reconstruction factor (R_local = N (normalisation from EV file) *E_local) for band 2'
    Rlocal2 = ofpch['band_2/R_local'][:]    # (5, 3119 )
    Rlocal_band2_hyb[:] = Rlocal2[np.newaxis, :, :] # one per PDU

    rmslocal_band2_hyb = hybrid_Band2.createVariable('rms_local', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6)
    rmslocal_band2_hyb.coordinates = 'latitude longitude'
    rmslocal_band2_hyb.units = 'N/A'
    rmslocal_band2_hyb.standard_name = 'N/A'
    rmslocal_band2_hyb.long_name = 'rms_local for band 2'
    rmslocal_band2_hyb.description = 'rms_local for band 2'
    rmslocal_band2_hyb[:] = ofpch['band_2/rms_local'][:]

    Vlocal_band2_hyb = hybrid_Band2.createVariable('V_local', 'f8', ['pdu','B2PCH'], zlib=True,
                                       complevel=6)
    #rmslocal_band1_hyb.coordinates = 'latitude longitude'
    Vlocal_band2_hyb.units = 'N/A'
    Vlocal_band2_hyb.standard_name = 'N/A'
    Vlocal_band2_hyb.long_name = 'V_local for band 2'
    Vlocal_band2_hyb.description = 'scaling/eigen value for band 2'  
    Vlocal2= ofpch['band_2/V_local'][:]    # (5, 3119)
    Vlocal_band2_hyb[:] = Vlocal2[np.newaxis, :] # one per PDU

    #
    # hybrid PCscore Band 3 group
    #---------------------------------------
    Elocal_band3_hyb = hybrid_Band3.createVariable('E_local', 'f8', ['pdu','B3PCH', 'chans_in_band3'], zlib=True,
                                       complevel=6)
    Elocal_band3_hyb.units = 'N/A'
    Elocal_band3_hyb.standard_name = 'N/A'
    Elocal_band3_hyb.long_name = 'E_local for band 3'
    Elocal_band3_hyb.description = 'Local eigen vector for band 3'
    Elocal3 = ofpch['band_3/E_local'][:]    # (5, 3345)
    Elocal_band3_hyb[:] = Elocal3[np.newaxis, :, :] # one per PDU
 
    plocal_band3_hyb = hybrid_Band3.createVariable('p_local', 'i4', ['scan_lines', 'pixels', 'B3PCH'], zlib=True,
                                       complevel=6)
    plocal_band3_hyb.coordinates = 'latitude longitude'
    plocal_band3_hyb.units = 'N/A'
    plocal_band3_hyb.standard_name = 'N/A'
    plocal_band3_hyb.long_name = 'p_local for band 3'
    plocal_band3_hyb.description = 'p_local for band 3'
    plocal_band3_hyb[:] = ofpch['band_3/p_local'][:] 

    Rlocal_band3_hyb = hybrid_Band3.createVariable('R_local', 'f8', ['pdu','B3PCH', 'chans_in_band3'], zlib=True,
                                       complevel=6)
    Rlocal_band3_hyb.units = 'N/A'
    Rlocal_band3_hyb.standard_name = 'N/A'
    Rlocal_band3_hyb.long_name = 'R_local for band 3'
    Rlocal_band3_hyb.description = 'reconstruction factor (R_local = N (normalisation from EV file) *E_local) for band 3'
    Rlocal3 = ofpch['band_3/R_local'][:]    # (5, 3345 )
    Rlocal_band3_hyb[:] = Rlocal3[np.newaxis, :, :] # one per PDU

    rmslocal_band3_hyb = hybrid_Band3.createVariable('rms_local', 'f4', ['scan_lines', 'pixels'], zlib=True,
                                       complevel=6)
    rmslocal_band3_hyb.coordinates = 'latitude longitude'
    rmslocal_band3_hyb.units = 'N/A'
    rmslocal_band3_hyb.standard_name = 'N/A'
    rmslocal_band3_hyb.long_name = 'rms_local for band 3'
    rmslocal_band3_hyb.description = 'rms_local for band 3'
    rmslocal_band3_hyb[:] = ofpch['band_3/rms_local'][:]

    Vlocal_band3_hyb = hybrid_Band3.createVariable('V_local', 'f8', ['pdu','B3PCH'], zlib=True,
                                       complevel=6)
    Vlocal_band3_hyb.units = 'N/A'
    Vlocal_band3_hyb.standard_name = 'N/A'
    Vlocal_band3_hyb.long_name = 'V_local for band 3'
    Vlocal_band3_hyb.description = 'scaling/eigen value for band 3'
    Vlocal3= ofpch['band_3/V_local'][:]    # (5, 3345)
    Vlocal_band3_hyb[:] = Vlocal3[np.newaxis, :] # one per PDU

    # add statistics
    #---------------------
    stat_res_mean = IPOgroup.createVariable('ResidualMean', 'f4', ['pdu', 'num_chans'], zlib=True, complevel=6)
    #stat_res_mean.coordinates = 'latitude'
    stat_res_mean.units = '1'
    stat_res_mean.standard_name = 'IPOresmean'
    stat_res_mean.long_name = 'IPO Statistics/ResidualMean'
    #stat_res_mean.valid_min = -1000.
    #stat_res_mean.valid_max = 1000.
    stat_res_mean.description = 'IPO Statistics/Residual Mean'

    res_mean = ofipo['Statistics']['ResidualMean'][:]    # (8461)
    stat_res_mean[:] = res_mean[np.newaxis,:]

    #---------------------
    stat_res_stdv = IPOgroup.createVariable('ResidualStdv', 'f4', ['pdu', 'num_chans'], zlib=True, complevel=6)
    #stat_res_mean.coordinates = 'latitude'
    stat_res_stdv.units = '1'
    stat_res_stdv.standard_name = 'IPOresstdv'
    stat_res_stdv.long_name = 'IPO Statistics ResidualStdv'
    #stat_res_stdv.valid_min = -1000.
    #stat_res_stdv.valid_max = 1000.
    stat_res_stdv.description = 'IPO Statistics Residual Standard deviation'

    res_stdv = ofipo['Statistics']['ResidualStdv'][:]    # (8461)
    stat_res_stdv[:] = res_stdv[np.newaxis, :]

    # per-PDU IPO metadata
    #---------------------
    NbrCases = IPOgroup.createVariable('NbrCases', 'i4', ['pdu'])
    NbrCases.units = '1'
    NbrCases.long_name = 'IPO Statistics NbrCases'
    NbrCases.description = 'Number of cases used for the IPO residual statistics'
    NbrCases[:] = ofipo['Statistics'].getncattr('/NbrCases')

    NbrOutliers = IPOgroup.createVariable('NbrOutliers', 'i4', ['pdu'])
    NbrOutliers.units = '1'
    NbrOutliers.long_name = 'Number of outliers'
    NbrOutliers.description = 'Number of outlier pixels in the PDU'
    NbrOutliers[:] = ofipo.NbrOutliers

    OutlierThreshold = IPOgroup.createVariable('OutlierThreshold', 'f4', ['pdu', 'outlier_threshold'])
    OutlierThreshold.units = '1'
    OutlierThreshold.long_name = 'Outlier threshold'
    OutlierThreshold.description = 'IPO outlier threshold'
    OutlierThreshold[:] = ofipo.OutlierThreshold[np.newaxis, :]

    OutlierSlope = IPOgroup.createVariable('OutlierSlope', 'f4', ['pdu', 'cluster'])
    OutlierSlope.units = '1'
    OutlierSlope.long_name = 'Outlier slope'
    OutlierSlope.description = 'IPO outlier slope in each band'
    OutlierSlope[:] = ofipo.OutlierSlope[np.newaxis, :]

    # constant over the orbit
    IPOgroup.NbrScores = ofipo.NbrScores
    IPOgroup.ScoreQuantisationFactor = ofipo.ScoreQuantisationFactor
    IPOgroup.ResidualQuantisationFactor = ofipo.ResidualQuantisationFactor

    L1cRadiances = IPOgroup.createVariable('IPO_L1cRadiances', 'f4', ['scan_lines','pixels','num_chans'], zlib=True, 
                                     complevel=6, fill_value=np.float32(-9999.0),shuffle=True, chunksizes=(1, 1, 1024))
    L1cRadiances.coordinates = 'latitude longitude'
    L1cRadiances.units = '1'
    L1cRadiances.standard_name = 'IPO_L1cRadiances'
    L1cRadiances.long_name = 'IPO L1cRadiances'
    #L1cRadiances.valid_min = -1000.
    #L1cRadiances.valid_max = 1000.
    L1cRadiances.description = 'IPO L1cRadiances'
    # Initialize the NetCDF variable
    #L1cRadiances[:] = -9999.0

    #if 'L1cRadiance' in ncipo.variables:
    if IPO == 'yes':
        pix    = ofipo['PixelNumber'][:]
        lin    = ofipo['LineNumber'][:]
        L1cRad = ofipo['L1cRadiance'][:]
        print('L1CRad shape=',L1cRad.shape)

        print('\n========== DEBUG L1cRadiances ==========')
        print('L1cRadiances:', L1cRadiances)
        print('dimensions  :', L1cRadiances.dimensions)
        print('shape       :', L1cRadiances.shape)
        print('ndim        :', L1cRadiances.ndim)

        print('\nL1cRad:')
        print('shape       :', L1cRad.shape)
        print('ndim        :', L1cRad.ndim)
        print('dtype       :', L1cRad.dtype)

        print('\nIndices:')
        print('lin         :', lin)
        print('pix         :', pix)
        print('========================================\n')

       # L1cRadiances[lin, pix, :] = L1cRad
        #for i, (l, p) in enumerate(zip(lin, pix)):
        for i, (l, p) in enumerate(zip(lin, pix)):
              l = int(l)
              p = int(p)
              L1cRadiances[l, p, :] = L1cRad[i, :]

    # stored as the raw int8 of the IPO file, unpacked automatically on read
    res_scale = ofipo.ResidualQuantisationFactor
    if not np.all(res_scale == res_scale[0]):
        raise ValueError('ResidualQuantisationFactor differs between bands: %s' % res_scale)
    PccResidual = IPOgroup.createVariable('IPO_PccResidual', 'i1', ['scan_lines','pixels','num_chans'], zlib=True, 
                                     complevel=6, fill_value=np.int8(-128),shuffle=True, chunksizes=(1, 1, 1024))
    PccResidual.coordinates = 'latitude longitude'
    PccResidual.units = '1'
    PccResidual.standard_name = 'IPO_PccResidual'
    PccResidual.long_name = 'IPO PccResidual'
    #L1cRadiances.valid_min = -1000.
    #L1cRadiances.valid_max = 1000.
    PccResidual.description = 'IPO PccResidual, noise normalised residual of the PC compression'
    PccResidual.scale_factor = res_scale[0]
    PccResidual.set_auto_scale(False) # write the raw quantised residuals
    # Initialize the NetCDF variable
    #PccResidual[:] = -9999.0

    #if 'PccResidual' in ncipo.variables:
    if IPO == 'yes':
        pix    = ofipo['PixelNumber'][:]
        lin    = ofipo['LineNumber'][:]
        PccRes = ofipo['PccResidual'][:]

        #PccResidual[lin, pix, :] = PccRes

        for i, (l, p) in enumerate(zip(lin, pix)):
              l = int(l)
              p = int(p)
              PccResidual[l, p, :] = PccRes[i, :]
        
        
    # IPO ResidualRMS is not copied: it is identical to PCscores/ResidualRms at the outlier pixels
    outlier_flag = IPOgroup.createVariable('outlier_flag', 'u1', ['scan_lines','pixels'], zlib=True, complevel=6)
    outlier_flag.coordinates = 'latitude longitude'
    outlier_flag.units = '1'
    outlier_flag.long_name = 'IPO outlier flag'
    outlier_flag.valid_min = 0
    outlier_flag.valid_max = 1
    outlier_flag.description = '1 if the pixel is an IPO outlier (IPO_L1cRadiances and IPO_PccResidual available), 0 otherwise'
    flag = np.zeros((scan_lines, pixels), dtype='u1')

    GQisFlagQual = IPOgroup.createVariable('GQisFlagQual', 'u1', ['scan_lines','pixels','cluster'], zlib=True,
                                           complevel=6, fill_value=np.uint8(255))
    GQisFlagQual.coordinates = 'latitude longitude'
    GQisFlagQual.units = '1'
    GQisFlagQual.long_name = 'IPO GQisFlagQual'
    GQisFlagQual.description = 'IPO GQisFlagQual in each band, for the outlier pixels'

    if IPO == 'yes':
        pix    = ofipo['PixelNumber'][:]
        lin    = ofipo['LineNumber'][:]
        flag[lin, pix] = 1

        # GQisFlagQual is an HDF5 bitfield, not seen by netCDF4: read it with h5py
        with h5py.File(ofipo.filepath(), 'r') as h5ipo:
            dset = h5ipo['GQisFlagQual']
            gqis = np.empty(dset.shape, dtype='u1')
            dset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, gqis, mtype=dset.id.get_type())
        gq = np.full((scan_lines, pixels, cluster), 255, dtype='u1')
        gq[lin, pix, :] = gqis
        GQisFlagQual[:] = gq

    outlier_flag[:] = flag

    # Maps group
    #---------------------
    Height = Maps.createVariable('Height', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                 fill_value=9.969209968386869e+36)
    Height.coordinates = 'latitude longitude'
    Height.units = 'm'
    Height.standard_name = 'surface_altitude'
    Height.long_name = 'Surface height of IASI pixel'
    Height.description = 'Surface height of the IASI pixel'
    Height[:] = of['Maps/Height'][:]

    HeightStd = Maps.createVariable('HeightStd', 'f4', ['scan_lines', 'pixels'], zlib=True, complevel=6,
                                    fill_value=9.969209968386869e+36)
    HeightStd.coordinates = 'latitude longitude'
    HeightStd.units = 'm'
    HeightStd.long_name = 'Standard deviation of surface height in IASI pixel'
    HeightStd.description = 'Standard deviation of the surface height within the IASI pixel'
    HeightStd[:] = of['Maps/HeightStd'][:]

    LandFraction_maps = Maps.createVariable('LandFraction', 'u1', ['scan_lines', 'pixels'], zlib=True, complevel=6)
    LandFraction_maps.coordinates = 'latitude longitude'
    LandFraction_maps.units = 'percent'
    LandFraction_maps.standard_name = 'land_area_fraction'
    LandFraction_maps.long_name = 'Land fraction in IASI pixel from Maps'
    LandFraction_maps.valid_min = 0
    LandFraction_maps.valid_max = 100
    LandFraction_maps.description = 'Land fraction in IASI pixel from the PRP Maps group (differs from the L1C one)'
    LandFraction_maps[:] = of['Maps/LandFraction'][:]


# ### merge_pdu_files
#---------------------
# merge the PDU files of one orbit into one orbit file
# variables with a 'scan_lines' dimension are concatenated along the lines,
# variables with a 'pdu' dimension (per-PDU fields) along the PDUs,
# the others are taken from the first file
#---------------------
CONCAT_DIMS = ('scan_lines', 'pdu')
CHUNK = 10  # process 10 scan_lines at a time to stay low on memory

def create_dimensions(src_root, dst_root):
    """Create all dimensions at root level (where they are defined)."""
    for name, dim in src_root.dimensions.items():
        if name not in dst_root.dimensions:
            if name in CONCAT_DIMS:
                dst_root.createDimension(name, None)  # unlimited
            else:
                dst_root.createDimension(name, len(dim))

def create_variables_in_group(src_grp, dst_grp):
    """Create all variables in a group (without writing data yet)."""
    for name, var in src_grp.variables.items():
        if name not in dst_grp.variables:
            atts = {a: var.getncattr(a) for a in var.ncattrs()}
            dst_var = dst_grp.createVariable(
                name, var.dtype, var.dimensions,
                zlib=True, complevel=6, fill_value=atts.pop('_FillValue', None)
            )
            # Preserve ALL attributes including coordinates and scale_factor
            dst_var.setncatts(atts)

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

def append_group_data(src_grp, dst_grp, offsets):
    """Recursively append data from src to dst along scan_lines or pdu, chunk by chunk."""
    for name, var in src_grp.variables.items():
        dst_var = dst_grp.variables[name]
        concat = [d for d in var.dimensions if d in CONCAT_DIMS]
        if concat:
            axis = var.dimensions.index(concat[0])
            offset = offsets[concat[0]]
            n = var.shape[axis]
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                # source slice
                src_idx = [slice(None)] * var.ndim
                src_idx[axis] = slice(start, end)
                # destination slice
                dst_idx = [slice(None)] * var.ndim
                dst_idx[axis] = slice(offset + start, offset + end)
                data = var[tuple(src_idx)]
                if name == 'pdu_index':
                    data = data + offsets['pdu']
                dst_var[tuple(dst_idx)] = data
        else:
            # no scan_lines or pdu dim: write only once (from first file)
            if offsets['pdu'] == 0:
                dst_var[:] = var[:]

    for grp_name, sub_grp in src_grp.groups.items():
        append_group_data(sub_grp, dst_grp.groups[grp_name], offsets)

def merge_pdu_files(list_file, outfile):
    """Merge the PDU files of list_file (sorted in time) into outfile."""
    with nc4.Dataset(outfile, 'w') as dst:
        # Copy global attributes from first file
        with nc4.Dataset(list_file[0], 'r') as src:
            dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
            create_dimensions(src, dst)
            create_groups_recursive(src, dst)
        # copy the raw (packed) values, scale_factor is copied as attribute
        dst.set_auto_maskandscale(False)

        # Append data file by file
        offsets = {'scan_lines': 0, 'pdu': 0}
        for i, f in enumerate(list_file):
            print(f"Appending file {i+1}/{len(list_file)}: {f}  (offsets={offsets})")
            with nc4.Dataset(f, 'r') as src:
                src.set_auto_maskandscale(False)
                append_group_data(src, dst, offsets)
                offsets['scan_lines'] += src.dimensions['scan_lines'].size
                offsets['pdu'] += src.dimensions['pdu'].size

        # Correct global sensing times
        with nc4.Dataset(list_file[0], 'r') as src_first:
            dst.setncattr('sensing_start_time', src_first.getncattr('sensing_start_time'))
        with nc4.Dataset(list_file[-1], 'r') as src_last:
            dst.setncattr('sensing_end_time', src_last.getncattr('sensing_end_time'))
        dst.setncattr('product_name', os.path.basename(outfile))
        dst.setncattr('summary', dst.getncattr('summary').replace(summary_pdu, summary_orbit))

    print(f"Done → {outfile}  (total scan_lines: {offsets['scan_lines']}, pdus: {offsets['pdu']})")
