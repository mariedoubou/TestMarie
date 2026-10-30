#!/usr/bin/env python
# coding: utf-8

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

import importlib

import Routines_Convert_IASI_PCS_R2_to_nc as iasi
import matplotlib.pyplot as plt
import Routine_cartopy_map as rmap
import matplotlib.pyplot as plt
from iasi_reader_Carlos import NativeIASI1c
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#importlib.reload(iasi) # to reload if needed

wn = np.linspace(644.75, 2759.75, 8461)
# in wavelength
wl = 1.E4/wn

# # def

def reconstructGPT(pcs, R1, R2, R3, chunk_lines=5):
    n_lines, n_pixels, n_pc = pcs.shape

    if n_pc == 300:
        bands = [
            (0, 90, R1),
            (90, 210, R2),
            (210, 300, R3),
        ]
    elif n_pc == 15:
        bands = [
            (0, 5, R1),
            (5, 10, R2),
            (10, 15, R3),
        ]
    else:
        raise ValueError("Unexpected number of PCs")

    n_channels = sum(R.shape[1] for _, _, R in bands)
    print('n_channels=',n_channels)

    # Use float32 to keep memory requirements manageable
    output = np.empty(
        (n_lines, n_pixels, n_channels),
        dtype=np.float32
    )

    for start in range(0, n_lines, chunk_lines):
        end = min(start + chunk_lines, n_lines)

        # Process one chunk of scanlines
        pc_chunk = np.asarray(pcs[start:end].data)

        spectra = []
        for i, j, R in bands:
            pc_band = pc_chunk[:, :, i:j].reshape(-1, j-i)

            # Matrix multiplication over all pixels in this chunk
            x = pc_band.astype(np.float32) @ np.asarray(R, dtype=np.float32)

            spectra.append(x.reshape(end-start, n_pixels, -1))

        output[start:end] = np.concatenate(spectra, axis=2)

        print(f"Processed scanlines {start} to {end-1}")

    return output


def reconstructLocalGPT(pcs, Rl1, Rl2, Rl3, chunk_lines=5):

    n_lines, n_pixels, n_pc = pcs.shape

    # Each band has 5 PCs
    if n_pc != 15:
        raise ValueError(f"Expected 15 PCs, got {n_pc}")

    if Rl1.shape != (n_lines, 5, 1997):
        raise ValueError(f"Unexpected Rl1 shape: {Rl1.shape}")

    if Rl2.shape != (n_lines, 5, 3119):
        raise ValueError(f"Unexpected Rl2 shape: {Rl2.shape}")

    if Rl3.shape != (n_lines, 5, 3345):
        raise ValueError(f"Unexpected Rl3 shape: {Rl3.shape}")

    bands = [
        (0,  5,  Rl1),
        (5, 10,  Rl2),
        (10, 15, Rl3),
    ]

    n_channels = Rl1.shape[2] + Rl2.shape[2] + Rl3.shape[2]

    print("n_channels =", n_channels)

    output = np.empty(
        (n_lines, n_pixels, n_channels),
        dtype=np.float32
    )

    for start in range(0, n_lines, chunk_lines):
        end = min(start + chunk_lines, n_lines)

        # PCs for this chunk: (chunk_lines, n_pixels, 15)
        pc_chunk = np.asarray(pcs[start:end].data, dtype=np.float32)

        spectra = []

        for i, j, Rl in bands:
            print('i,j,Rl.shape=',i,j,Rl.shape)
            # PCs for this band: (chunk_lines, n_pixels, 5)
            pc_band = pc_chunk[:, :, i:j]

            # Line-specific reconstruction matrices:
            # (chunk_lines, 5, n_channels_in_band)
            R_chunk = np.asarray(Rl[start:end], dtype=np.float32)

            # Batched matrix multiplication:
            # (chunk_lines, n_pixels, 5)
            # @ (chunk_lines, 5, n_channels_in_band)
            # -> (chunk_lines, n_pixels, n_channels_in_band)
            x = np.matmul(pc_band, R_chunk)

            spectra.append(x)

        # Concatenate the three spectral bands
        output[start:end] = np.concatenate(spectra, axis=2)

        print(f"Processed scanlines {start} to {end-1}")

    return output



def reconstruct_local_pcs(hpcs, Rl1, Rl2, Rl3):

    n_lines, n_pixels, _ = hpcs.shape

    n_channels = (
        Rl1.shape[2] +
        Rl2.shape[2] +
        Rl3.shape[2]
    )

    output = np.empty(
        (n_lines, n_pixels, n_channels),
        dtype=np.float32
    )

    for iline in range(n_lines):

        pcs1 = np.asarray(hpcs[iline,:,0:5], dtype=np.float32)
        pcs2 = np.asarray(hpcs[iline,:,5:10], dtype=np.float32)
        pcs3 = np.asarray(hpcs[iline,:,10:15], dtype=np.float32)

        ev1 = np.asarray(Rl1[iline], dtype=np.float32)
        ev2 = np.asarray(Rl2[iline], dtype=np.float32)
        ev3 = np.asarray(Rl3[iline], dtype=np.float32)

        spec1 = pcs1 @ ev1
        spec2 = pcs2 @ ev2
        spec3 = pcs3 @ ev3

        output[iline] = np.concatenate(
            [spec1, spec2, spec3],
            axis=1
        )

    return output

# ## derivative_planck(T,lam)

def derivative_planck(T,lam):
    # from Eulalie extracted from Filipe Aires PhD part 1 see IASI_nois part II
    # T en Kelvin
    # lam en longueur d'ondes (en metres!)
    h=6.62608e-34
    k=1.38066e-23
    c=2.99792e8

    #print(T,lam)
    P1 = (2*np.pi)*(c**3)*(h**2)*np.exp((c*h)/(float(k)*lam*T))
    P2 = k*(lam**6)*(T**2)*(np.exp((c*h)/(k*lam*T))-1)**2
    return P1/P2

# ## IASI_noise(T,lam)

def IASI_noise(T,lam):
    
    # from Eulalie to compute the noise extracted from Filipe Aires PhD
    
    wavenumber = [650, 770, 790, 980, 1000, 1070, 1080, 1200, 1210, 1650, 1660, 2090, 2100, 2420, 2430, 2500, 2600, 2700, 2760]
    # in wavelength
    #wavelength = 1.E4/wavenumber
    wl2freq=lambda t: 1.E4/t
    wavelength   = [wl2freq(t)*1E-6 for t in wavenumber]
    noise = [0.28, 0.28, 0.34,  0.34, 0.28, 0.28, 0.34, 0.34, 0.28, 0.28, 0.34, 0.5, 0.5, 0.5, 0.6, 0.77, 1.1, 1.58, 1.97]
    
  
    noise = noise[::-1]
    wavelength = wavelength[::-1]
    
    std80 = np.interp(wlnp, wavelength, noise)
    
    P1 = derivative_planck(T,lam)
    P2 = derivative_planck(280,lam)
    std80 = np.interp(lam, wavelength, noise)
    
    return (P1/P2)*std80

# ## def rad_to_bt(rad,wn)

def rad_to_bt(rad,wn):
# convert a radiance at a certain wn

#
    import math
    # in units: rad: mW/m2/sr/cm-1
    #            wavenumber: cm-1
    # out T in K
    c1 = 1.1910427e-5
    c2 = 1.4387752

    #ind      = np.where(rad > 0)
    #radiance = (np.array(rad))[ind]

    tb = (c2 * wn) / np.log( (c1*wn*wn*wn / rad)+1. )

    return(tb)

# ## rad_to_bt_iasi(rad,wn)

def rad_to_bt_iasi(rad,wn):
    import math
    # in units: rad: mW/m2/sr/cm-1
    #            wavenumber: cm-1
    # out T in K
    c1 = 1.1910427e-5
    c2 = 1.4387752

    ind      = np.where(rad > 0)
    wnum     = (np.array(wn))[ind]
    radiance = (np.array(rad))[ind]
   
    tb = (c2 * wnum) / np.log( (c1*wnum*wnum*wnum / radiance)+1. )
    
    return(tb,wnum)


# ## def plot_one_spect_1_pixels(wn,spec,ifov,mini,maxi,savefi='no',unit='rad',color='black',add_vert = 'no')

def plot_one_spect_1_pixels(wn,spec,ifov,mini,maxi,savefi='no',unit='rad',color='black',add_vert = 'no'):
# plot one sprectrum for all 4 pixels
    # here spec of size [wn,30,4]
# plot_one_spect_all_pixels(wn,spec*1000.*100.,ifov,0,120,savefi='no',unit='rad')
# for the radiance *1000*100 to get the correct unit mw/m2/sr/cm-1

    fontsize=22
    
    wl2freq=lambda t: 1.E4/t
    newlabel = [15,12,10,8,6,4]
    newpos   = [wl2freq(t) for t in newlabel]

    fig, ax1 = plt.subplots(figsize=(18,8))
    
    ax1.tick_params(axis="x", labelsize=fontsize-2,rotation=0)
    ax1.tick_params(axis="y", labelsize=fontsize-2,rotation=0)

    ax1.set_ylim(mini,maxi)
    ax1.set_xlim(wn[0],wn[len(wn)-1])
    
    if unit=='rad' or unit=='BT':
        ax1.plot(wn,spec[:,ifov,0],label='Pixel 1',color=color)
        #ax1.plot(wn,spec[:,ifov,1],label='Pixel 2',color='yellow')
        #ax1.plot(wn,spec[:,ifov,2],label='Pixel 3',color='red')
        #ax1.plot(wn,spec[:,ifov,3],label='Pixel 4',color='green')
        
    if unit=='noise':
        ax1.plot(wn,spec,label='Pixel 1',color=color)
    
    if unit=='rad':
        ax1.set_ylabel('Radiance (mW/m$^2$/sr/cm$^{-1}$)',fontsize=fontsize)
    if unit=='BT':
        ax1.set_ylabel('Brigthness Temperature (K)',fontsize=fontsize)
    ax1.set_xlabel('Wavenumber (cm$^{-1}$)',fontsize=fontsize,color='black') 
    if unit == 'noise':
        ax1.set_ylabel('Noise std (K)',fontsize=fontsize)
    #ax1.legend(loc='best',fontsize=18)
    
    if unit =='ratio':
        ax1.set_ylabel('Ratio R2 over R1',fontsize=fontsize)
        ax1.plot(wn,spec[:,ifov,0],label='Pixel 1',color=color)
        print('values ar:',spec[:,ifov,0])
        
    # add a top axes showing the wavelength
    ax2 = ax1.twiny()
    ax2.tick_params(axis="x", labelsize=fontsize-2,rotation=0)
    ax2.set_xticks(newpos)
    ax2.set_xticklabels(newlabel)
    ax2.set_xlabel('Wavelength '+u'(\u03bcm)',fontsize=fontsize,color='black')
    ax2.set_xlim(ax1.get_xlim())
    
    # add vertical lines
    if add_vert != 'no':
        vert=[650,770,980,1000,1070,1080,1150,1210,1650,2100,2150,2250,2350,2420,2700,2760]
        for v in vert:
            ax1.axvline(x=v, color='black', linestyle='--')
        alpha=0.2
        ax1.axvspan(650, 770, color='blue', alpha=alpha)  #  T
        ax1.axvspan(770, 980, color='orange', alpha=alpha) # cloud
        ax1.axvspan(1000, 1070, color='green', alpha=0.1)
        ax1.axvspan(1080, 1150, color='orange', alpha=alpha)  # cloud red
        ax1.axvspan(1210, 1650, color='purple', alpha=alpha)
        ax1.axvspan(2100, 2150, color='brown', alpha=alpha)
        ax1.axvspan(2150, 2250, color='blue', alpha=alpha)  # T and NO2 pink
        ax1.axvspan(2350, 2420, color='blue', alpha=alpha)  # T gray
        ax1.axvspan(2420, 2700, color='olive', alpha=alpha)
        ax1.axvspan(2700, 2760, color='orange', alpha=alpha)  # cloud and surf  cyan
        ax1.axvspan(1650, 2100, color='yellow', alpha=alpha)
    if savefi!='no':
        plt.savefig(savefi,dpi=100,bbox_inches='tight')

# ## def plot_one_spect_1_pixels_from_netcdf(wn,spec,mini,maxi,savefi='no',unit='rad',color='black',add_vert = 'no')

def plot_one_spect_1_pixels_from_netcdf(wn,spec,mini,maxi,savefi='no',unit='rad',color='black',add_vert = 'no'):
# plot one sprectrum for all 4 pixels
    # here spec of size [wn]
# plot_one_spect_all_pixels(wn,spec*1000.*100.,0,120,savefi='no',unit='rad')
# for the radiance *1000*100 to get the correct unit mw/m2/sr/cm-1

    print('spec=',spec)
    
    fontsize=22
    
    wl2freq  = lambda t: 1.E4/t
    newlabel = [15,12,10,8,6,4]
    newpos   = [wl2freq(t) for t in newlabel]

    fig, ax1 = plt.subplots(figsize=(18,8))
    
    ax1.tick_params(axis="x", labelsize=fontsize-2,rotation=0)
    ax1.tick_params(axis="y", labelsize=fontsize-2,rotation=0)

    ax1.set_ylim(mini,maxi)
    ax1.set_xlim(wn[0],wn[len(wn)-1])
    
    if unit=='rad' or unit=='BT':
        print('spec in if=',spec)
        ax1.plot(wn,spec[:],label='Pixel 1',color=color)
        
    if unit=='noise':
        ax1.plot(wn,spec,label='Pixel 1',color=color)
    
    if unit=='rad':
        ax1.set_ylabel('Radiance (mW/m$^2$/sr/cm$^{-1}$)',fontsize=fontsize)
    if unit=='BT':
        ax1.set_ylabel('Brigthness Temperature (K)',fontsize=fontsize)
        
    ax1.set_xlabel('Wavenumber (cm$^{-1}$)',fontsize=fontsize,color='black') 
    if unit == 'noise':
        ax1.set_ylabel('Noise std (K)',fontsize=fontsize)
    #ax1.legend(loc='best',fontsize=18)
    
    # add a top axes showing the wavelength
    ax2 = ax1.twiny()
    ax2.tick_params(axis="x", labelsize=fontsize-2,rotation=0)
    ax2.set_xticks(newpos)
    ax2.set_xticklabels(newlabel)
    ax2.set_xlabel('Wavelength '+u'(\u03bcm)',fontsize=fontsize,color='black')
    ax2.set_xlim(ax1.get_xlim())
    
    # add vertical lines
    if add_vert != 'no':
        vert=[650,770,980,1000,1070,1080,1150,1210,1650,2100,2150,2250,2350,2420,2700,2760]
        for v in vert:
            ax1.axvline(x=v, color='black', linestyle='--')
        alpha=0.2
        ax1.axvspan(650, 770, color='blue', alpha=alpha)  #  T
        ax1.axvspan(770, 980, color='orange', alpha=alpha) # cloud
        ax1.axvspan(1000, 1070, color='green', alpha=0.1)
        ax1.axvspan(1080, 1150, color='orange', alpha=alpha)  # cloud red
        ax1.axvspan(1210, 1650, color='purple', alpha=alpha)
        ax1.axvspan(2100, 2150, color='brown', alpha=alpha)
        ax1.axvspan(2150, 2250, color='blue', alpha=alpha)  # T and NO2 pink
        ax1.axvspan(2350, 2420, color='blue', alpha=alpha)  # T gray
        ax1.axvspan(2420, 2700, color='olive', alpha=alpha)
        ax1.axvspan(2700, 2760, color='orange', alpha=alpha)  # cloud and surf  cyan
        ax1.axvspan(1650, 2100, color='yellow', alpha=alpha)
    if savefi!='no':
        plt.savefig(savefi,dpi=100,bbox_inches='tight')

# ## def plot_one_spect_all_pixels(wn,spec,ifov,mini,maxi,savefi='no',unit='rad',add_vert='no')

def plot_one_spect_all_pixels(wn,spec,ifov,mini,maxi,savefi='no',unit='rad',add_vert='no'):
# plot one sprectrum for all 4 pixels
# spec shape: [wn,30,4]
# plot_one_spect_all_pixels(wn,spec*1000.*100.,ifov,0,120,savefi='no',unit='rad')
# for the radiance *1000*100 to get the correct unit mw/m2/sr/cm-1

    print(spec.shape,np.min(spec),np.max(spec))
    wl2freq=lambda t: 1.E4/t
    newlabel = [15,12,10,8,6,4]
    newpos   = [wl2freq(t) for t in newlabel]
    print('newpos=',newpos)
    print('wn=',wn)
    print('ifov=',ifov)
    print('mini=',mini)
    print('maxi=',maxi)

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

# ## def mmm_print(var,texte):

def mmm_print(var,texte):
    ind = np.where((var<10000000.)&(var>-1000000.))
    print(texte,np.min(var[ind]),np.max(var[ind]),np.mean(var[ind]),np.median(var[ind]),np.std(var[ind]))
    return

# ## def  plot_one_spect_all_pixels(wn,spec,ifov,mini,maxi,savefi='no',unit='rad')

def plot_one_spect_all_pixels(wn,spec,ifov,mini,maxi,savefi='no',unit='rad'):
# plot one sprectrum for all 4 pixels
# plot_one_spect_all_pixels(wn,spec*1000.*100.,ifov,0,120,savefi='no',unit='rad')
    # here spec [765,120,8461]
# for the radiance *1000*100 to get the correct unit mw/m2/sr/cm-1

    print(spec.shape)
    print('ifovin=',ifov)
    
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
    print(ifov)

# # PCS

#PC score inputs
sat   = 'M03'
#dirin = '/DSNNAS2/Repro/users/marieb/IASI_PCS/25_12_2019_M03/IASI_xxx_1C_M03_20191225031157Z_20191225045356Z/'
dirin = '/DSNNAS2/Repro/users/marieb/IASI_PCS/15_01_2022_M03/IASI*' #IASI_xxx_1C_M03_20220115085056Z_20220115103255Z/'
dirlist = glob.glob(dirin)
dirlist.sort()
print(dirlist)

dirnum = 1
file   = glob.glob(dirlist[dirnum]+'/merge*.nc')
file.sort()
print(file)

filepdu = glob.glob(dirlist[dirnum]+'/IASI*.nc')
filepdu.sort()


print(file)
print(len(filepdu),filepdu[0])


nc_fid = nc4.Dataset(file[0], "r")
nc_attrs = nc_fid.ncattrs()
print("NetCDF Global Attributes:")

for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
nc_fid.close()

# ## read variables from merged file

nc_fid = nc4.Dataset(file[0], "r")
lat    = nc_fid.variables['latitude'][:]
lon    = nc_fid.variables['longitude'][:]
lf     = nc_fid.variables['LandFraction'][:]
band   = 'Band1'
part   = 'P1'
pcsBand1P1 = nc_fid.groups['PCscores'].groups[band][part][:]
pcsBand1P3 = nc_fid.groups['PCscores'].groups[band]['P3'][:]
pcs = iasi.get_pc_scores_from_CDR(file[0])
print('----------------')

print(pcs.shape)

# ## read variable from a 3 min PDUs

# chose one pdu and read it
num = 33
nc_fidpdu = nc4.Dataset(filepdu[num], "r")
print(len(filepdu),num,filepdu[num])
latpdu  = nc_fidpdu.variables['latitude'][:]
lonpdu  = nc_fidpdu.variables['longitude'][:]
lfpdu   = nc_fidpdu.variables['LandFraction'][:]
band = 'Band1'
part = 'P1'
pcsBand1P1pdu = nc_fidpdu.groups['PCscores'].groups[band][part][:]
pcsBand1P3pdu = nc_fidpdu.groups['PCscores'].groups[band]['P3'][:]
pcspdu = iasi.get_pc_scores_from_CDR(filepdu[num])
print('----------------')
#nc_fidpdu.close()

print(latpdu)

print(latpdu.shape,lat.shape,pcsBand1P1pdu.shape,pcsBand1P3.shape,pcspdu.shape)

print(len(latpdu))
line   = 21
offset = 743

var    = pcs #pcsBand1P3  #lf #lon
varpdu = pcspdu #pcsBand1P3pdu # lf #lonpdu

dim3 = 'yes'
for i in range(23):
    if dim3 == 'no':
        print(i,var[line+offset,i],varpdu[line,i],var[line+offset,i]-varpdu[line,i])
    if dim3 == 'yes':
         ban = 119
         print(i,var[line+offset,i,ban],varpdu[line,i,ban],var[line+offset,i,ban]-varpdu[line,i,ban])

# ## plot a map

ban = 3
var_to_plot = pcs[:,:,ban]
print(np.min(var_to_plot),np.max(var_to_plot),var_to_plot.shape)

print(pcspdu.shape)
var_to_plotpdu = pcspdu[:,:,ban]
print(np.min(var_to_plotpdu),np.max(var_to_plotpdu))

date = 'xxx'
sat  = 'MetopC'
var_to_plot = lf #pcsR[:,:,0]
mini =  np.min(var_to_plot) #-9
maxi = np.max(var_to_plot) #11 #100 #1604 #100
print(mini,maxi)
title_lab = 'LandFraction'



ncol=0
colorp= 'RdYlBu_r' #'jet' #'viridis'
dirfig = '/DSNNAS2/Repro/users/marieb/'

rmap.create_map(dirfig+'Test_IASIL1C_LF.png', lat,lon,var_to_plot,
              projection='Robin', vminv=mini, vmaxv=maxi,
              c_colour=colorp, n_colors = ncol, title_lab = title_lab,
              title_str='',psize=0.5,colorbar_labsize=15,reverse_colpal='no')


print(lat.shape,np.min(lat),np.max(lat))
print(lat[20,75])
print(lon[20,75])

print(lat[0,:])

var = pcs[0,:,0]
print(var.shape,np.min(var),np.max(var))
#print(var[20,75,1])

var = lon
print(var.shape,np.min(var),np.max(var))
print(var[20,75])

# ## read EV files

ev_dir = '/DSNNAS2/Repro/users/marieb/IASI_PCS/EV/'
print(sat)
sat = 'M03'
if sat == 'M01':
    ev1_file = 'IASI_EV1_xx_M01_20120101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx'
    ev2_file = 'IASI_EV2_xx_M01_20120101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx'
    ev3_file = 'IASI_EV2_xx_M01_20120101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx'

if sat == 'M02':
    ev1_file = 'IASI_EV1_xx_M02_20060101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx'
    ev2_file = 'IASI_EV1_xx_M02_20060101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx'
    ev3_file = 'IASI_EV1_xx_M02_20060101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx'

if sat == 'M03':
    ev1_file = 'IASI_EV1_xx_M03_20180101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx'
    ev2_file = 'IASI_EV2_xx_M03_20180101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx'
    ev3_file = 'IASI_EV3_xx_M03_20180101000000Z_xxxxxxxxxxxxxxZ_20200131000201Z_xxxx_xxxxxxxxxx'

evf1 = ev_dir+ev1_file
evf2 = ev_dir+ev2_file
evf3 = ev_dir+ev3_file
print(evf2)
print(file[0])


iasi.inspect_EV(evf2)

iasi.inspect_EV(evf3)

evf1

R1 = iasi.load_reconstruction_operator(evf1)
R2 = iasi.load_reconstruction_operator(evf2)
R3 = iasi.load_reconstruction_operator(evf3)

print(R1.shape)
print(R2.shape)
print(R3.shape)


## the full_spec will be of size[line,&20,wn]
full_spec = reconstructGPT(pcs, R1, R2, R3, chunk_lines=10) 
print(full_spec.shape) # will be 765,120,8461

full_spec_30_4 = full_spec.reshape(765, 30, 4, 8461)

print(full_spec_30_4.shape)

print(np.min(full_spec_30_4),np.max(full_spec_30_4))

print(np.sum(full_spec_30_4 < 0))
print(np.sum(full_spec_30_4 > 0))

# # read IPO

IPOrad  = nc_fid.groups['IPO']['IPO_L1cRadiances'][:]

print(IPOrad.shape)
print(type(IPOrad))

IPOrad_30_4 = IPOrad.reshape(765, 30, 4, 8461)
print(IPOrad_30_4.shape)

# ## check valid pixels for a certain channel for 120 pixels

ichan = 100
# Extract the values for this channel
data = IPOrad[:, :, ichan]

# Find line and pixel indices where values are not NaN
lines, pixels = np.where(~np.isnan(data.filled(np.nan)))

# Print the results
for line, pixel in zip(lines, pixels):
    print(f"Line = {line}, Pixel = {pixel}, Lat={lat[line,pixel]}, Lon={lon[line,pixel]}, Rad= {IPOrad[line, pixel, ichan]} ")

print(lat2.shape)

# ## check valid pixels for a certain channel for 30 fov x 4 pixels

data = IPOrad_30_4[:, :, :, ichan]
rad = data.filled(np.nan)

valid = (
    ~np.ma.getmaskarray(data)
    & np.isfinite(rad)
    & np.isfinite(lat2)
    & np.isfinite(lon2)
)

lines, fovs, pixs = np.where(valid)

print("Number of valid pixels:", len(lines))

for line, fov, pix in zip(lines, fovs, pixs):
    print(
        f"Line={line}, FOV={fov}, Pix={pix}, "
        f"Lat={lat2[line, fov, pix]}, "
        f"Lon={lon2[line, fov, pix]}, "
        f"Rad={rad[line, fov, pix]}"
    )

# ##  do a map plot 

ichan = 100  # Your selected channel

# Extract radiance for this channel
data = IPOrad[:, :, ichan]

# Keep valid, unmasked and non-NaN radiances and valid coordinates
rad = data.filled(np.nan)

valid = (
    ~np.ma.getmaskarray(data)
    & np.isfinite(rad)
    & np.isfinite(lat)
    & np.isfinite(lon)
)

orbit_valid = np.isfinite(lat2) & np.isfinite(lon2)

# Get the valid positions
lines, pixels = np.where(valid)

# Extract coordinates and radiances
lats = lat[lines, pixels]
lons = lon[lines, pixels]
rads = rad[lines, pixels]

# Create the map
fig = plt.figure(figsize=(12, 7))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=135))

ax.set_global()
ax.coastlines(resolution='110m')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='white')

# Plot each pixels of the orbit + valid pixel, coloured by radiance

ax.scatter(
    lon2[orbit_valid],
    lat2[orbit_valid],
    s=2,
    color='lightgreen',
    alpha=0.35,
    transform=ccrs.PlateCarree(),
    label='Entire orbit',
    zorder=1
)

# 2. Overlay pixels with valid radiance for the selected channel
sc = ax.scatter(
    lons, lats,
    c=rads,
    s=12,
    cmap='viridis',
    transform=ccrs.PlateCarree(),
    zorder=2
)

# Add colour bar
cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.05)
cbar.set_label(f'Radiance, channel {ichan}')

ax.set_title(f'IPOrad radiance at channel {ichan}')
plt.tight_layout()
plt.show()

# 1. Plot the entire orbit in a pale colour
orbit_valid = np.isfinite(lat2) & np.isfinite(lon2)

ax.scatter(
    lon2[orbit_valid],
    lat2[orbit_valid],
    s=2,
    color='lightgrey',
    alpha=0.35,
    transform=ccrs.PlateCarree(central_longitude=135),
    label='Entire orbit',
    zorder=1
)

# 2. Overlay pixels with valid radiance for the selected channel
sc = ax.scatter(
    lons, lats,
    c=rads,
    s=12,
    cmap='viridis',
    transform=ccrs.PlateCarree(central_longitude=135),
    zorder=2
)

print(lat.shape)

# # test for a specific location

# lat, lon have shape (765, 120)

lat0 = -20.546  # tonga volcano for eruption 15 jan 2022
lon0 = -175.390

# longitude difference wrapped into [-180,180]
dlon = ((lon - lon0 + 180) % 360) - 180

# approximate distance in degrees
dist2 = (lat - lat0)**2 + (dlon*np.cos(np.deg2rad(lat0)))**2

iline, ipixel = np.unravel_index(np.argmin(dist2), dist2.shape)

print(f"Closest line  : {iline}")
print(f"Closest pixel : {ipixel}")
print(f"Latitude      : {lat[iline, ipixel]}")
print(f"Longitude     : {lon[iline, ipixel]}")

# ## test reshaping

lon2 = lon.reshape(765,30,4)
lat2 = lat.reshape(765,30,4)

print(lat.shape, lat2.shape)

# if lat/lon shape [765,30,4]

lat0 = -20.546
lon0 = -175.390

dlon = ((lon2 - lon0 + 180) % 360) - 180

dist2 = (lat2 - lat0)**2 + (dlon*np.cos(np.deg2rad(lat0)))**2

iline, ifov, ipix = np.unravel_index(
    np.argmin(dist2),
    dist2.shape
)

print(f"line  = {iline}")
print(f"fov   = {ifov}")
print(f"pixel = {ipix}")

print(f"lat = {lat2[iline,ifov,ipix]}")
print(f"lon = {lon2[iline,ifov,ipix]}")

# ### test map 120

colors=['red','blue','green','yellow']
colors2 = [colors[i % len(colors)] for i in range(120)]
print(colors2)

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

colors=['red','blue','green','yellow']
colors2 = [colors[i % len(colors)] for i in range(120)]

# Map
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.gridlines(draw_labels=True)

# Plot the first 4 points for each line

line = 0
for line in [700,701]: #range(765): #[0]: #[200,201,202]:
    for i in range(16):
        ax.plot(
            lon[line, i],
            lat[line, i],
            'o',
            color=colors2[i],
            markersize=10,
            transform=ccrs.PlateCarree()
        )

plt.title("First 4 latitude/longitude points for each line")
plt.show()

# ### test map 30,4
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

colors=['red','blue','green','yellow']
colors2 = [colors[i % len(colors)] for i in range(120)]

# Map
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.gridlines(draw_labels=True)

# Plot the first 4 points for each line


line = 468
ifov = 0
for ifov in [0,1,2,3]: #,1,2,3]: #[200,201,202]:
    for i in range(4):
        ax.plot(
            lon2[line, ifov,i],
            lat2[line, ifov,i],
            'o',
            color=colors2[i],
            markersize=10,
            transform=ccrs.PlateCarree()
        )

plt.title("First 4 latitude/longitude points for each line")
plt.show()

# ## plot some spectrum
line  = 234
ifov   = 29 #12
pixel = 3 #1


wn = np.linspace(644.75, 2759.75, 8461)
# in wavelength
wl = 1.E4/wn

plt.plot(wn,full_spec_30_4[line,ifov,0,:]*1000.*100.,color='orange')
plt.plot(wn,full_spec_30_4[line,ifov,1,:]*1000.*100.,color='green')
plt.plot(wn,full_spec_30_4[line,ifov,2,:]*1000.*100.,color='black')
plt.plot(wn,full_spec_30_4[line,ifov,3,:]*1000.*100.,color='blue')
#plt.plot(wn,full_spec_30_4[10,ifov+25,0,:]*1000.*100.,color='grey')


# ## with my colors

### plot all 4 pixels

print('ifov=',ifov,line)
mini = np.min(full_spec_30_4[line,:,:,:].transpose(2,0,1)[:,ifov,:]*1000*100*0.5)
maxi = np.max(full_spec_30_4[line,:,:,:].transpose(2,0,1)[:,ifov,:]*1000*100*0.5)

print(full_spec_30_4[line,:,:,:].shape,full_spec_30_4[line,:,:,:].transpose(2,0,1).shape)
print(mini,maxi)

plot_one_spect_all_pixels(wn,full_spec_30_4[line,:,:,:].transpose(2,0,1)*0.5*1000*100,ifov,mini,maxi,savefi='no',unit='rad')
print('ifov=',ifov,line)

print(full_spec_30_4[line,:,:,:].shape)

mini = 0
maxi = 200
plot_one_spect_1_pixels_from_netcdf(wn,full_spec_30_4[line,:,:,:].transpose(2,0,1)[:,0,0]*0.5*1000.*100.,mini,maxi,savefi='no',unit='rad',color='black',add_vert = 'yes')


ipix = 0
print(line,ifov,ipix)
print(full_spec_30_4[line,:,:,:].transpose(2,0,1)[:,ifov,ipix])

# transform in BT

print(wn)
print(line,ifov)
bt,wni = rad_to_bt_iasi(full_spec_30_4[line,:,:,:].transpose(2,0,1)[:,ifov,ipix]*0.5*100.*1000.,wn)

print(bt.shape,np.min(bt),np.max(bt),bt)

mini = 60
maxi = 300
plot_one_spect_1_pixels_from_netcdf(wn,bt,mini,maxi,savefi='no',unit='BT',color='black',add_vert = 'no')

mini = 0
maxi = 200
plot_one_spect_1_pixels_from_netcdf(wn,full_spec_30_4[line,:,:,:].transpose(2,0,1)[:,0,0]*100.*1000.*0.5,mini,maxi,savefi='no',unit='rad',color='black',add_vert = 'no')

# # Read local PCS

print(file[0])

hpcs = iasi.get_hpc_scores_from_CDR(file[0])
print('----------------')
print(hpcs.shape)

Rl1, Rl2, Rl3 = iasi.get_hpc_R_from_CDR(file[0])

print(Rl1.shape)
print(Rl2.shape)
print(Rl3.shape)

print(np.min(Rl1),np.max(Rl1))
print(np.min(Rl2),np.max(Rl2))
print(np.min(Rl3),np.max(Rl3))

print(np.min(R1),np.max(R1))
print(np.min(R2),np.max(R2))
print(np.min(R3),np.max(R3))

print(np.min(hpcs),np.max(hpcs))

#sl = reconstructLocalGPT(hpcs,Rl1,Rl2,Rl3)

sl2 = reconstruct_local_pcs(hpcs,Rl1,Rl2,Rl3)

print(sl2.shape,np.min(sl2),np.max(sl2))

sl_30_4 = sl2.reshape(765, 30, 4, 8461)

print(sl_30_4.shape)

print(np.sum(sl_30_4 < 0))
print(np.sum(sl_30_4 > 0))

print('ifov=',ifov,'line=',line)
mini = np.min(sl_30_4[line,:,:,:].transpose(2,0,1)[:,ifov,:]*1000*100*0.5)
maxi = np.max(sl_30_4[line,:,:,:].transpose(2,0,1)[:,ifov,:]*1000*100*0.5)

print(mini,maxi)

plot_one_spect_all_pixels(wn,sl_30_4[line,:,:,:].transpose(2,0,1)*0.5*1000*100,ifov,mini,maxi,savefi='no',unit='rad')



# # read IASI native

dirin = '/DSNNAS2/Repro/users/marieb/IASI_PCS_full/15_01_2022/'
#filepath = "IASI_xxx_1C_M01_20260901104155Z_20260901122059Z_N_O_20260901113242Z.nat"
filepath = glob.glob(dirin+'IASI_xxx*nat')
print(len(filepath),filepath)

#print(filepath[1])
numf = 1
print(filepath[numf])
native = NativeIASI1c(filepath[numf])

# example usage
native.wavenumbers, native.spectra

# example using xarray
ds = native.to_dataset()
ds

print(ds.spectrum.shape)

lonnat    = ds.longitude
latnat    = ds.latitude
spect_nat = ds.spectrum

print(spect_nat.shape)

# extract for plotting
print(line)
spectrum_nat = spect_nat[line,:,:,:].data
print(spectrum_nat.shape)

print(spectrum_nat.transpose(2,0,1).shape)

print('line = ',line,' ifov=',ifov)
mini = np.min(spectrum_nat.transpose(2,0,1)[:,ifov,:]*1000*100)
maxi = np.max(spectrum_nat.transpose(2,0,1)[:,ifov,:]*1000*100)

print(mini,maxi)

plot_one_spect_all_pixels(wn,spectrum_nat.transpose(2,0,1)*1000*100,ifov,mini,maxi,savefi='no',unit='rad')

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

colors=['red','blue','green','yellow']
colors2 = [colors[i % len(colors)] for i in range(120)]

# Map
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.gridlines(draw_labels=True)

# Plot the first 4 points for each line

for ifov in [0,1,2,3]: #,1,2,3]: #[200,201,202]:
    for i in range(4):
        ax.plot(
            lonnat[line, ifov,i],
            latnat[line, ifov,i],
            'o',
            color=colors2[i],
            markersize=10,
            transform=ccrs.PlateCarree()
        )

plt.title("First 4 latitude/longitude points for each line")
plt.show()

print(latnat.shape,lat2.shape)

# ## compute the difference between global PCS and native

difflat = latnat - lat2

print(spectrum_nat.shape)
print(spect_nat[line,:,:].shape)
print(full_spec_30_4[line,:,:,:].shape)
print(sl_30_4[line,:,:,:].shape)

print('line=',line)

print(np.mean(spect_nat[line,:,:,:].data))

print(np.mean(full_spec_30_4[line,:,:,:].data)*0.5)

diffspec = spect_nat[line,:,:,:].data - (full_spec_30_4[line,:,:,:]*0.5) # need to divide by 2!!!

print(diffspec.shape)

print(np.min(difflat.data),np.max(difflat.data))

print('diff spectrum:',np.min(diffspec.data),np.max(diffspec.data))
print('diff spec (changed units):',np.min(diffspec*1000*100.),np.max(diffspec)*1000*100)
print('diff spec (changed units):',np.min(diffspec[ifov,0,:]*1000*100.),np.max(diffspec[ifov,0,:])*1000*100)

mini = -0.9
maxi = 0.8
plot_one_spect_1_pixels_from_netcdf(wn,diffspec[ifov,0,:]*1000.*100.,mini,maxi,savefi='no',unit='rad',color='black',add_vert = 'no')

# global + local
print(sl_30_4.shape,full_spec_30_4.shape)

spec_glob_loc = sl_30_4[line,:,:,:]+full_spec_30_4[line,:,:,:]
print(spect_nat.shape)

diffspec = spect_nat[line,:,:,:] - (spec_glob_loc[:,:,:]*0.5) # need to divide by 2!!!

print(diffspec.shape)

print(spec_glob_loc.shape)

diffspecHG = spect_nat[line,:,:] - (sl_30_4[line,:,:,:]*0.5+full_spec_30_4[line,:,:,:]*0.5)

mini = -1.4
maxi = 1.4
print(line,ifov,ipix)
plot_one_spect_1_pixels_from_netcdf(wn,diffspec[ifov,ipix,:]*1000.*100.,mini,maxi,savefi='no',unit='rad',color='black',add_vert = 'no')

mini = -1.4
maxi = 1.4
plot_one_spect_1_pixels_from_netcdf(wn,diffspecHG.transpose(2,0,1)[:,0,0]*1000.*100.,mini,maxi,savefi='no',unit='rad',color='black',add_vert = 'no')

print(spect_nat.dims)
print(spect_nat.shape)

# # plot native, global, global+local

ipix = 3
line = 10
ifov = 10

line=236
ifov=27
ipix=0

print(line,ifov,ipix)
print(spect_nat[line,:,:].shape)
print(full_spec_30_4[line,:,:,:].shape)
print(sl_30_4[line,:,:,:].shape)

wl2freq=lambda t: 1.E4/t
newlabel = [15,12,10,8,6,4]
newpos   = [wl2freq(t) for t in newlabel]

mini = -10
maxi = 75
unit = 'rad'
zoom = 10. #(to see the diff nat - reconstructed)
print('newpos=',newpos)
print('wn=',wn)
print('ifov=',ifov)
print('ipix=',ipix)
print('iline=',line)
print('mini=',mini)
print('maxi=',maxi)

fig, ax1 = plt.subplots(figsize=(14,8))

ax1.tick_params(axis="x", labelsize=14,rotation=0)
ax1.tick_params(axis="y", labelsize=14,rotation=0)

ax1.set_ylim(mini,maxi)
ax1.set_xlim(wn[0],wn[len(wn)-1])


ax1.plot(wn, full_spec_30_4[line,ifov,ipix,:]*0.5*100.*1000., label='Reconstructed global', color='yellow',linewidth = 2)

ax1.plot(wn, full_spec_30_4[line,ifov,ipix,:]*0.5*100.*1000.+sl_30_4[line,ifov,ipix,:]*0.5*100.*1000.,label = 'Reconstructed glo+loc',color='green',linewidth = 0.5)

ax1.plot(wn, spect_nat[line,ifov,ipix,:].data*100.*1000., label='Natif', color='blue',linewidth = 0.3)
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

add_diff_axe = 'yes'
if add_diff_axe == 'yes':
    # add a right axes showing the zoom for the difference
    ax3 = ax1.twinx()
    
    ax3.plot(wn, (spect_nat[line,ifov,ipix,:].data - (full_spec_30_4[line,ifov,ipix,:]*0.5))*100.*1000.,
           label='Diff: native - reconstructed from global PCS',  color='black', linewidth = 1)

    ax3.plot(wn,(spect_nat[line,ifov,ipix,:].data - (full_spec_30_4[line,ifov,ipix,:]+sl_30_4[line,ifov,ipix,:])*0.5)*100.*1000.,
           label = 'Diff: native - reconstructed from glo+loc',color='pink',linewidth = 0.5) 
    
    #ax3.plot(wn, sl_30_4[line,ifov,ipix,:]*0.5*100.*1000., label='Reconstructed local',  color='red')

    if unit=='rad':
        ax3.set_ylabel('Difference (Rad in mW/m$^2$/sr/cm$^{-1}$',fontsize=18)
    if unit=='BT':
        ax3.set_ylabel('Difference (BT in K)',fontsize=18)
        
    ax3.set_xlabel('Wavenumber (cm$^{-1}$)',fontsize=18,color='black') 
    ax3.tick_params(axis="x", labelsize=14,rotation=0)
    ax3.tick_params(axis="y", labelsize=14,rotation=0)
    ax3.set_ylim(mini/zoom,maxi/zoom)
    ax3.set_xlim(wn[0],wn[len(wn)-1])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    
    ax1.legend(
        lines1 + lines3,
        labels1 + labels3,
        loc='best',
        fontsize=14
    )
    #ax3.legend(loc='best',fontsize=18)
    
savefi = 'no'
if savefi!='no':
    plt.savefig(savefi,dpi=100,bbox_inches='tight')


