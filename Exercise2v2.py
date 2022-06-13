# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:43:29 2022

@author: melbe
"""
#%% LOAD PACKAGES

import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmocean.cm as cmo
import cftime

#------------------------------------------
def weighted_avg(data,weights): #data must be #time, lat, lon
    return data.weighted(weights).mean(dim=('lat','lon'))
#    return np.mean(np.average(data, axis=1, weights=weights),axis=1) #average weighted over latitude, and normal over longitude
#------------------------------------------

#%% LOAD VARIABLES

# ERA5 - 1950-2021
t2m = xr.open_dataset('ERA5-singleleveldata.nc').t2m ### si10,t2m,msl,tp
s10 = xr.open_dataset('ERA5-singleleveldata.nc').si10
msl = xr.open_dataset('ERA5-singleleveldata.nc').msl
tpr = xr.open_dataset('ERA5-singleleveldata.nc').tp # mm/day
geo = xr.open_dataset('ERA5-pressureleveldata.nc').z ### z,u,v @ 300 hPa
ugp = xr.open_dataset('ERA5-pressureleveldata.nc').u
vgp = xr.open_dataset('ERA5-pressureleveldata.nc').v
u10 = xr.open_dataset('ERA5-singleleveldata_v2.nc').u10
v10 = xr.open_dataset('ERA5-singleleveldata_v2.nc').v10

# Historical - 1950-2021
# =============================================================================
# ta = xr.open_dataset('1950_2021_monmean.nc').ta     # air temperature
# =============================================================================
ua = xr.open_dataset('1950_2021_monmean.nc').ua     # eastward wind
va = xr.open_dataset('1950_2021_monmean.nc').va     # northward wind
# =============================================================================
# ts = xr.open_dataset('1950_2021_monmean.nc').ts     # surface temperature
# snd = xr.open_dataset('1950_2021_monmean.nc').snd   # surface snow thickness
# =============================================================================
prl = xr.open_dataset('1950_2021_monmean.nc').prl   # lwe of LS precipitation
prc = xr.open_dataset('1950_2021_monmean.nc').prc   # convective precipitation rate
prsn = xr.open_dataset('1950_2021_monmean.nc').prsn # lwe of snowfall amount
psl = xr.open_dataset('1950_2021_monmean.nc').psl   # air pressure at sea level
zg = xr.open_dataset('1950_2021_monmean.nc').zg     # geopotential height
# =============================================================================
# hur = xr.open_dataset('1950_2021_monmean.nc').hur   # relative humidity
# clt = xr.open_dataset('1950_2021_monmean.nc').clt   # cloud area fraction
# =============================================================================
tas = xr.open_dataset('1950_2021_monmean.nc').tas   # air temperature 2m
# =============================================================================
# sa = xr.open_dataset('1950_2021_monmean.nc')['as']  # surface albedo
# rss = xr.open_dataset('1950_2021_monmean.nc').rss   # surface net shortwave flux
# rls = xr.open_dataset('1950_2021_monmean.nc').rls   # surface net longwave flux
# rlut = xr.open_dataset('1950_2021_monmean.nc').rlut # toa net longwave flux
# tauu = xr.open_dataset('1950_2021_monmean.nc').tauu # surface eastward stress
# tauv = xr.open_dataset('1950_2021_monmean.nc').tauv # surface northward stress
# evap = xr.open_dataset('1950_2021_monmean.nc').evap # lwe of water evaporation
# =============================================================================
spd = xr.open_dataset('1950_2021_monmean.nc').spd   # wind_speed

#%% TEMPERATURE - PREPARATION

# Annual mean
t2m_am = t2m.mean(dim='time')
tas_am = tas.mean(dim='time')

# January / July
month_idxs = t2m.groupby('time.month').groups
jan_idxs = month_idxs[1]
jul_idxs = month_idxs[7]
t2m_jan = t2m.isel(time=jan_idxs).mean(dim='time')
t2m_jul = t2m.isel(time=jul_idxs).mean(dim='time')

month_idxsd = tas.groupby('time.month').groups
jan_idxsd = month_idxsd[1]
jul_idxsd = month_idxsd[7]
tas_jan = tas.isel(time=jan_idxsd).mean(dim='time')
tas_jul = tas.isel(time=jul_idxsd).mean(dim='time')

# Regrid for difference
t2m_rg = t2m_am.interp(latitude=tas_am.lat, longitude=tas_am.lon)
td_am = t2m_rg - tas_am

t2m_rg = t2m_jan.interp(latitude=tas_jan.lat, longitude=tas_jan.lon)
td_jan = t2m_rg - tas_jan

t2m_rg = t2m_jul.interp(latitude=tas_jul.lat, longitude=tas_jul.lon)
td_jul = t2m_rg - tas_jul

#%% TEMPERATURE - PLOTS

plots = ['ERA5 Year', 'ERA5 January', 'ERA5 July',
         'Model Year', 'Model January', 'Model July']
ts = [t2m_am, t2m_jan, t2m_jul, tas_am, tas_jan, tas_jul]

vmin, vmax = -70, 50 
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

fig, axs = plt.subplots(nrows=2,ncols=3,
                        subplot_kw={'projection': ccrs.Robinson()},
                        figsize=(17,8))
axs=axs.flatten()
for i, plot in enumerate(plots):
        data = ts[i]
        if i < 3:
            lons, lats = np.meshgrid(data.longitude, data.latitude)
        else:
            lons, lats = np.meshgrid(data.lon, data.lat)
        pcm = axs[i].pcolormesh(lons, lats, data-273.15,
                          transform = ccrs.PlateCarree(),
                          cmap=cmo.balance,vmin=vmin, vmax=vmax, norm=norm)
        axs[i].set_title(plot)
        axs[i].coastlines()
        axs[i].set_global()
fig.subplots_adjust(bottom=0.25, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.02)
cbar_ax = fig.add_axes([0.2, 0.2, 0.6, 0.02])
cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(label=r'2m temperature [°C]',fontsize=13)
plt.suptitle('1950-2021 Mean Surface (2m) Temperature',fontsize=18)
plt.show()

# Difference
plots = ['ERA5-model Year', 'ERA5-model January', 'ERA5-model July']
ts = [td_am, td_jan, td_jul]

vmin, vmax = -15, 35
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

fig, axs = plt.subplots(nrows=1,ncols=3,
                        subplot_kw={'projection': ccrs.Robinson()},
                        figsize=(17,6))
#plt.suptitle('ERA5-model 1950-2021 2m Temperature (°C)')
axs=axs.flatten()
for i, plot in enumerate(plots):
        data = ts[i]
        lons, lats = np.meshgrid(data.lon, data.lat)
        pcm = axs[i].pcolormesh(lons, lats, data,
                          transform = ccrs.PlateCarree(),
                          cmap=cmo.balance,vmin=vmin, vmax=vmax, norm=norm)
        axs[i].set_title(plot)
        axs[i].coastlines()
        axs[i].set_global()
fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.02)
cbar_ax = fig.add_axes([0.2, 0.3, 0.6, 0.02])
cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(label=r'2m temperature anomaly [°C]',fontsize=13)
plt.show()

#%% PRECIPITATION - PREPARATION

# Sum components
pr = 24*60*60*1000*(prc + prl + prsn) # total precipitation in mm/day

# Annual mean
tpr_am = tpr.mean(dim='time') * 1000.
pr_am = pr.mean(dim='time')

# January / July
month_idxs = tpr.groupby('time.month').groups
jan_idxs = month_idxs[1]
jul_idxs = month_idxs[7]
tpr_jan = tpr.isel(time=jan_idxs).mean(dim='time') * 1000.
tpr_jul = tpr.isel(time=jul_idxs).mean(dim='time') * 1000.

month_idxs = pr.groupby('time.month').groups
jan_idxs = month_idxs[1]
jul_idxs = month_idxs[7]
pr_jan = pr.isel(time=jan_idxs).mean(dim='time')
pr_jul = pr.isel(time=jul_idxs).mean(dim='time')

# Regrid for difference
tpr_rg = tpr_am.interp(latitude=pr_am.lat, longitude=pr_am.lon)
td_am = tpr_rg - pr_am
tpr_rg = tpr_jan.interp(latitude=pr_jan.lat, longitude=pr_jan.lon)
td_jan = tpr_rg - pr_jan
tpr_rg = tpr_jul.interp(latitude=pr_jul.lat, longitude=pr_jul.lon)
td_jul = tpr_rg - pr_jul

#%% PRECIPITATION - PLOTS

plots = ['ERA5 Year', 'ERA5 January', 'ERA5 July',
         'Model Year', 'Model January', 'Model July']
ts = [tpr_am, tpr_jan, tpr_jul, pr_am, pr_jan, pr_jul]

fig, axs = plt.subplots(nrows=2,ncols=3,
                        subplot_kw={'projection': ccrs.Robinson()},
                        figsize=(17,8))
axs=axs.flatten()
for i, plot in enumerate(plots):
        data = ts[i]
        if i < 3:
            lons, lats = np.meshgrid(data.longitude, data.latitude)
        else:
            lons, lats = np.meshgrid(data.lon, data.lat)
        pcm = axs[i].pcolormesh(lons, lats, data,
                          transform = ccrs.PlateCarree(),
                          cmap=cmo.rain)
        axs[i].set_title(plot)
        axs[i].coastlines()
        axs[i].set_global()
fig.subplots_adjust(bottom=0.25, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.02)
cbar_ax = fig.add_axes([0.2, 0.2, 0.6, 0.02])
cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(label=r'total precipitation [mm/day]',fontsize=13)
plt.suptitle('1950-2021 Mean Precipitation',fontsize=18)
plt.show()


# Difference
plots = ['ERA5-model Year', 'ERA5-model January', 'ERA5-model July']
ts = [td_am, td_jan, td_jul]

fig, axs = plt.subplots(nrows=1,ncols=3,
                        subplot_kw={'projection': ccrs.Robinson()},
                        figsize=(17,6))
#plt.suptitle('ERA5-model 1950-2021 2m Temperature (°C)')
axs=axs.flatten()
for i, plot in enumerate(plots):
        data = ts[i]
        lons, lats = np.meshgrid(data.lon, data.lat)
        pcm = axs[i].pcolormesh(lons, lats, data,
                          transform = ccrs.PlateCarree(),
                          cmap=cmo.rain)
        axs[i].set_title(plot)
        axs[i].coastlines()
        axs[i].set_global()
fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.02)
cbar_ax = fig.add_axes([0.2, 0.3, 0.6, 0.02])
cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(label=r'total precipitation anomaly [mm/day]',fontsize=13)
plt.show()

#%% PRESSURE AND WIND - PREPARATION

# Annual mean
msl_am = msl.mean(dim='time') / 100.
psl_am = psl.mean(dim='time')
u10_am = u10.mean(dim='time')
v10_am = v10.mean(dim='time')
ua_am = ua.mean(dim='time')
va_am = va.mean(dim='time')

# January / July
month_idxs = msl.groupby('time.month').groups
jan_idxs = month_idxs[1]
jul_idxs = month_idxs[7]
msl_jan = msl.isel(time=jan_idxs).mean(dim='time') / 100.
msl_jul = msl.isel(time=jul_idxs).mean(dim='time') / 100.

month_idxs = psl.groupby('time.month').groups
jan_idxs = month_idxs[1]
jul_idxs = month_idxs[7]
psl_jan = psl.isel(time=jan_idxs).mean(dim='time')
psl_jul = psl.isel(time=jul_idxs).mean(dim='time')

month_idxs = u10.groupby('time.month').groups
jan_idxs = month_idxs[1]
jul_idxs = month_idxs[7]
u10_jan = u10.isel(time=jan_idxs).mean(dim='time')
u10_jul = u10.isel(time=jul_idxs).mean(dim='time')

month_idxs = v10.groupby('time.month').groups
jan_idxs = month_idxs[1]
jul_idxs = month_idxs[7]
v10_jan = v10.isel(time=jan_idxs).mean(dim='time')
v10_jul = v10.isel(time=jul_idxs).mean(dim='time')

month_idxs = ua.groupby('time.month').groups
jan_idxs = month_idxs[1]
jul_idxs = month_idxs[7]
ua_jan = ua.isel(time=jan_idxs).mean(dim='time')
ua_jul = ua.isel(time=jul_idxs).mean(dim='time')

month_idxs = va.groupby('time.month').groups
jan_idxs = month_idxs[1]
jul_idxs = month_idxs[7]
va_jan = va.isel(time=jan_idxs).mean(dim='time')
va_jul = va.isel(time=jul_idxs).mean(dim='time')

# Regrid for difference
msl_rg = msl_am.interp(latitude=psl_am.lat, longitude=psl_am.lon)
pd_am = msl_rg - psl_am
msl_rg = msl_jan.interp(latitude=psl_jan.lat, longitude=psl_jan.lon)
pd_jan = msl_rg - psl_jan
msl_rg = msl_jul.interp(latitude=psl_jul.lat, longitude=psl_jul.lon)
pd_jul = msl_rg - psl_jul

u10_rg = u10_am.interp(latitude=ua_am.lat, longitude=ua_am.lon)
ud_am = u10_rg - ua_am
u10_rg = u10_jan.interp(latitude=ua_jan.lat, longitude=ua_jan.lon)
ud_jan = u10_rg - ua_jan
u10_rg = u10_jul.interp(latitude=ua_jul.lat, longitude=ua_jul.lon)
ud_jul = u10_rg - ua_jul

v10_rg = v10_am.interp(latitude=va_am.lat, longitude=va_am.lon)
vd_am = v10_rg - va_am
v10_rg = v10_jan.interp(latitude=va_jan.lat, longitude=va_jan.lon)
vd_jan = v10_rg - va_jan
v10_rg = v10_jul.interp(latitude=va_jul.lat, longitude=va_jul.lon)
vd_jul = v10_rg - va_jul

#%%



#%% PRESSURE AND WIND - PLOTS

plots = ['ERA5 Year', 'ERA5 January', 'ERA5 July',
         'Model Year', 'Model January', 'Model July']
ts = [msl_am, msl_jan, msl_jul, psl_am, psl_jan, psl_jul]
us = [u10_am, u10_jan, u10_jul, ua_am, ua_jan, ua_jul]
vs = [v10_am, v10_jan, v10_jul, va_am, va_jan, va_jul]

fig, axs = plt.subplots(nrows=2,ncols=3,
                        subplot_kw={'projection': ccrs.Robinson()},
                        figsize=(17,8))
axs=axs.flatten()
for i, plot in enumerate(plots):
        data = ts[i]
        quivu = us[i]
        quivv = vs[i]
        if i < 3:
            lons, lats = np.meshgrid(data.longitude, data.latitude)
        else:
            lons, lats = np.meshgrid(data.lon, data.lat)
        pcm = axs[i].pcolormesh(lons, lats, data,
                          transform = ccrs.PlateCarree(),
                          cmap=cmo.speed)
        axs[i].quiver(lons,lats,quivu,quivv)
        axs[i].set_title(plot)
        axs[i].coastlines()
        axs[i].set_global()
fig.subplots_adjust(bottom=0.25, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.02)
cbar_ax = fig.add_axes([0.2, 0.2, 0.6, 0.02])
cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(label=r'mean sea level pressure [hPa]',fontsize=13)
plt.suptitle('1950-2021 Mean Sea Level Pressure',fontsize=18)
plt.show()


# =============================================================================
# # Difference
# plots = ['ERA5-model Year', 'ERA5-model January', 'ERA5-model July']
# ts = [pd_am, pd_jan, pd_jul]
# 
# vmin, vmax = -20, 45
# norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
# 
# fig, axs = plt.subplots(nrows=1,ncols=3,
#                         subplot_kw={'projection': ccrs.Robinson()},
#                         figsize=(17,6))
# #plt.suptitle('ERA5-model 1950-2021 2m Temperature (°C)')
# axs=axs.flatten()
# for i, plot in enumerate(plots):
#         data = ts[i]
#         lons, lats = np.meshgrid(data.lon, data.lat)
#         pcm = axs[i].pcolormesh(lons, lats, data,
#                           transform = ccrs.PlateCarree(),
#                           cmap=cmo.delta,vmin=vmin,vmax=vmax,norm=norm)
#         axs[i].set_title(plot)
#         axs[i].coastlines()
#         axs[i].set_global()
# fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
#                     wspace=0.02, hspace=0.02)
# cbar_ax = fig.add_axes([0.2, 0.3, 0.6, 0.02])
# cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
# cbar.set_label(label=r'mean sea level pressure anomaly [hPa]',fontsize=13)
# plt.show()
# =============================================================================
