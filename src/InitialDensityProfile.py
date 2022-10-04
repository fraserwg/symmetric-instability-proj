import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                    )

logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import font_manager as fm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean.cm as cmo
import xarray as xr

logging.info('Importing custom python libraries')
import okapy
from okapy.phi import generate_phi_z

logging.info('Setting paths')
base_path = Path('../').absolute().resolve()
figure_path = base_path / 'figures'
figure_path.mkdir(exist_ok=True)
raw_path = base_path / 'data/raw'

logging.info('Setting plotting defaults')
# fonts
fpath = Path('/System/Library/Fonts/Supplemental/PTSans.ttc')
font_prop = fm.FontProperties(fname=fpath)
plt.rcParams['font.family'] = font_prop.get_family()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]

# font size
matplotlib.use('pgf')
plt.rc('xtick', labelsize='8')
plt.rc('ytick', labelsize='8')
plt.rc('text', usetex=False)
plt.rcParams['axes.titlesize'] = 10
plt.rcParams["text.latex.preamble"] = "\\usepackage{euler} \\usepackage{paratype}  \\usepackage{mathfont} \\mathfont[digits]{PT Sans}"
plt.rcParams["pgf.preamble"] = plt.rcParams["text.latex.preamble"]
plt.rc('text', usetex=False)


# Output
dpi = 600

argo_path = raw_path / 'argo-profiles.1901681.nc'
logging.info('Attempting to open {}'.format(argo_path))
if not argo_path.exists():
    logging.info('{} does not exist'.format(argo_path))

    prefix = Path(okapy.__file__).parent
    trajectory_path = (prefix / '../data/argo-profiles-1901681.nc').resolve()

    logging.info('Attempting to open {}'.format(trajectory_path))
    assert trajectory_path.exists()
    ds = xr.open_dataset(trajectory_path)

    logging.info('Saving the file to {}'.format(argo_path))
    ds.to_netcdf(argo_path)

else:
    ds = xr.open_dataset(argo_path)

gebco_path = raw_path / 'GEBCO-bathymetry-data/gebco_2021_n30.0_s-30.0_w-85.0_e-10.0.nc'
logging.info('Attempting to open {}'.format(gebco_path))
ds_bathymetry = xr.open_dataset(gebco_path)
ds_bathymetry = ds_bathymetry.coarsen(lon=5, lat=5, boundary='trim').mean()

logging.info('Producing density profile with okapy')
z = np.linspace(0, 1500, 300)
rho = generate_phi_z()(z)

logging.info('Doing the plotting')
fig = plt.figure(figsize=(6, 3.25))

gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1, 1],
                       height_ratios=[14, 1]
                       )

ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())

# Plot the bathymetry
cax_bathy = ax0.pcolormesh(ds_bathymetry['lon'],
                           ds_bathymetry['lat'],
                           -ds_bathymetry['elevation'],
                           shading='nearest',
                           rasterized=True,
                           cmap=cmo.deep,
                           vmin=0
                           )

# Add some land
ax0.add_feature(cfeature.NaturalEarthFeature('physical',
                                             'land',
                                             '110m',
                                             edgecolor='face',
                                             facecolor='grey'
                                             ))

# Axes limits, labels and features
ax0.axhline(0, c='k', ls='--')

ax0.set_ylim(-12, 30)
ax0.set_xlim(-85, -25)

ax0.set_xticks(np.arange(-85, -24, 15), crs=ccrs.PlateCarree())
ax0.set_yticks(np.arange(-10, 31, 10), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax0.xaxis.set_major_formatter(lon_formatter)
ax0.yaxis.set_major_formatter(lat_formatter)

ax0.set_xlabel('Longitude')
ax0.set_ylabel('Latitude')
ax0.set_title('The Tropical Atlantic')
ax0.set_title('($\\,$a$\\,$)', loc='left')

# Colorbars
cbax1 = fig.add_subplot(gs[1, 0])
cb1 = plt.colorbar(cax_bathy, cax=cbax1,
                   label='Depth (m)', orientation='horizontal')
ax0.plot(ds.LONGITUDE, ds.LATITUDE, c='m', label='Float trajectory')
ax0.scatter(ds.LONGITUDE[0], ds.LATITUDE[0],
            c='m', label='Starting point', marker='x')

# Density plots
ax2 = fig.add_subplot(gs[:, 1])

ax2.set_title('Neutral density')
ax2.set_title('($\\,$b$\\,$)', loc='left')
ax2.plot(rho - 1000, z, label='Derived density profile', c='k')
ax2.invert_yaxis()


ax2.set_xlabel('$\\gamma^n$ (kg$\\,$m$^{-3}$)', loc='center', usetex=True)
ax2.set_ylabel('Depth (m)')
ax2.set_xlim(23, 28)
ax2.axvline(23.45, c="k", ls="-.", label="$\\gamma^n = 23.45$")
ax2.axvline(26.5, c="k", ls=":", label="$\\gamma^n = 26.5$")

plt.rc('text', usetex=True)
ax2.legend(loc="lower left")
plt.rc('text', usetex=False)


ax2.set_xticks(range(23, 29))

ax2.set_ylim(z[-1], 0)

fig.tight_layout(h_pad=3)

figure_file = figure_path / 'argo_trajectory.pdf'
fig.savefig(figure_file, dpi=dpi)
logging.info('Figure saved to {}'.format(figure_file))
