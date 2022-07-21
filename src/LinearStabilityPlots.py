from matplotlib import gridspec
from matplotlib import font_manager as fm
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                    )

logging.info('Importing standard python libraries')
from pathlib import Path
from os import cpu_count

logging.info('Importing third party python libraries')
import numpy as np
from scipy.sparse.linalg import eigs
import scipy.sparse as sps
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import xarray as xr
import cmocean.cm as cmo
from joblib import Parallel, delayed

logging.info('Setting paths')
base_path = Path('../').absolute().resolve()
figure_path = base_path / 'figures'
figure_path.mkdir(exist_ok=True)
data_path = base_path / 'data'
raw_path = data_path / 'raw'
processed_path = data_path / 'processed'

logging.info('Setting plotting defaults')
# fonts
fpath = Path('/System/Library/Fonts/Supplemental/PTSans.ttc')
font_prop = fm.FontProperties(fname=fpath)
plt.rcParams['font.family'] = font_prop.get_family()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]

# font size
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
plt.rc('text', usetex=False)
plt.rcParams['axes.titlesize'] = 12

# output
dpi = 600

logging.info('Setting plots to plot')
dispersion = False
streamfunction = False
sigma_latitude = False
dwbc = True
n_jobs = max(cpu_count(), 64)

logging.info('Setting flow parameters')
# General parameters
hydrostatic = False
A_r = 4e-4
N2 = 2.5e-5
f = 1.01e-5
lambda_arr = np.logspace(0, 2.5, num=500)

# Bickley jet
Lx = 400e3
dx = 2e3
nx = int(Lx / dx)
x = np.arange(0, Lx, dx)

V_0 = 1
x_mid = 40e3
delta_b = 30e3

# Rankine vortex
R = 400e3
dr = 2e3
nr = int(R / dr)
r = np.arange(dr, R + dr, dr)

R0 = 100e3
V0 = -1


def V_phi(r, R0=R0, V0=V0):
    """ Velocity of Rankine vortex
    """
    if r < R0:
        V_phi = r * V0 / R0
    else:
        V_phi = R0 * V0 / r
    return V_phi


def dV_phi_dr(r, R0=100e3, V0=1):
    """ Derivative of V_phi w.r.t r
    """
    if r < R0:
        dV_phi_dr = V0 / R0
    else:
        dV_phi_dr = - R0 * V0 / r / r
    return dV_phi_dr


def dV_dx(x, V_0=V_0, x_mid=x_mid, delta_b=delta_b):
    """ returns the Bickley jet relative vorticity at x in s^{-1}
    """
    rel_vort = - 2 * V_0 / delta_b * np.tanh((x - x_mid) / delta_b) / np.square(np.cosh((x - x_mid) / delta_b))
    return rel_vort


# Create a matrix representation of the curvature operator
diagonals = np.array([1, -2, 1]) / np.square(dx)
d2_dx2 = sps.diags(diagonals, offsets=[-1, 0, 1], shape=(nx, nx))

# Create a matrix representation of the d/dr operators
diagonals = np.array([1, -2, 1]) / np.square(dr)
d2_dr2 = sps.diags(diagonals, offsets=[-1, 0, 1], shape=(nr, nr))

diagonals = np.array([-1, 1]) / 2 / dr
d_dr = sps.diags(diagonals, offsets=[-1, 1], shape=(nr, nr))
rinv = np.ones((nr, nr)) / r[..., np.newaxis]
rinv_d_dr = d_dr.multiply(rinv)

rinv2_arr = np.square(rinv)
rinv2 = sps.eye(nr).multiply(rinv2_arr)

V_phi_arr = np.array([V_phi(r, R0=R0, V0=V0) for r in r])
dV_phi_dr_arr = np.array([dV_phi_dr(r, R0=R0, V0=V0) for r in r])


# Create functions to calculate eigenvalues and eigenfunctions


def calculate_centrifugal_eigenvalue(lambda_val, f, return_eigenvectors=False):
    """ Calculates the eigenvalue of the LSA equation (i.e. \hat{\omega}^2)

    Arguments:
    lambda_val (float)--> Value of lambda at which to solve the problem

    Returns:
    eigan_val (tuple) --> first element is the eigenvalue, second the
        eigenfunction
    """
    # Create an array representation of the f * zeta operator
    fprime_zeta_arr = (f + 2 * V_phi_arr / r) * \
        (f + dV_phi_dr_arr + V_phi_arr / r)
    fprime_zeta = sps.dia_matrix(fprime_zeta_arr * np.identity(nr))

    m = 2 * np.pi / lambda_val
    m2 = np.square(m)
    eigen_operator = -N2 / m2 * (d2_dr2 + rinv_d_dr - rinv2) + fprime_zeta

    if hydrostatic:
        eigen_val = eigs(eigen_operator, k=1, which='SR',
                         return_eigenvectors=return_eigenvectors)
    else:
        generalised_M = sps.eye(
            nr) - (d2_dr2 + rinv_d_dr - rinv2) / m2  # Identiy - d2/dx2
        eigen_val = eigs(eigen_operator, k=1, M=generalised_M,
                         which='SR', return_eigenvectors=return_eigenvectors)

    return eigen_val


def calc_centrifugal_sigma(lambda_val, f, viscosity=A_r):
    m = 2 * np.pi / lambda_val
    m2 = np.square(m)
    omega_hat = np.lib.scimath.sqrt(calculate_centrifugal_eigenvalue(lambda_val, f)[0])
    omega = omega_hat - 1j * viscosity * m2
    sigma = omega.imag
    return sigma


def max_centrifugal_sigma(f, lambda_arr=lambda_arr, viscosity=A_r):
    sigma_max = 0
    for lambda_val in lambda_arr:
        sigma = calc_centrifugal_sigma(lambda_val, f, viscosity=viscosity)
        if sigma > sigma_max:
            sigma_max = sigma
    return sigma_max


def calculate_inertial_eigenvalue(lambda_val, f, return_eigenvectors=False):
    """ Calculates the eigenvalue of the LSA equation (i.e. \hat{\omega}^2)

    Arguments:
    lambda_val (float)--> Value of lambda at which to solve the problem

    Returns:
    eigan_val (tuple) --> first element is the eigenvalue, second the eigenfunction
    """
    # Create an array representation of the f * zeta operator
    f_zeta_arr = f * (f + dV_dx(x, V_0, x_mid, delta_b))
    f_zeta = sps.dia_matrix(f_zeta_arr * np.identity(nx))
    m = 2 * np.pi / lambda_val
    m2 = np.square(m)
    eigen_operator = -N2 / m2 * d2_dx2 + f_zeta

    if hydrostatic:
        eigen_val = eigs(eigen_operator, k=1, which='SR',
                         return_eigenvectors=return_eigenvectors)
    else:
        generalised_M = sps.eye(nx) - d2_dx2 / m2  # Identiy - d2/dx2
        eigen_val = eigs(eigen_operator, k=1, M=generalised_M,
                         which='SR', return_eigenvectors=return_eigenvectors)

    return eigen_val


def calc_inertial_sigma(lambda_val, f, viscosity=A_r):
    m = 2 * np.pi / lambda_val
    m2 = np.square(m)
    omega_hat = np.lib.scimath.sqrt(
        calculate_inertial_eigenvalue(lambda_val, f))
    omega = omega_hat - 1j * viscosity * m2
    sigma = omega.imag
    return sigma


def max_inertial_sigma(f, lambda_arr=lambda_arr, viscosity=A_r):
    sigma_max = 0
    for lambda_val in lambda_arr:
        sigma = calc_inertial_sigma(lambda_val, f, viscosity=viscosity)[0]
        if sigma > sigma_max:
            sigma_max = sigma
    return sigma_max


def produce_dispersion_dataset(instability):
    eigen_value_arr = np.empty_like(lambda_arr, dtype='complex')
    psi_arr = np.empty((lambda_arr.size, r.size, 1), dtype='complex')

    if instability == 'inertial':
        calculate_eigenvalue = calculate_inertial_eigenvalue
        attrs = {'V_0': V_0, 'x_mid': x_mid, 'delta_b': delta_b, 'N2': N2,
                 'f': f, 'hydrostatic': str(hydrostatic), 'nx': nx, 'dx': dx,
                 'Lx': Lx}
    elif instability == 'centrifugal':
        attrs = {'V0': V0, 'R0': R0, 'N2': N2, 'f': f,
                 'hydrostatic': str(hydrostatic), 'nr': nr, 'dr': dr, 'R': R}
        calculate_eigenvalue = calculate_centrifugal_eigenvalue
    else:
        raise NotImplementedError('instability must be either "inertial" or "centrifugal"')

    for ii in range(len(lambda_arr)):
        eigen_value_arr[ii], psi_arr[ii] = calculate_eigenvalue(lambda_arr[ii], f, return_eigenvectors=True)

    psi_arr = psi_arr.squeeze().real  # Get rid of the extra dimension of the eigenfunction array. Get rid of the complex part (should be zero anyway)

    viscosity_arr = np.logspace(np.log10(5e-7), -2, num=500)
    
    # Analysis is made easier by converting the eigenfunctions and eigenvalues into xarray objects.
    if instability == 'inertial':
        ds = xr.Dataset(coords={'lambda': lambda_arr, 'viscosity': viscosity_arr, 'lon': x}, attrs=attrs)
        across_coord = 'lon'
    elif instability == 'centrifugal':
        ds = xr.Dataset(coords={'lambda': lambda_arr, 'viscosity': viscosity_arr, 'radial_coord': r}, attrs=attrs)
        across_coord = 'radial_coord'

    ds['eigen_value'] = ds['lambda'].copy(data=eigen_value_arr)
    ds['psi_unnormalised'] = (('lambda', across_coord), psi_arr)

    ds['m'] = 2 * np.pi / ds['lambda']
    ds['m2'] = np.square(ds['m'])

    # Calculate the growth rate from the eigenvalues
    ds['omega_hat'] = ds['lambda'].copy(data=np.lib.scimath.sqrt(ds['eigen_value']))
    ds['omega'] = ds['omega_hat'] - 1j * ds['viscosity'] * ds['m2']
    ds['sigma'] = ds['omega'].where(ds['omega'].imag >= 0).imag.transpose('viscosity', 'lambda')  # Selects sigma >= 0 at the same time
    ds['sigma_normalised'] = ds['sigma'] / f

    # Calculate the wavelength of the least stable vertical mode as a function of the viscosity
    ds['lambda_unstable'] = ds['sigma'].idxmax(dim='lambda')
    ds['sigma_unstable'] = ds['sigma'].max('lambda')
    ds['time_scale_days'] = 1 / (ds['sigma_unstable'] * 24 * 60 * 60)

    # Apply one of two normalisations to the streamfunction to enable comparisons
    psi_max_idx = abs(ds['psi_unnormalised']).argmax(across_coord)
    ds['psi_sign_corrected'] = np.sign(ds['psi_unnormalised'].isel({across_coord: psi_max_idx})) * ds['psi_unnormalised']  # Correct the sign
    ds['psi_norm_area'] = ds['psi_sign_corrected'] / ds['psi_sign_corrected'].sum(across_coord)  # Gives the normalised streamfunction
    ds['psi_norm_max'] = ds['psi_sign_corrected'] / ds['psi_sign_corrected'].max(across_coord)
    return ds


def produce_sigma_lat_dataset(A_r):
    logging.info('Calculating inertial_sigma')
    inertial_sigma = Parallel(n_jobs=n_jobs)(delayed(max_inertial_sigma)(f, viscosity=A_r) for f in f_arr)

    logging.info('Calculating centrifugal_sigma')
    centrifugal_sigma = Parallel(n_jobs=n_jobs)(delayed(max_centrifugal_sigma)(f, viscosity=A_r) for f in f_arr)

    logging.info('Saving sigma_lat dataset')
    attrs = {'V_0': V_0, 'x_mid': x_mid, 'delta_b': delta_b, 'N2': N2,
             'f': f, 'hydrostatic': str(hydrostatic), 'nx': nx, 'dx': dx,
             'Lx': Lx, 'nr': nr, 'dr': dr, 'R': R, 'A_r': A_r, 'V0': V0,
             'R0': R0}

    ds_sigma_lat = xr.Dataset(data_vars={'f': ('latitude', f_arr),
                                         'centrifugal_sigma': ('latitude', centrifugal_sigma),
                                         'inertial_sigma': ('latitucde', inertial_sigma)},
                              coords={'latitude': latitude},
                              attrs=attrs)
    return ds_sigma_lat


if dispersion or streamfunction:
    logging.info('Calculating dispersion relation for inertial instability')
    ds_disp_inertial = produce_dispersion_dataset('inertial')
    ds_disp_inertial.to_zarr(processed_path / 'inertial_dispersion', mode='w')

    logging.info('Calculating dispersion relation for centrifugal instability')
    ds_disp_centrifugal = produce_dispersion_dataset('centrifugal')
    ds_disp_centrifugal.to_zarr(processed_path / 'centrifugal_dispersion',
                                mode='w')

if dispersion:
    logging.info('Making dispersion relation plots')
    clim = 1.2
    xmax = 450

    fig, axs = plt.subplot_mosaic([['ax0', 'ax1'], ['ax2', 'ax2']],
                                  figsize=[6, 4], sharey=True,
                                  gridspec_kw={'height_ratios': [12, 1]})

    ax0 = axs['ax0']
    cax = ax0.contourf(ds_disp_inertial['lambda'], ds_disp_inertial['viscosity'],
                       ds_disp_inertial['sigma_normalised'],
                       levels=np.linspace(0, clim, 13), cmap=cmo.amp)

    ax0.plot(ds_disp_inertial['lambda_unstable'],
             ds_disp_inertial['viscosity'],
             '-.k', lw=1, label='$\\lambda^*(A_r)$')

    ax0.set_xscale('log')
    ax0.set_xlim(1, xmax)

    ax0.set_xlabel('$\\lambda$ (m)')
    ax0.set_ylabel('$A_r$ (m$^2\\,$s$^{-1}$)')
    ax0.set_title('($\\,$a$\\,$)', loc='left')
    ax0.set_title('Inertial instability')

    ax1 = axs['ax1']
    cax = ax1.contourf(ds_disp_centrifugal['lambda'],
                       ds_disp_centrifugal['viscosity'],
                       ds_disp_centrifugal['sigma_normalised'],
                       levels=np.linspace(0, clim, 13), cmap=cmo.amp)

    cb = plt.colorbar(cax, cax=axs['ax2'], orientation='horizontal')
    cb.set_label('$\\sigma / f$', rotation=0, labelpad=15)
    cb.set_ticks(np.arange(0, clim + 0.1, 0.2))

    ax1.plot(ds_disp_centrifugal['lambda_unstable'],
             ds_disp_centrifugal['viscosity'],
             '-.k', lw=1, label='$\\lambda^*(A_r)$')

    ax1.legend(loc='upper left')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1, xmax)
    ax1.set_ylim(5e-7, 1.5e-2)
    ax1.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

    ax1.set_xlabel('$\\lambda$ (m)')
    ax1.set_title('($\\,$b$\\,$)', loc='left')
    ax1.set_title('Centrifugal instability')

    fig.tight_layout()
    fig.savefig(figure_path / 'vertical_scale_selection.pdf')


if streamfunction:
    logging.info('Making streamfunction plots')
    # Define z and calculate psi(x, z)
    ds_disp_centrifugal['z'] = np.linspace(-400, -0, num=100)
    lambda_unstable = ds_disp_centrifugal['lambda_unstable'].sel({'viscosity': A_r}, method='nearest')
    psi2D_centrifugal = np.cos(2 * np.pi / lambda_unstable * ds_disp_centrifugal['z']) * ds_disp_centrifugal['psi_norm_max'].sel({'lambda': lambda_unstable}, method='nearest')

    ds_disp_inertial['z'] = np.linspace(-400, -0, num=100)
    lambda_unstable = ds_disp_inertial['lambda_unstable'].sel({'viscosity': A_r}, method='nearest')
    psi2D_inertial = np.cos(2 * np.pi / lambda_unstable * ds_disp_inertial['z']) * ds_disp_inertial['psi_norm_max'].sel({'lambda': lambda_unstable}, method='nearest')

    # Plotting
    clim = 1.01

    fig, axs = plt.subplot_mosaic([['ax0', 'ax1'], ['ax2', 'ax2']],
                                  figsize=[6, 4], sharey=False,
                                  gridspec_kw={'height_ratios': [12, 1]})

    axs['ax0'].contourf(ds_disp_inertial['lon'] * 1e-3, -ds_disp_inertial['z'],
                        psi2D_inertial, cmap=cmo.balance,
                        levels=np.linspace(-clim, clim, 10),
                        vmin=-clim, vmax=clim)

    cax = axs['ax1'].contourf(ds_disp_centrifugal['radial_coord'] * 1e-3,
                              -ds_disp_centrifugal['z'], psi2D_centrifugal,
                              cmap=cmo.balance, levels=np.linspace(-clim, clim, 10),
                              vmin=-clim, vmax=clim)

    cb = plt.colorbar(cax, cax=axs['ax2'], ticks=np.linspace(-1, 1, 5),
                      orientation='horizontal')

    cb.set_label('$\\psi$', rotation=0)

    axs['ax1'].invert_yaxis()
    axs['ax0'].set_xlabel('$x$ (km)')
    axs['ax1'].set_xlabel('$r$ (km)')
    axs['ax0'].set_ylabel('Depth (m)')
    axs['ax0'].set_title('($\\,$a$\\,$)', loc='left')
    axs['ax0'].set_title('Inertial instability')
    axs['ax1'].set_title('($\\,$b$\\,$)', loc='left')
    axs['ax1'].set_title('Centrifugal instability')
    axs['ax1'].set_xticks(np.linspace(0, 400, 9))

    axs['ax1'].set_xlim((0, 400))
    axs['ax0'].set_xticks([0, 100, 200, 300])
    axs['ax1'].set_xticks([0, 100, 200, 300])
    axs['ax0'].set_yticks([0, 100, 200])
    axs['ax1'].set_yticks([0, 100, 200])
    axs['ax0'].set_ylim(200, 0)
    axs['ax1'].set_ylim(200, 0)
    axs['ax1'].set_yticklabels([])

    axs['ax0'].set_xlim(0, 300)
    axs['ax1'].set_xlim(0, 300)

    rankine_vorticity = np.array([f + dV_phi_dr(r, R0, V0) + V_phi(r, R0, V0) / r for r in r])
    bickley_vorticity = np.array([f + dV_dx(x, V_0, x_mid, delta_b) for x in x])

    ax1b = axs['ax1'].twinx()
    ax1b.plot(r * 1e-3, rankine_vorticity / f, 'k', lw=1.5)
    ax1b.set_ylabel('$\\zeta / f$')
    ax1b.axhline(0, c='k', ls=':')
    ax1b.set_ylim(-2, 3.75)

    ax0b = axs['ax0'].twinx()
    ax0b.plot(x * 1e-3, bickley_vorticity / f, 'k', lw=1.5)
    ax0b.axhline(0, c='k', ls=':')
    ax0b.set_ylim(-2, 3.75)
    ax0b.set_yticklabels([])

    fig.tight_layout()
    fig.savefig(figure_path / 'overturning_structure.pdf')


if sigma_latitude:
    logging.info('Making sigma_latitude plots')
    latitude = np.linspace(0.01, 90, 200)
    tomega = 2 * np.pi / 24 / 60 / 60
    f_arr = tomega * np.sin(np.radians(latitude))

    sigma_lat_4em4_path = processed_path / 'sigma_lat_4em4'
    if not sigma_lat_4em4_path.exists():
        ds_sigma_lat_4em4 = produce_sigma_lat_dataset(4e-4)
        ds_sigma_lat_4em4.to_zarr(sigma_lat_4em4_path, mode='w')
    else:
        ds_sigma_lat_4em4 = xr.open_zarr(sigma_lat_4em4_path)

    sigma_lat_1em6_path = processed_path / 'sigma_lat_1em6'
    if not sigma_lat_1em6_path.exists():
        ds_sigma_lat_1em6 = produce_sigma_lat_dataset(1e-6)
        ds_sigma_lat_1em6.to_zarr(sigma_lat_1em6_path, mode='w')
    else:
        ds_sigma_lat_1em6 = xr.open_zarr(sigma_lat_1em6_path)

    logging.info('Doing the sigma_lat plotting')
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(6, 4))

    axs[0].plot(ds_sigma_lat_1em6['inertial_sigma'],
                ds_sigma_lat_1em6['latitude'],
                label='Inertial', c='k', ls='-')

    axs[0].plot(ds_sigma_lat_1em6['centrifugal_sigma'],
                ds_sigma_lat_1em6['latitude'],
                label='Centrifugal', c='k', ls='-.')

    axs[0].set_ylim(0, 25)
    axs[0].set_xlim(0, 1.4e-5)

    axs[0].set_xlabel('$\\sigma$ (s$^{-1}$)')
    axs[1].set_xlabel('$\\sigma$ (s$^{-1}$)')
    axs[0].set_ylabel('Latitude')

    formatter0 = EngFormatter(unit='$^\circ$N', sep='', usetex=True)
    axs[0].yaxis.set_major_formatter(formatter0)

    axs[0].xaxis.major.formatter._useMathText = True
    axs[1].xaxis.major.formatter._useMathText = True

    axs[1].plot(ds_sigma_lat_4em4['inertial_sigma'],
                ds_sigma_lat_4em4['latitude'],
                label='Inertial', c='k', ls='-')

    axs[1].plot(ds_sigma_lat_4em4['centrifugal_sigma'],
                ds_sigma_lat_4em4['latitude'],
                label='Centrifugal', c='k', ls='-.')

    axs[1].set_xlim(0, 1.4e-5)

    axs[1].legend()

    axs[0].set_title('$A_r = 1 \\times 10^{-6}$ m$^{2}\\,$s$^{-1}$')
    axs[0].set_title('($\\,$a$\\,$)', loc='left')

    axs[1].set_title('$A_r = 4 \\times 10^{-4}$ m$^{2}\\,$s$^{-1}$')
    axs[1].set_title('($\\,$b$\\,$)', loc='left')

    fig.tight_layout()
    fig.savefig(figure_path / 'sigma_lat.pdf')


if dwbc:
    logging.info('Making dwbc plots')
    Lx = 400e3
    dx = 2e3
    nx = int(Lx / dx)
    x = np.arange(0, Lx, dx)

    V_0 = 0.2
    x_mid = 60e3
    delta_b = 30e3

    f = 2.3e-11 * 0.75 * 111e3
    N2 = 1e-6

    xmax = 450
    clim = 1.5

    ds_disp_dwbc = produce_dispersion_dataset('inertial')

    fig = plt.figure(figsize=[6, 4])

    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 1],
                           height_ratios=[12, 1])

    ax0 = fig.add_subplot(gs[0, 0])

    cax0 = ax0.contourf(ds_disp_dwbc['lambda'], ds_disp_dwbc['viscosity'],
                        ds_disp_dwbc['sigma_normalised'],
                        levels=np.linspace(0, clim, 13), cmap=cmo.amp)

    ax0.plot(ds_disp_dwbc['lambda_unstable'],
             ds_disp_dwbc['viscosity'],
             '-.k', lw=1, label='$\\lambda^*(A_r)$')

    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_xlim(1, xmax)

    ax0.set_xlabel('$\\lambda$ (m)')
    ax0.set_ylabel('$A_r$ (m$^2\\,$s$^{-1}$)')
    ax0.set_title('($\\,$a$\\,$)', loc='left')
    ax0.set_title('Dispersion')

    cbax0 = fig.add_subplot(gs[1, 0])
    cb0 = fig.colorbar(cax0, cax=cbax0, orientation='horizontal')
    cb0.set_label('$\\sigma / f$', rotation=0, labelpad=15)
    cb0.set_ticks(np.arange(0, clim + 0.1, 0.25))
    ax0.legend(loc='upper left')

    ax0.set_ylim(5e-7, 1.5e-2)
    ax0.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

    ax1 = fig.add_subplot(gs[:, 1])
    ax1.set_title('($\\,$b$\\,$)', loc='left')
    ax1.set_title('Growth rate and latitude')

    ax1.set_xlabel('$\\sigma$ (s$^{-1}$)')
    ax1.set_ylabel('Latitude')
    formatter0 = EngFormatter(unit='$^\circ$N', sep='', usetex=True)
    ax1.yaxis.set_major_formatter(formatter0)

    # Need to calculate sigma_lat for the plots
    dwbc_sigma_lat_ds = raw_path / 'dwbc_sigma_lat'
    if not dwbc_sigma_lat_ds.exists():

        latitude = np.linspace(0.01, 50, 120)
        tomega = 2 * np.pi / 24 / 60 / 60
        f_arr = tomega * np.sin(np.radians(latitude))

        dwbc_sigma_1em6 = Parallel(n_jobs=n_jobs)(delayed(max_inertial_sigma)(f, viscosity=1e-6) for f in f_arr)
        dwbc_sigma_4em4 = Parallel(n_jobs=n_jobs)(delayed(max_inertial_sigma)(f, viscosity=4e-4) for f in f_arr)
        dwbc_sigma = np.stack([dwbc_sigma_1em6, dwbc_sigma_4em4], axis=0)

        attrs = {'V_0': V_0, 'x_mid': x_mid, 'delta_b': delta_b, 'N2': N2,
                 'hydrostatic': str(hydrostatic), 'nx': nx, 'dx': dx,
                 'Lx': Lx}

        ds_sigma_lat = xr.Dataset(data_vars={'f': ('latitude', f_arr),
                                             'dwbc_sigma': (('A_r', 'latitude'), dwbc_sigma)},
                                  coords={'latitude': latitude,
                                          'A_r': [1e-6, 4e-4]},
                                  attrs=attrs)

        ds_sigma_lat.to_zarr(dwbc_sigma_lat_ds)

    else:
        ds_sigma_lat = xr.open_zarr(dwbc_sigma_lat_ds)

    # Now plot sigma_lat

    fig.tight_layout()
    fig.savefig(figure_path / 'dwbc_lsa.pdf')
