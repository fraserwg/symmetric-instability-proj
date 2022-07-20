import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                    )

logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

logging.info('Setting paths')
base_path = Path('../').absolute().resolve()
figure_path = base_path / 'figures'
figure_path.mkdir(exist_ok=True)

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

logging.info('Defining velocity profile functions')


def bickley_jet(x, V0, xmid, deltab):
    return V0 * (1 - np.square(np.tanh((x - xmid) / deltab)))


def rankine_vortex(r, V0, R0):
    V = np.empty_like(r)
    V[r < R0] = r[r < R0] * V0 / R0
    V[r >= R0] = R0 * V0 / r[r >= R0]
    return V


logging.info('Setting Bickley jet parameters')
Lx = 400e3
dx = 2e3
x = np.arange(0, Lx + dx, dx)

V0 = 1
xmid = 40e3
deltab = 30e3

logging.info('Setting Rankine vortex parameters')
R = 400e3  # Domain width in m
dr = 2e3  # Grid spacing in m
r = np.arange(0, R + dr, dr)

R0 = 40e3
V0_rankine = -1


logging.info('Plotting the velocity profiles')
fig, axs = plt.subplots(2, 1, figsize=[6, 4], sharex=True,)

axs[0].plot(x * 1e-3, bickley_jet(x, V0, xmid, deltab), c='k')
axs[1].plot(r * 1e-3, rankine_vortex(r, V0_rankine, R0), c='k')

axs[0].set_ylim(-0.1, 1.1)
axs[1].set_ylim(-1.1, 0.1)
axs[0].set_xlim(0, Lx * 1e-3)

axs[0].set_title('Bickley jet')
axs[0].set_title('(\\,$a$\\,$)', loc='left')

axs[1].set_title('Rankine vortex')
axs[1].set_title('($\\,$b$\\,$)', loc='left')

axs[0].set_xlabel('$x$ (km)')
axs[1].set_xlabel('$r$ (km)')

axs[0].set_ylabel('$V(x)$ (m$\\,$s$^{-1}$)')
axs[1].set_ylabel('$V_{\\phi}(r)$ (m$\\,$s$^{-1}$)')

fig.tight_layout()

figure_name = figure_path / 'lsa_velocity.pdf'
logging.info('Saving figure to {}'.format(figure_name))
fig.savefig(figure_name)

logging.info('Execution complete')
