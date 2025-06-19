import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import emcee
from astropy.time import Time 
from astropy.coordinates import EarthLocation, get_body, solar_system_ephemeris
import astropy.units as u
from getdist import plots, MCSamples
from matplotlib.ticker import MaxNLocator

# --- Data Loading ---
borexino_data = np.loadtxt("data/Eccentricity_Data_Fig4.txt")
time_all = borexino_data[:, 0]  
ref_time = Time('2011-12-11T00:00:00', format='isot', scale='utc')
start_date = Time('2019-10-01T00:00:00', format='isot', scale='utc')
end_date   = Time('2021-10-01T00:00:00', format='isot', scale='utc')
start_days = (start_date - ref_time).value
end_days = (end_date - ref_time).value
index = np.where((time_all >= start_days) & (time_all <= end_days))[0]

time = time_all[index]
data = borexino_data[index, 1]
sigma = borexino_data[index, 2]

# --- Distance Calculations ---
def coordinates_to_decdeg(deg, min, sec):
    return deg + min/60 + sec/3600

latitude = coordinates_to_decdeg(42, 27, 10)
longitude = coordinates_to_decdeg(13, 34, 30)
loc = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg, height=936.45*u.m)

obs_time = ref_time + time*u.day
with solar_system_ephemeris.set('de430'):
    sun_coords = get_body('sun', obs_time, loc)
    jup_coords = get_body('jupiter', obs_time, loc)

dsun = sun_coords.distance.to(u.au)
djup = jup_coords.distance.to(u.au)

# --- Bayesian Model ---
def loglikelihood(theta, data, sigma, dsun, djup):
    Rsun_1AU, Rjup_5AU, Rb = theta
    model = Rsun_1AU * (1.0/dsun.value)**2 + Rjup_5AU * (5.0/djup.value)**2 + Rb
    return -0.5 * np.sum(((model - data) / sigma) ** 2)

def logprior(theta):
    Rsun_1AU, Rjup_5AU, Rb = theta
    if not (0 <= Rjup_5AU <= 4 and 0 <= Rb <= 50):
        return -np.inf
    mu_sun = 25.0
    sig_sun = 2.0
    lp_sun = -0.5 * ((Rsun_1AU - mu_sun) / sig_sun) ** 2
    return lp_sun

def logposterior(theta, data, sigma, dsun, djup):
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood(theta, data, sigma, dsun, djup)

# --- MCMC Sampling ---
Nens = 100
Nburnin = 1000
Nsamples = 5000
ndims = 3

Rsun_1AU_init = np.random.normal(25, 2, Nens)
Rjup_5AU_init = np.random.uniform(0, 4, Nens)
Rb_init = np.random.uniform(0, 50, Nens)
inisamples = np.array([Rsun_1AU_init, Rjup_5AU_init, Rb_init]).T

argslist = (data, sigma, dsun, djup)
sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)
sampler.run_mcmc(inisamples, Nsamples + Nburnin, progress=True)
samples_emcee = sampler.get_chain(flat=True, discard=Nburnin)

# --- Plotting and Analysis ---
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 22

names = ['Rsun', 'Rjup', 'Rb']
labels = [
r'\frac{\mathcal{R}_{\mathrm{sun}}}{(1\,\mathrm{AU})^2}\,(\mathrm{cpd}/100\mathrm{t})',
r'\frac{\mathcal{R}_{\mathrm{jup}}}{(5\,\mathrm{AU})^2}\,(\mathrm{cpd}/100\mathrm{t})',
r'\mathcal{R}_B\,(\mathrm{cpd}/100\mathrm{t})'
]

gdsamples = MCSamples(samples=samples_emcee, names=names, labels=labels)

g = plots.get_subplot_plotter(width_inch=3)
g.settings.axes_labelsize = 24
g.settings.axes_fontsize = 18
g.settings.legend_fontsize = 16
g.settings.alpha_filled_add = 0.4
g.settings.figure_legend_frame = False
g.settings.title_limit_fontsize = 13.5

g.plot_2d(gdsamples, 'Rjup', 'Rb', filled=True, contour_colors=['#1f77b4'])

fig = plt.gcf()
ax = plt.gca()

ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontsize(6)
    tick.set_fontweight('bold')

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.xaxis.label.set_fontsize(10)
ax.xaxis.label.set_fontweight('bold')
ax.yaxis.label.set_fontsize(10)
ax.yaxis.label.set_fontweight('bold')
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
plt.show()

# --- Print Results ---
def median_and_cred(samples):
    median = np.median(samples)
    low = np.percentile(samples, 16)
    high = np.percentile(samples, 84)
    return median, high - median, median - low

rsun_1au, rsun_hi, rsun_lo = median_and_cred(samples_emcee[:, 0])
rjup_5au, rjup_hi, rjup_lo = median_and_cred(samples_emcee[:, 1])
rb, rb_hi, rb_lo = median_and_cred(samples_emcee[:, 2])

print(f"Rsun/(1 AU)^2: {rsun_1au:.2f} +{rsun_hi:.2f} -{rsun_lo:.2f} (cpd/100t)")
print(f"Rjup/(5 AU)^2: {rjup_5au:.2f} +{rjup_hi:.2f} -{rjup_lo:.2f} (cpd/100t)")
print(f"RB: {rb:.2f} +{rb_hi:.2f} -{rb_lo:.2f} (cpd/100t)")


