
import numpy as np
import matplotlib.pyplot as plt
import emcee
from astropy.time import Time 
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.coordinates import get_body
from astropy.coordinates import solar_system_ephemeris
from getdist import plots, MCSamples
import matplotlib.pyplot as plt
import matplotlib as mpl

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
with solar_system_ephemeris.set('de442s'):
    sun_coords = get_body('sun', obs_time, loc)
    mar_coords = get_body('mars', obs_time, loc)

dsun = sun_coords.distance.to(u.au)
dmar = mar_coords.distance.to(u.au)

# --- MCMC Sampling ---
def loglikelihood(theta, data, sigma, dsun, dmar):
    Rsun_1AU, Rmar_5AU, Rb = theta
    model = Rsun_1AU * (1.0/dsun.value)**2 + Rmar_5AU * (2.0/dmar.value)**2 + Rb
    return -0.5 * np.sum(((model - data) / sigma) ** 2)

def logprior(theta):
    Rsun_1AU, Rmar_5AU, Rb = theta
    
    if not (-13 <= Rmar_5AU <= 13 and 0 <= Rb <= 50 and 0 <= Rsun_1AU <= 50):
        return -np.inf
    return 0

def logposterior(theta, data, sigma, dsun, dmar):
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood(theta, data, sigma, dsun, dmar)

Nens = 100
Nburnin = 1000
Nsamples = 5000
ndims = 3

Rsun_1AU_init = np.random.uniform(0, 50, Nens)
Rmar_5AU_init = np.random.uniform(-13, 13, Nens)
Rb_init = np.random.uniform(0, 50, Nens)
inisamples = np.array([Rsun_1AU_init, Rmar_5AU_init, Rb_init]).T

argslist = (data, sigma, dsun, dmar)
sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)

sampler.run_mcmc(inisamples, Nsamples + Nburnin, progress=True)
samples_emcee = sampler.get_chain(flat=True, discard=Nburnin)

# --- Plotting and Analysis ---
mpl.rcParams['font.family'] = 'serif'  # Use serif fonts (e.g., Times New Roman)
mpl.rcParams['mathtext.fontset'] = 'stix'  # Use STIX math fonts
mpl.rcParams['mathtext.rm'] = 'serif'  # Match regular math font to serif

names = ['Rsun', 'Rmar', 'Rb']
labels = [
    r'\frac{\mathcal{R}_{\mathrm{sun}}}{(1\,\mathrm{AU})^2}\,(\mathrm{cpd}/100\mathrm{t})',
    r'\frac{\mathcal{R}_{\mathrm{mar}}}{(2\,\mathrm{AU})^2}\,(\mathrm{cpd}/100\mathrm{t})',
    r'\mathcal{R}_B\,(\mathrm{cpd}/100\mathrm{t})'
]

gdsamples = MCSamples(samples=samples_emcee, names=names, labels=labels)

g = plots.get_subplot_plotter()
g.settings.figure_legend_frame = False
g.settings.alpha_filled_add = 0.4
g.settings.title_limit_labels = False
g.settings.axis_marker_lw = 0.0
g.settings.linewidth_contour = 1.5
g.settings.num_plot_contours = 2

blue = '#1f77b4' 
g.triangle_plot([gdsamples], ['Rsun','Rmar', 'Rb'], filled=True, contour_colors=[blue],line_args=[{'ls': '-', 'color': blue}],markers=None)
fig = plt.gcf()
ax = plt.gca()

for ax in fig.axes:
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(25)
        tick.set_fontweight('bold')
        ax.xaxis.label.set_fontsize(35)
        ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontsize(35)
    ax.yaxis.label.set_fontweight('bold')
    ax.xaxis.labelpad = 40
    ax.yaxis.labelpad = 40
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
plt.show()

# --- Print Results ---
def median_and_cred(samples):
    median = np.median(samples)
    low = np.percentile(samples, 16)
    high = np.percentile(samples, 84)
    return median, high - median, median - low

rsun_1au, rsun_hi, rsun_lo = median_and_cred(samples_emcee[:, 0])
rmar_5au, rmar_hi, rmar_lo = median_and_cred(samples_emcee[:, 1])
rb, rb_hi, rb_lo = median_and_cred(samples_emcee[:, 2])

print(f"Rsun/(1 AU)^2: {rsun_1au:.2f} +{rsun_hi:.2f} -{rsun_lo:.2f} (cpd/100t)")
print(f"Rmar/(5 AU)^2: {rmar_5au:.2f} +{rmar_hi:.2f} -{rmar_lo:.2f} (cpd/100t)")
print(f"RB: {rb:.2f} +{rb_hi:.2f} -{rb_lo:.2f} (cpd/100t)")
