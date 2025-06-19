
import numpy as np
import matplotlib.pyplot as plt
import emcee
from astropy.time import Time 
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.coordinates import get_body
from astropy.coordinates import solar_system_ephemeris

# --- Data Loading ---
borexino_data = np.loadtxt("data/Eccentricity_Data_Fig4.txt")
time_all = borexino_data[:, 0]  
ref_time = Time('2011-12-11T00:00:00', format='isot', scale='utc')
start_date = Time('2019-10-01T00:00:00', format='isot', scale='utc')
end_date   = Time('2021-10-01T00:00:00', format='isot', scale='utc')
start_days = (start_date - ref_time).value
end_days = (end_date - ref_time).value
index1 = np.where((time_all >= start_days) & (time_all <= end_days))[0]

time1 = time_all[index1]
Rate1 = borexino_data[index1, 1]
sigma1 = borexino_data[index1, 2]
Trend1 = borexino_data[index1, 3]
data1 = Rate1-Trend1

time_all = borexino_data[:, 0]  
ref_time = Time('2011-12-11T00:00:00', format='isot', scale='utc')
start_date2 = Time('2011-12-11T00:00:00', format='isot', scale='utc')
end_date2   = Time('2013-12-11T00:00:00', format='isot', scale='utc')
start_days2 = (start_date2 - ref_time).value
end_days2 = (end_date2 - ref_time).value
index2 = np.where((time_all >= start_days2) & (time_all <= end_days2))[0]

time2 = time_all[index2]
Rate2 = borexino_data[index2, 1]
sigma2 = borexino_data[index2, 2]
Trend2 = borexino_data[index2, 3]
data2 = Rate2-Trend2

# --- Distance Calculations ---
def coordinates_to_decdeg(deg, min, sec):
    return deg + min/60 + sec/3600

latitude = coordinates_to_decdeg(42, 27, 10)
longitude = coordinates_to_decdeg(13, 34, 30)
loc = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg, height=936.45*u.m)

def distance_array(time):
    obs_time = ref_time + time*u.day
    with solar_system_ephemeris.set('de442s'):
        sun_coords = get_body('sun', obs_time, loc)
        sat_coords = get_body('saturn', obs_time, loc)
        dsun = sun_coords.distance.to(u.au)
        dsat = sat_coords.distance.to(u.au)
        return dsun, dsat

dsun1, dsat1 = distance_array(time1) 
dsun2, dsat2 = distance_array(time2)  

# --- Bayesian Model---
def loglikelihood(theta, data, sigma, dsun, dsat):
    Rsun_1AU, Rsat_10AU = theta 
    integral1 = np.mean((1.0/dsun.value)**2)
    integral2 = np.mean((10.0/dsat.value)**2)
    model = Rsun_1AU * ((1.0/dsun.value)**2 - integral1)+ Rsat_10AU *( (10.0/dsat.value)**2 - integral2)
    return -0.5 * np.sum(((model - data) / sigma) ** 2)

def logprior(theta):
    Rsun_1AU, Rsat_10AU = theta
    
    if not (-10 <= Rsat_10AU <= 10):
        return -np.inf
    
    mu_sun = 25.0
    sig_sun = 2.0
    lp_sun = -0.5 * ((Rsun_1AU - mu_sun) / sig_sun) ** 2
    
    return lp_sun

def logposterior(theta, data, sigma, dsun, dsat):
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood(theta, data, sigma, dsun, dsat)

Nens = 100
Nburnin = 1000
Nsamples = 5000
ndims = 2

Rsun_1AU_init = np.random.normal(25, 2, Nens)
Rsat_10AU_init = np.random.uniform(-10, 10, Nens)
inisamples = np.array([Rsun_1AU_init, Rsat_10AU_init]).T

def get_samples(argslist):
    sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)
    sampler.run_mcmc(inisamples, Nsamples + Nburnin, progress=True)
    samples_emcee = sampler.get_chain(flat=True, discard=Nburnin)
    return samples_emcee

argslist1 = (data1, sigma1, dsun1, dsat1)
samples1 = get_samples(argslist1)
argslist2 = (data2,sigma2,dsun2, dsat2)
samples2=get_samples(argslist2)

# --- Plotting and Analysis ---
from getdist import MCSamples
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'  
mpl.rcParams['mathtext.fontset'] = 'stix'  
mpl.rcParams['mathtext.rm'] = 'serif'  

gdsamples1 = MCSamples(samples=samples1[:, 1], names=['Rsat'], labels=[r'$\mathcal{R}_{\mathrm{sat}}$'])
gdsamples2 = MCSamples(samples=samples2[:, 1], names=['Rsat'], labels=[r'$\mathcal{R}_{\mathrm{sat}}$'])

# Get normalized 1D densities
density1 = gdsamples1.get1DDensity('Rsat')
density2 = gdsamples2.get1DDensity('Rsat')
density1.P = density1.P 
density2.P = density2.P 

plt.plot(density1.x, density1.P, color='red', linewidth=3, label='2019–2021')
plt.plot(density2.x, density2.P, color='blue', linestyle='--', linewidth=3, label='2011–2013')

plt.xlabel(r'$\mathbf{\frac{\mathcal{R}_{\mathrm{sat}}}{(10\,\mathrm{AU})^2}}$ (cpd/100t)', fontsize=16)
plt.ylabel(r'$\mathbf{P/P_{\mathrm{max}}}$', fontsize=16)
plt.ylabel(r'$\mathbf{P/P_{\mathrm{max}}}$', fontsize=16)

plt.legend(fontsize=14, prop={'weight': 'bold'})
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# --- Print Results ---
def median_and_cred(samples):
    median = np.median(samples)
    low = np.percentile(samples, 16)
    high = np.percentile(samples, 84)
    return median, high - median, median - low

def print_res(samples_emcee):
    rsun_1au, rsun_hi, rsun_lo = median_and_cred(samples_emcee[:, 0])
    rsat_10AU, rsat_hi, rsat_lo = median_and_cred(samples_emcee[:, 1])
    print(f"Rsun/(1 AU)^2: {rsun_1au:.2f} +{rsun_hi:.2f} -{rsun_lo:.2f} (cpd/100t)")
    print(f"Rsat/(10 AU)^2: {rsat_10AU:.2f} +{rsat_hi:.2f} -{rsat_lo:.2f} (cpd/100t)")

print("Results from 2019-2021")
print_res(samples1)
print("Results from 2011-2013")
print_res(samples2)
