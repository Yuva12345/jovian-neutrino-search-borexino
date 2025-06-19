import numpy as np
import matplotlib.pyplot as plt
import emcee
from astropy.time import Time 
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.coordinates import get_body
from astropy.coordinates import solar_system_ephemeris
from scipy.special import ndtri 

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
trend=borexino_data[index,3]

# --- Distance Calculations ---
def coordinates_to_decdeg(deg, min, sec):
    return deg + min/60 + sec/3600

latitude = coordinates_to_decdeg(42, 27, 10)
longitude = coordinates_to_decdeg(13, 34, 30)
loc = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg, height=936.45*u.m)

def distance_array(time1):
    obs_time = ref_time + time1*u.day
    with solar_system_ephemeris.set('de442s'):
        sun_coords = get_body('sun', obs_time, loc)
        ven_coords = get_body('venus', obs_time, loc)
        jup_coords = get_body('jupiter', obs_time, loc)
        sat_coords = get_body('saturn',obs_time,loc)
        dsun = sun_coords.distance.to(u.au)
        dven = ven_coords.distance.to(u.au)
        djup = jup_coords.distance.to(u.au)
        dsat =  sat_coords.distance.to(u.au)
        return dsun, dven, djup, dsat

dsun, dven, djup, dsat = distance_array(time)
print(np.mean(djup))
print(np.mean(dven))
print(np.mean(dsun))
print(np.mean(dsat))

# --- Bayesian Model for Jupiter---
def loglikelihood_sun_jupiter(theta, data, sigma, dsun, djup):
    Rsun_1AU, Rjup_5AU, Rb = theta
    model = Rsun_1AU * (1.0/dsun.value)**2 + Rjup_5AU * (5.0/djup.value)**2 + Rb
    return -0.5 * np.sum(((model - data) / sigma) ** 2)

def logprior_sun_jupiter(theta):
    Rsun, Rjup,Rb = theta
    if not (0 <= Rjup <= 4 and 0 <= Rb <= 50):
        return -np.inf
    mu_sun = 25.0
    sig_sun = 2.0
    lp_sun = -0.5 * ((Rsun - mu_sun) / sig_sun) ** 2
    return lp_sun

def logposterior_sun_jupiter(theta, data, sigma, dsun, djup):
    lp = logprior_sun_jupiter(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood_sun_jupiter(theta, data, sigma, dsun, djup)

Nens = 100
Nburnin = 1000
Nsamples = 5000

Rsun_1AU_init = np.random.normal(25, 2, Nens)
Rven_init = np.random.uniform(-50, 50, Nens)
Rjup_init = np.random.uniform(0, 4, Nens)
Rsat_init = np.random.uniform(-10, 10, Nens)
Rb_init = np.random.uniform(0,50,Nens)
inisamples_sun_jup = np.array([Rsun_1AU_init, Rjup_init,Rb_init]).T

argslist1 = (data, sigma, dsun, djup)
sampler_sun_jup = emcee.EnsembleSampler(Nens, 3, logposterior_sun_jupiter, args=argslist1)
sampler_sun_jup.run_mcmc(inisamples_sun_jup, Nsamples + Nburnin, progress=True)
samples_sun_jup = sampler_sun_jup.get_chain(flat=True, discard=Nburnin)

print("Sun+Jupiter model completed!")

# --- Bayesian Model for Venus---
def loglikelihood_sun_venus(theta, data, sigma, dsun, dven):
    Rsun_1AU, Rven_1AU, Rb = theta
    model = Rsun_1AU * (1.0/dsun.value)**2 + Rven_1AU * (1.0/dven.value)**2 + Rb
    return -0.5 * np.sum(((model - data) / sigma) ** 2)

def logprior_sun_venus(theta):
    Rsun, Rven, Rb = theta
    if not (-50 <= Rven <= 50 and 0 <= Rb <= 50):
        return -np.inf
    mu_sun = 25.0
    sig_sun = 2.0
    lp_sun = -0.5 * ((Rsun - mu_sun) / sig_sun) ** 2
    return lp_sun

def logposterior_sun_venus(theta, data, sigma, dsun, dven):
    lp = logprior_sun_venus(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood_sun_venus(theta, data, sigma, dsun, dven)

Nens = 100
Nburnin = 1000
Nsamples = 5000

inisamples_sun_ven = np.array([Rsun_1AU_init, Rven_init,Rb_init]).T

argslist2 = (data, sigma, dsun, dven)
sampler_sun_venus = emcee.EnsembleSampler(Nens, 3, logposterior_sun_venus, args=argslist2)
sampler_sun_venus.run_mcmc(inisamples_sun_ven, Nsamples + Nburnin, progress=True)
samples_sun_venus = sampler_sun_venus.get_chain(flat=True, discard=Nburnin)

print("Sun+Venus model completed!")

# --- Bayesian Model for Saturn---
def loglikelihood_sun_saturn(theta, data, sigma, dsun, dsat):
    Rsun_1AU, Rsat_10AU, Rb = theta
    model = Rsun_1AU * (1.0/dsun.value)**2 + Rsat_10AU * (10.0/dsat.value)**2 + Rb
    return -0.5 * np.sum(((model - data) / sigma) ** 2)

def logprior_sun_sat(theta):
    Rsun, Rsat, Rb = theta
    if not (-10 <= Rsat <= 10 and 0 <= Rb <= 50):
        return -np.inf
    mu_sun = 25.0
    sig_sun = 2.0
    lp_sun = -0.5 * ((Rsun - mu_sun) / sig_sun) ** 2
    return lp_sun

def logposterior_sun_sat(theta, data, sigma, dsun, dsat):
    lp = logprior_sun_sat(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood_sun_saturn(theta, data, sigma, dsun, dsat)


Nens = 100
Nburnin = 1000
Nsamples = 5000

inisamples_sun_sat = np.array([Rsun_1AU_init, Rsat_init,Rb_init]).T

argslist4 = (data, sigma, dsun, dsat)
sampler_sun_sat = emcee.EnsembleSampler(Nens, 3, logposterior_sun_sat, args=argslist4)
sampler_sun_sat.run_mcmc(inisamples_sun_sat, Nsamples + Nburnin, progress=True)
samples_sun_sat = sampler_sun_sat.get_chain(flat=True, discard=Nburnin)

print("Sun+Saturn model completed!")

# --- Bayesian Model for Sun---
def loglikelihood_sun_only(theta, data, sigma, dsun):
    Rsun_1AU, Rb = theta
    model = Rsun_1AU * (1.0/dsun.value)**2 + Rb
    return -0.5 * np.sum(((model - data) / sigma) ** 2)

def logprior_sun_only(theta):
    Rsun, Rb = theta
    if not (0 <= Rb <= 50):
        return -np.inf
    mu_sun = 25.0
    sig_sun = 2.0
    lp_sun = -0.5 * ((Rsun - mu_sun) / sig_sun) ** 2
    return lp_sun

def logposterior_sun_only(theta, data, sigma, dsun):
    lp = logprior_sun_only(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood_sun_only(theta, data, sigma, dsun)

inisamples_sun_only = np.array([Rsun_1AU_init, Rb_init]).T  # Shape (100, 2)

argslist3 = (data, sigma, dsun)
sampler_sun_only = emcee.EnsembleSampler(Nens, 2, logposterior_sun_only, args=argslist3)
sampler_sun_only.run_mcmc(inisamples_sun_only, Nsamples + Nburnin, progress=True)
samples_sun_only = sampler_sun_only.get_chain(flat=True, discard=Nburnin)

print("Sun-only model completed!")

# --- Best fit values---
def median_and_cred(samples):
    median = np.median(samples)
    low = np.percentile(samples, 16)
    high = np.percentile(samples, 84)
    return median, high - median, median - low

rsun_sv, rsun_sv_hi, rsun_sv_lo = median_and_cred(samples_sun_venus[:, 0])
rven_sv, rven_sv_hi, rven_sv_lo = median_and_cred(samples_sun_venus[:, 1])
rb_sv, rb_sv_hi, rb_sv_lo = median_and_cred(samples_sun_venus[:, 2])

rsun_sj, rsun_sj_hi, rsun_sj_lo = median_and_cred(samples_sun_jup[:, 0])
rjup_sj, rjup_sj_hi, rjup_sj_lo = median_and_cred(samples_sun_jup[:, 1])
rb_sj, rb_sj_hi, rb_sj_lo = median_and_cred(samples_sun_jup[:, 2])

rsun_ss, rsun_ss_hi, rsun_ss_lo = median_and_cred(samples_sun_sat[:, 0])
rsat_ss, rsat_ss_hi, rsat_ss_lo = median_and_cred(samples_sun_sat[:, 1])
rb_ss, rb_ss_hi, rb_ss_lo = median_and_cred(samples_sun_sat[:, 2])

rsun_so, rsun_so_hi, rsun_so_lo = median_and_cred(samples_sun_only[:, 0])
rb_so, rb_so_hi, rb_so_lo = median_and_cred(samples_sun_only[:, 1])

print(f"\nSun+Venus model:")
print(f"Rsun/(1 AU)²: {rsun_sv:.2f} +{rsun_sv_hi:.2f} -{rsun_sv_lo:.2f} (cpd/100t)")
print(f"Rven/(1 AU)²: {rven_sv:.2f} +{rven_sv_hi:.2f} -{rven_sv_lo:.2f} (cpd/100t)")
print(f"Rb: {rb_sv:.2f} +{rb_sv_hi:.2f} -{rb_sv_lo:.2f} (cpd/100t)")

print(f"\nSun+Jupiter model:")
print(f"Rsun/(1 AU)²: {rsun_sj:.2f} +{rsun_sj_hi:.2f} -{rsun_sj_lo:.2f} (cpd/100t)")
print(f"Rjup/(5 AU)²: {rjup_sj:.2f} +{rjup_sj_hi:.2f} -{rjup_sj_lo:.2f} (cpd/100t)")
print(f"Rb: {rb_sj:.2f} +{rb_sj_hi:.2f} -{rb_sj_lo:.2f} (cpd/100t)")

print(f"\nSun+Saturn model:")
print(f"Rsun/(1 AU)²: {rsun_ss:.2f} +{rsun_ss_hi:.2f} -{rsun_ss_lo:.2f} (cpd/100t)")
print(f"Rsat/(10 AU)²: {rsat_ss:.2f} +{rsat_ss_hi:.2f} -{rsat_ss_lo:.2f} (cpd/100t)")
print(f"Rb: {rb_ss:.2f} +{rb_ss_hi:.2f} -{rb_ss_lo:.2f} (cpd/100t)")

print(f"\nSun-only model:")
print(f"Rsun/(1 AU)²: {rsun_so:.2f} +{rsun_so_hi:.2f} -{rsun_so_lo:.2f} (cpd/100t)")
print(f"Rb/(1 AU)²: {rb_so:.2f} +{rb_so_hi:.2f} -{rb_so_lo:.2f} (cpd/100t)")
    

def model_sun_jup(time, Rsun_1AU, Rjup_5AU, Rb, dsun, djup):
    model = Rsun_1AU * (1.0/dsun.value)**2 + Rjup_5AU * (5.0/djup.value)**2 + Rb
    return model
    
def model_sun_ven(time, Rsun_1AU, Rven_1AU, Rb, dsun, dven):
    model = Rsun_1AU * (1.0/dsun.value)**2 + Rven_1AU * (1.0/dven.value)**2 + Rb
    return model

def model_sun_sat(time, Rsun_1AU, Rsat_10AU, Rb, dsun, dsat):
    model = Rsun_1AU * (1.0/dsun.value)**2 + Rsat_10AU * (10.0/dsat.value)**2 + Rb
    return model

def model_sun_only(time, Rsun_1AU, Rb, dsun):
    return Rsun_1AU * (1.0 / dsun.value)**2 + Rb


time_fine = np.linspace(time.min(), time.max(), 500)
dsun_fine, dven_fine, djup_fine, dsat_fine = distance_array(time_fine)

model_best_fit_jup = model_sun_jup(time_fine, rsun_sj, rjup_sj, rb_sj, dsun_fine, djup_fine)
model_best_fit_ven = model_sun_ven(time_fine, rsun_sv, rven_sv, rb_sv, dsun_fine, dven_fine)
model_best_fit_sat = model_sun_sat(time_fine, rsun_ss, rsat_ss, rb_ss, dsun_fine, dsat_fine)
model_sun_only_vals = model_sun_only(time_fine, rsun_so, rb_so, dsun_fine)

# --- Plotting and Analysis ---
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'  
mpl.rcParams['mathtext.fontset'] = 'stix'  
mpl.rcParams['mathtext.rm'] = 'serif'   

plt.figure(figsize=(12, 7))
time_shifted = time - start_days
time_fine_shifted = time_fine - start_days

plt.errorbar(time_shifted, data, yerr=sigma, fmt='o', color='red', markersize=6, alpha=0.7, label='Rate time-series')

plt.plot(time_fine_shifted, model_best_fit_jup, 'k-', linewidth=2.5, 
         label=r'$\frac{\mathcal{R}_\mathrm{sun}}{(1~\mathrm{AU})^2} + \frac{\mathcal{R}_\mathrm{jup}}{(5~\mathrm{AU})^2} + \mathcal{R}_\mathrm{B}$')

plt.plot(time_fine_shifted, model_best_fit_ven, 'g-', linewidth=2.5, 
         label=r'$\frac{\mathcal{R}_\mathrm{sun}}{(1~\mathrm{AU})^2} + \frac{\mathcal{R}_\mathrm{ven}}{(1~\mathrm{AU})^2} + \mathcal{R}_\mathrm{B}$')

plt.plot(time_fine_shifted, model_best_fit_sat, color='magenta', linestyle='--', linewidth=2.5, 
         label=r'$\frac{\mathcal{R}_\mathrm{sun}}{(1~\mathrm{AU})^2} + \frac{\mathcal{R}_\mathrm{sat}}{(10~\mathrm{AU})^2} + \mathcal{R}_\mathrm{B}$')

plt.plot(time_fine_shifted, model_sun_only_vals, 'b--', linewidth=2.5, 
         label=r'$\frac{\mathcal{R}_\mathrm{sun}}{(1~\mathrm{AU})^2} + \mathcal{R}_\mathrm{B}$')

plt.xlabel(r'Days since 2019-10-01', fontsize=16)

plt.ylabel(r'$\mathcal{R}_i\ \mathrm{[cpd/100t]}$', fontsize=16)
plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1, 1.15), frameon=True)
plt.grid(True, alpha=0.3)
plt.gca().tick_params(axis='both', which='major', labelsize=12, width=1.5)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.show()

# --- Goodness of fit test ---
model_sj_data= model_sun_jup(time, rsun_sj, rjup_sj, rb_sj, dsun, djup)
model_sv_data= model_sun_ven(time, rsun_sv, rven_sv, rb_sv, dsun, dven)
model_ss_data= model_sun_sat(time, rsun_ss, rsat_ss, rb_ss, dsun, dsat)
model_so_data= model_sun_only(time, rsun_so, rb_so, dsun)

chi2_sun_jupiter = np.sum(((model_sj_data - data) / sigma)**2)
chi2_sun_venus = np.sum(((model_sv_data - data) / sigma)**2)
chi2_sun_saturn = np.sum(((model_ss_data - data) / sigma)**2)
chi2_sun_only = np.sum(((model_so_data - data) / sigma)**2)

print(f"\nModel comparison:")
print(f"χ² (Sun+Jupiter): {chi2_sun_jupiter:.1f}")
print(f"χ² (Sun+Venus): {chi2_sun_venus:.1f}")
print(f"χ² (Sun+Saturn): {chi2_sun_saturn:.1f}")
print(f"χ² (Sun only): {chi2_sun_only:.1f}")

def dof(dp, fp):
    return dp-fp

dp1=len(data)
fp1=3
fp2=2
dof_jup=dof(dp1,fp1)
dof_ven=dof(dp1,fp1)
dof_sat=dof(dp1,fp1)
dof_sun=dof(dp1,fp2)
print(dof_jup,dof_ven,dof_sat, dof_sun)
chi2_sun_jupiter_dof = chi2_sun_jupiter/dof_jup
chi2_sun_venus_dof = chi2_sun_venus/dof_ven
chi2_sun_saturn_dof = chi2_sun_saturn/dof_sat
chi2_sun_only_dof = chi2_sun_only/dof_sun

print(f"χ²/dof (Sun+Jupiter): {chi2_sun_jupiter_dof:.1f}")
print(f"χ²/dof (Sun+Venus): {chi2_sun_venus_dof:.1f}")
print(f"χ²/dof (Sun+Saturn): {chi2_sun_saturn_dof:.1f}")
print(f"χ²/dof (Sun only): {chi2_sun_only_dof:.1f}")
