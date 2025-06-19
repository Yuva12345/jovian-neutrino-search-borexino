from scipy.special import ndtri 
import numpy as np
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
    jup_coords = get_body('jupiter', obs_time, loc)

dsun = sun_coords.distance.to(u.au)
djup = jup_coords.distance.to(u.au)

# --- Dynesty ---
M = len(data)

def prior_transform(theta): 
    Rsunprime, Rjupprime, Rbprime = theta
    min1 = 0
    max1 = 50
    min2 = 0
    max2 = 4
    musun=25 
    sunsigma=2 
    Rsun=musun + sunsigma*ndtri(Rsunprime) 
    Rjup = Rjupprime * (max2 - min2) + min2
    Rb = Rbprime * (max1 - min1) + min1
    return (Rsun, Rjup, Rb)

LN2PI = np.log(2. * np.pi)
LNSIGMA = np.log(sigma)

def loglikelihood_dynesty(theta):
    Rsun, Rjup, Rb = theta
    norm = -0.5 * M * LN2PI - np.sum(np.log(sigma))
    model = Rsun * (1.0/dsun.value)**2 + Rjup * (5.0/djup.value)**2 + Rb
    chisq = np.sum(((data - model) / sigma) ** 2)
    return norm - 0.5 * chisq

nlive = 1024
bound = 'multi'
ndims = 3
sample = 'unif'
tol = 0.1

from dynesty import DynamicNestedSampler
dsampler = DynamicNestedSampler(loglikelihood_dynesty,prior_transform, ndims, bound=bound, sample=sample) 
dsampler.run_nested(nlive_init=nlive,print_progress= True)
dres= dsampler.results 

# --- Print evidence ---
logZdynestydynamic=dres.logz[-1] 
logZerrdynestydynamic=dres.logzerr[-1] 
print("Dynamic: log(Z) = {} ± {}".format(logZdynestydynamic, logZerrdynestydynamic))

