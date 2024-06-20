# ##Runup and Reverse Shoaling Calculations
# 
# ###  Reverse Shoaling
# Reverse shoaling is used to estimate deepwater wave height Ho from wave heights measured at intermediate depths equal to, or deeper than, the breaking depth hb.
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
g = 9.81
pi = np.pi


def qkhfs( w, h ):
    """
    Quick iterative calculation of kh in gravity-wave dispersion relationship
    kh = qkhfs(w, h )
    
    Input
        w - array of angular wave frequencies = 2*pi/T where T = wave period [1/s]
        h - array of water depths [m]
    Returns
        kh - [len(wa), len(ha)] array of wavenumber * depth [ ]

    Orbital velocities from kh are accurate to 3e-12 !

    RL Soulsby (2006) \"Simplified calculation of wave orbital velocities\"
    HR Wallingford Report TR 155, February 2006
    Eqns. 12a - 14
    """
    tanh = np.tanh
    g = 9.81
    x = w**2.0 *h/g
    y = np.sqrt(x) * (x<1.) + x *(x>=1.)
    # is this faster than a loop?
    t = tanh( y )
    y = y-( (y*t -x)/(t+y*(1.0-t**2.0)))
    t = tanh( y )
    y = y-( (y*t -x)/(t+y*(1.0-t**2.0)))
    t = tanh( y )
    y = y-( (y*t -x)/(t+y*(1.0-t**2.0)))
    kh = y
    return kh


T = 10. # wave period (s)
H = 1.54
h = 8.
Bf=.08 # beach steepness in foreshore ()
gam = 0.78 # Ratio height to depth at breaking Hb/hb. Could 0.78.

# Case 1 - Measured at break height
Hb = H
hb = Hb/gam
w = 2.*pi/T
Lo = g*T*T/(2.*pi)
ko = 2*pi/Lo
kh = qkhfs( w, hb )
Ks = 1./ np.sqrt( np.tanh(kh)*(1.+2.*kh/np.sinh(2.*kh)) )
Ksn = (4.*ko*h)**(-1./4.) * (1.+0.25*ko*h+(13./228.)*(ko*h)**2.)
Ksb = (4.*ko*h)**(-1./4.) * (1.+0.25*ko*h+(13./228.)*(ko*h)**2.)*(1.+(3./8.)*(H/Lo)**1.5*(ko*h)**(-3.))


Ho = Hb/Ks # deepwater wave height (m)
Lo = g*T*T/(2.*pi)
I = Bf*(np.sqrt(Lo/Ho))
print 'Deepwater wave height Ho = ',Ho,' m'
print 'Break height Hb = ',Hb,' m'
print 'Break depth hb = ',hb,' m'
print 'Shoaling coefficient Ks = ',Ks,' ()'
print 'Approximate explicit shoaling coefficient Ks = ',Ksn,' ()'
print '...with breaker correction Ksb = ',Ksb,' ()'
print 'Deepwater wavelength Lo = ',Lo,' m'
print 'Irribarren number = ',I
eta = 0.35*Bf*np.sqrt(Ho*Lo)
Sinc = 0.75*Bf*np.sqrt(Ho*Lo)
SIG = 0.06*np.sqrt(Ho*Lo)
R2 = 1.1*(eta+0.5*np.sqrt(Sinc**2+SIG**2)) #Eqns 6 and 7
print "R2, eta, Sinc, SIG: ",R2,eta,Sinc,SIG
if(I<0.3):
    R2L = 0.043*np.sqrt(Ho*Lo)
    print "Dissipative R2: ",R2L
Bang = np.arctan(Bf)
x = R2/np.sin(Bang)
print "Beach angle, runup distance x: {0}, {1}",Bang*180./pi, x


# Case 2 - Measured in deep water
Ho = H
Lo = g*T*T/(2.*pi)
ko = 2*pi/Lo
Hb = 0.39*pow(g,(1./5.))*pow( T*Ho*Ho, (2./5.) ) # Komar 6.6
hb = Hb/gam
kh = qkhfs( w, hb )
Ks = 1./ np.sqrt( np.tanh(kh)*(1.+2.*kh/np.sinh(2.*kh)) )
Ksn = (4.*ko*h)**(-1./4.) * (1.+0.25*ko*h+(13./228.)*(ko*h)**2.)
Ksb = (4.*ko*h)**(-1./4.) * (1.+0.25*ko*h+(13./228.)*(ko*h)**2.)*(1.+(3./8.)*(Ho/Lo)**1.5*(ko*h)**(-3.))

I = Bf*(np.sqrt(Lo/Ho))
print 'Deepwater wave height Ho = ',Ho,' m'
print 'Break height Hb = ',Hb,' m'
print 'Break depth hb = ',hb,' m'
print 'Shoaling coefficient Ks = ',Ks,' ()'
print 'Approximate explicit shoaling coefficient Ks = ',Ksn,' ()'
print '...with breaker correction Ksb = ',Ksb,' ()'
print 'Deepwater wavelength Lo = ',Lo,' m'
print 'Irribarren number = ',I
eta = 0.35*Bf*np.sqrt(Ho*Lo)
Sinc = 0.75*Bf*np.sqrt(Ho*Lo)
SIG = 0.06*np.sqrt(Ho*Lo)
R2 = 1.1*(eta+0.5*np.sqrt(Sinc**2+SIG**2)) #Eqns 6 and 7
print "R2, eta, Sinc, SIG: ",R2,eta,Sinc,SIG
if(I<0.3):
    R2L = 0.043*np.sqrt(Ho*Lo)
    print "Dissipative R2: ",R2L
Bang = np.arctan(Bf)
x = R2/np.sin(Bang)
print "Beach angle, runup distance x: ",Bang*180./pi, x


h = np.logspace(-2, 3., 50)
w = 2.*pi/T
ko = 2*pi/Lo
wa = np.array([w])
kh = np.squeeze(qkhfs( wa, h ))
Cgo = 0.5*g*T/(2*pi)
n = 0.5+kh/np.sinh(2.*kh)
Cg = n*g*T/(2.*pi)
Ks = 1./ np.sqrt( np.tanh(kh)*(1.+2*kh/np.sinh(2*kh)) )
Ksb = (4.*ko*h)**(-1./4.) * (1.+0.25*ko*h+(13./228.)*(ko*h)**2.)*(1.+(3./8.)*(Ho/Lo)**1.5*(ko*h)**(-3.))
print np.shape(h), np.shape(Ks), np.shape(Ksb)
plt.plot(h/Lo,Ks)
plt.plot(h/Lo,Ksb)
plt.xlim((0.,.05))
plt.ylim((.9,2.2))


eta = 0.35*Bf*np.sqrt(Ho*Lo)
Sinc = 0.75*Bf*np.sqrt(Ho*Lo)
SIG = 0.06*np.sqrt(Ho*Lo)
R2 = 1.1*(eta*0.5*np.sqrt(Sinc**2+SIG**2)) #Eqns 6 and 7


# <h2>Bateman Equations for Multiple Decay</h2>
# 
# The inventory of each of four compounds was calculated assuming a multi-step reaction model. These calculations require several assumptions (Eganhouse and Pontolillo, 2008), as follows: (1) reactions and loss occur through first-order kinetics; (2) the only competing reaction is loss through unspecified physical processes discussed below; and (3) rates are constant over time and uniform throughout the sediment deposit (except the physical loss rate). The time-rate of change in $DDE$ inventory was determined by dechlorination to $DDMU$ and by possible losses through other processes. Likewise, the change in $DDMU$ inventory was determined losses from dechlorination to $DDNU$, gains from dechlorination of parent $DDE$, and losses by other processes. $DDNU$ inventory changes were analogous, with losses to an unspecified compound, gains from parent $DDMU$, and possible losses through other processes. The inventory of the final, unspecified product changes with gains from transformation of $DDMU$ and possible losses to other processes. The coupled differential equations (1) are
# 
# $\begin{matrix}
# \frac{d{{C}_{DDE}}}{dt}=-{{\lambda }_{DDE\to DDMU}}{{C}_{DDE}}-{{\lambda }_{Loss}}{{C}_{DDE}}  \   \frac{d{{C}_{DDMU}}}{dt}=-{{\lambda }_{DDMU\to DDNS}}{{C}_{DDMU}}+{{\lambda }_{DDE\to DDMU}}{{C}_{DDE}}-{{\lambda }_{Loss}}{{C}_{DDMU}}  \   \frac{d{{C}_{DDNU}}}{dt}=-{{\lambda }_{DDNU\to ?}}{{C}_{DDNU}}+{{\lambda }_{DDMU\to DDNU}}{{C}_{DDMU}}-{{\lambda }_{Loss}}{{C}_{DDNU}}  \   \frac{d{{C}_{UN}}}{dt}={{\lambda }_{DDNU\to UN}}{{C}_{DDNU}}-{{\lambda }_{Loss}}{{C}_{UN}}  \\end{matrix}$
# 
# where the molar concentrations $C$ [$\mu$mol/kg dry sediment] of each compound and the transformation rate coefficients  [y-1] are labeled with subscripts. The losses to other processes accounted for physical removal of compounds from the sediment. Processes that have been suggested include direct desorption, resuspension and desorption, desorption into porewater and irrigation, and uptake by benthic deposit feeders. We have assumed that the combined rate coefficient for these processes applies equally to all four compounds. 
# 

from scipy.integrate import odeint
from pylab import *
get_ipython().magic('matplotlib inline')

# decay rates are global so they can be seen inside the function 
global lam
lam = array([.04, .01, .000, .0])

# define a function to represent coupled ordinary differential eqns.
def dcdt(c,t):
    dfdt = np.zeros(4)
    dfdt[0] = c[0]* -lam[0] - c[0]*lam[3]
    dfdt[1] = c[1]* -lam[1] + c[0]*lam[0] - c[1]*lam[3] 
    dfdt[2] = c[2]* -lam[2] + c[1]*lam[1] - c[2]*lam[3]
    dfdt[3] =                 c[2]*lam[2] - c[3]*lam[3]
    return dfdt
    
# intial concentration for four constituents
C0 = array([.68, .23, .06, 0.])
# time array
t = linspace(0.0,100.,50)
# call 
C = odeint(dcdt,C0,t)

print "Shape of the final concentration matrix: ",shape(C)
fig = plt.figure()
plt.plot(t,C[:,0],label='$DDE$')
plt.plot(t,C[:,1],label='$DDMU$')
plt.plot(t,C[:,2],label='$DDNU$')
plt.plot(t,C[:,3],label='?')
plt.plot(t,np.sum(C,1),label='Total')
plt.xlabel('Time (years)')
plt.ylabel('Concentration ($\mu$mol / kg)')
plt.legend(loc='upper right')


whos


# <h2>Read in latest HRRR dataset using Siphon query and write local netCDF file</h2>
# 

import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

# Resolve the latest HRRR dataset
from siphon.catalog import TDSCatalog
latest_hrrr = TDSCatalog('http://thredds-jumbo.unidata.ucar.edu/thredds/catalog/grib/HRRR/CONUS_3km/surface/latest.xml')
hrrr_ds = list(latest_hrrr.datasets.values())[0]

# Set up access via NCSS
from siphon.ncss import NCSS
ncss = NCSS(hrrr_ds.access_urls['NetcdfSubset'])

# Create a query to ask for all times in netcdf4 format for
# the Temperature_surface variable, with a bounding box
query = ncss.query()
dap_url = hrrr_ds.access_urls['OPENDAP']

query.all_times().accept('netcdf4').variables('u-component_of_wind_height_above_ground',
                                              'v-component_of_wind_height_above_ground')
query.lonlat_box(45, 41., -63, -71.5)

# Get the raw bytes and write to a file.
data = ncss.get_data_raw(query)
with open('test_uv.nc', 'wb') as outf:
    outf.write(data)


# <h2>Read the netCDF file back in</h2>
# 

import xray
nc = xray.open_dataset('test_uv.nc')
nc


uvar_name='u-component_of_wind_height_above_ground'
vvar_name='v-component_of_wind_height_above_ground'
uvar = nc[uvar_name]
vvar = nc[vvar_name]
grid = nc[uvar.grid_mapping]
grid


uvar


lon0 = grid.longitude_of_central_meridian
lat0 = grid.latitude_of_projection_origin
lat1 = grid.standard_parallel
earth_radius = grid.earth_radius


# <h2>Plot Lambert Conformal with Cartopy</h2>
# 

import cartopy
import cartopy.crs as ccrs
#cartopy wants meters, not km
x = uvar.x.data*1000.
y = uvar.y.data*1000.

#globe = ccrs.Globe(ellipse='WGS84') #default
globe = ccrs.Globe(ellipse='sphere', semimajor_axis=grid.earth_radius)

crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, 
                            standard_parallels=(lat0,lat1), globe=globe)
print(uvar.x.data.shape)
print(uvar.y.data.shape)
print(uvar.time1.shape)


uvar[-1,:,:].time1.data


klev = 0
u = uvar[istep,klev,:,:].data
v = vvar[istep,klev,:,:].data
spd = np.sqrt(u*u+v*v)


fig = plt.figure(figsize=(10,16))
ax = plt.axes(projection=ccrs.PlateCarree())
c = ax.pcolormesh(x,y,spd, transform=crs,zorder=0)
cb = fig.colorbar(c,orientation='vertical',shrink=0.5)
cb.set_label('m/s')
ax.coastlines(resolution='10m',color='gray',zorder=1,linewidth=3)
ax.quiver(x,y,u,v,transform=crs,zorder=2,scale=100)
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
plt.title(uvar[istep].time1.data);
plt.axis([-71.2, -70., 42.3, 43])
#plt.axis([-72,-69.8,40.6, 43.5]);


import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')


# ### Relationship between decay rate coefficient _k_  and half life _thalf_
# 
# C = C0*exp(-kt)
# 
# Solve for: C=0.5C0
# 
# Gives: k = -ln(0.5)/t
# 

# C = Co exp(-kt )
# 0.5 = exp (-k thalf)
# 7Be, 53.3
thalf = 53.3*3600.*24.
k=-np.log(0.5)/thalf
print '7Be k =',k,' s-1'

# 234Th, 24.1 days
thalf = 24.1*3600.*24.
k=-np.log(0.5)/thalf
print '234Th k =',k,' s-1'


# ### Solution for time-dependent decay (no burial or mixing)
# 

Co=1.
k = .1
thalf = -np.log(0.5)/k
tmax = 5.*thalf
dt = tmax / 10.
t = np.arange(0.,tmax+dt,dt)

C = Co*np.exp(-k*t)

fig = plt.figure(figsize=(6,4))
plt.plot(t,C)
plt.xlabel('Time')
plt.ylabel('C/Co')
ts = 'Half-life: {0:.2f}'.format(thalf)
plt.title(ts)


# ### Solution to steady burial and decay (no mixing)
# 

import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
zmax = 1.
dz = 0.12
zb = np.arange(0.,zmax+dz,dz)
Co=1
w = 0.01
k = .1

C = Co*np.exp(-k*zb/w)

fig = plt.figure(figsize=(6,4))
plt.plot(C,-zb)
plt.xlabel('C/Co')
plt.ylabel('z (m)')
ts = 'w = {0:.2f} m/s; k = {1:.2f} 1/s'.format(w,k)
plt.title(ts)


# # Read WFS data from USGS ScienceBase into Shapely/Cartopy
# 

import numpy as np
import matplotlib.pyplot as plt
import geojson
from owslib.wfs import WebFeatureService
from shapely.geometry import Polygon, mapping, asShape, shape
import cartopy.crs as ccrs
from cartopy.io.img_tiles import MapQuestOpenAerial, MapQuestOSM, OSM
get_ipython().magic('matplotlib inline')


# Read shapefile data from USGS ScienceBase WFS 1.1 service in JSON format
# (shapefile was previosly uploaded to ScienceBase, creating the WFS service)


# getCapabilities
#https://www.sciencebase.gov/catalogMaps/mapping/ows/5342c54be4b0aa151574a8dc?service=wfs&version=1.0.0&request=GetCapabilities
# containes LatLongBoundingBox for each feature


# some USGS ScienceBase Geoserver WFS endpoints:
#endpoint='https://www.sciencebase.gov/catalogMaps/mapping/ows/5342c54be4b0aa151574a8dc'
endpoint='https://www.sciencebase.gov/catalogMaps/mapping/ows/5342c5fce4b0aa151574a8ed'
#endpoint='https://www.sciencebase.gov/catalogMaps/mapping/ows/5342e124e4b0aa151574a969'
wfs = WebFeatureService(endpoint, version='1.1.0')
print wfs.version


shp = wfs.contents.keys()
print shp


a = wfs.contents['sb:footprint']
b = a.boundingBoxWGS84


shp = filter(lambda a: a != 'sb:footprint', shp)
print shp


def flip_geojson_coordinates(geo):
    if isinstance(geo, dict):
        for k, v in geo.iteritems():
            if k == "coordinates":
                z = np.asarray(geo[k])
                f = z.flatten()
                geo[k] = np.dstack((f[1::2], f[::2])).reshape(z.shape).tolist()
            else:
                flip_geojson_coordinates(v)
    elif isinstance(geo, list):
        for k in geo:
            flip_geojson_coordinates(k)


#srs='EPSG:4326' # v1.0 syntax
srs='urn:x-ogc:def:crs:EPSG:4326'  # v1.1 syntax
json_response = wfs.getfeature(typename=[shp[0]], propertyname=None, srsname=srs, outputFormat='application/json').read()
geo = geojson.loads(json_response)
flip_geojson_coordinates(geo)


print geo.keys()


print geo['type']


geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))

plt.figure(figsize=(12,12))
# Open Source Imagery from MapQuest (max zoom = 16?)
tiler = MapQuestOpenAerial()
# Open Street Map (max zoom = 18?)
#tiler = OSM()
ax = plt.axes(projection=tiler.crs)
dx=b[2]-b[0]
dy=b[3]-b[1]
extent = (b[0]-0.1*dx,b[2]+0.1*dx,b[1]-0.1*dy,b[3]+0.1*dy)
ax.set_extent(extent, geodetic)
ax.add_image(tiler, 14)
#ax.add_geometries([polygon],ccrs.PlateCarree(),
#                          facecolor=BLUE, edgecolor=GRAY,alpha=0.5)
for p in geo.get("features", []):
    multi_poly = asShape(p.get("geometry"))
    print 'bounds from Shapely: ',multi_poly.bounds
#    name=p['properties']['NAME']
#    print name
    ax.add_geometries(multi_poly,ccrs.PlateCarree(),
                edgecolor='black',facecolor='none',hatch='/')
#title(name)
    
gl=ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
#ax.add_feature(coast_10m,edgecolor='black')
#ax.coastlines()


# Start with wave statistics Hs, Td, Dir, h
# Also need currents
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
pi = np.pi
sin = np.sin
cos = np.cos
sqrt = np.sqrt
tanh = np.tanh
arcsin = np.arcsin
arccos = np.arccos
exp = np.exp
log = np.log
plot = plt.plot


# Check to see the Abreu time series works
# select r and phi to match curves in Figs 1 and 2
r = .8
phi = -pi/4.
Uw = 1.
T = 5.
n = 50
w = 2.*pi/T
wt = np.linspace( 0., 2.*pi, n) # phase
f = sqrt( 1. - r**2 )
numer = sin(wt) + ( r*sin(phi)/(1.+sqrt(1.-r**2)) )
denom = (1.-r*cos(wt+phi))
ut = Uw*f*numer/denom
numer2 = cos(wt)-r*cos(phi)-r**2/(1.+sqrt(1.-r**2))*sin(phi)*sin(wt+phi)
at = Uw*w*f*numer2/denom**2

# alternative formulation Eqns 16a,b in Malarkey & Davies
phi = -phi #
P = sqrt(1.-r*r) # same as f
b = r/(1.+P)
fbt = 1.-b*b
numer = sin(wt)-b*sin(phi)
denom = (1.+b*b-2.*b*cos(wt-phi))
utm = Uw*fbt*numer/denom
numer2 = (1.+b*b)*cos(wt)-2.*b*cos(phi)+2.*b*b*sin(phi)*sin(wt-phi)
atm = Uw*w*fbt*numer2/denom**2

# Appendix E of Malarkey * Davies
# Phase of umax, and umin
c = b*sin(phi)
tmm = arcsin((4.*c*(b*b-c*c)-(1.-b*b)*(1.+b*b-2.*c*c))/((1.+b*b)**2-4.*c*c))
tmp = arcsin((4.*c*(b*b-c*c)+(1.-b*b)*(1.+b*b-2.*c*c))/((1.+b*b)**2-4.*c*c))
if(tmm<0.):
    tmm = tmm+2.*pi
if(tmp<0.):
    tmp = tmp+2*pi
print tmm, tmp
umax = 1+c
umin = umax-2
# zero upcrossing
tz = arcsin(b*sin(phi)) # = arcsin(c)
tzd = 2.*arccos(c)+tz
# sigma Eqn 19
sig1 = arcsin( (4.*c*(b*b-c*c)+(1.-b*b)*(1.+b*b-2.*c*c))/((1.+b*b)**2-4.*c*c) )
if( phi <= 0.5*pi ):
    sig = (1./pi)*(sig1-tz)
else :
    sig = (1./pi)*(pi-sig1-tz)
print tz, sig, tzd
print (tmp-tz)/pi # sigma from Eqn 5

plot(wt/(2.*pi),ut/Uw)
plot(wt/(2.*pi),at/(Uw*w))
plot(wt/(2.*pi),utm/Uw,'--')
plot(wt/(2.*pi),atm/(Uw*w),'--')
plot(tmm/(2.*pi),umin,'or')
plot(tmp/(2.*pi),umax,'ob')
plot(tz/(2.*pi),0,'ok')
plot(tzd/(2.*pi),0,'ok')
plt.xlabel('t/T')
plt.ylabel('u/Uw, a/(Uw $\omega$)')


def ursell( aw, k, h ):
    """
    Calculate Ursell number
    Reussink et al. Eqn 6.
    """
    return (3./4.)*aw*k/(k*h)**3.

def qkhfs( w, h ):
    """
    Quick iterative calculation of kh in gravity-wave dispersion relationship
    kh = qkhfs(w, h )
    
    Input
        w - angular wave frequency = 2*pi/T where T = wave period [1/s]
        h - water depth [m]
    Returns
        kh - wavenumber * depth [ ]

    Orbital velocities from kh are accurate to 3e-12 !

    RL Soulsby (2006) \"Simplified calculation of wave orbital velocities\"
    HR Wallingford Report TR 155, February 2006
    Eqns. 12a - 14
    """
    g = 9.81
    x = w**2.0 *h/g
    y = sqrt(x) * (x<1.) + x *(x>=1.)
    # is this faster than a loop?
    t = tanh( y )
    y = y-( (y*t -x)/(t+y*(1.0-t**2.0)))
    t = tanh( y )
    y = y-( (y*t -x)/(t+y*(1.0-t**2.0)))
    t = tanh( y )
    y = y-( (y*t -x)/(t+y*(1.0-t**2.0)))
    kh = y
    return kh

def Bfit( Ur ):
    """
    Ruessink et al. Eqn. 9
    """
    p1 = 0.
    p2 = 0.857
    p3 = -0.471
    p4 = 0.297
    B = p1 + (p2 - p1)/(1 + exp( (p3-log(Ur))/p4 ))
    return B
    
def Phifit( Ur ):
    # Ruessink et al. eqn 10
    dtr = pi/180.
    p5 = 0.815
    p6 = 0.672
    phi = dtr*(-90.) + dtr*90. * tanh(p5/(Ur**p6))
    return phi # reverse sign for Malarkey & Davies formula

def rfromB ( B ):
    # Solve Ruessink eqn. 11 for r
    b = sqrt(2.)*B/(sqrt(B**2+9)) # use directly in M&D formula
    r = 2.*b/(b**2+1.)
    return r

def abreu_ut ( Uw, r, phi, w ):
    """
    Calculate u(t) and a(t) using Abreu et al. (2010) eqn. 7
    """
    n = 50
    # w = 2.*pi/T
    wt = np.linspace( 0., 2.*pi, n) # phase
    f = sqrt( 1. - r**2 )
    numer = sin(wt) + ( r*sin(phi)/(1.+sqrt(1.-r**2)) )
    denom = (1.-r*cos(wt+phi))
    ut = Uw*f*numer/denom
    numer2 = cos(wt)-r*cos(phi)-r**2/(1.+sqrt(1.-r**2))*sin(phi)*sin(wt+phi)
    at = Uw*w*f*numer2/denom**2
    return wt, ut, at


# Input is Hs, T, h
Hs = 1. 
T = 10.
h = 4.

w = 2*pi/T
aw = 0.5*Hs
kh = qkhfs( w, h )
k = kh/h
Ur = ursell( aw, k, h) 
B = Bfit( Ur )
Phi = Phifit( Ur )
r = rfromB( B )
phi = -Phi-pi/2.
Su = B*cos(Phi)
Au = B*sin(Phi)
print "Hs, T, h: ",Hs, T, h
print "kh, k, Ur: ",kh, k, Ur
print "B, Phi: ",B, Phi
print "Su, Au:",Su,Au
print "r, phi: ",r, phi
wt,ut,at = abreu_ut( Uw, r, phi, w )
plot(wt/(2.*pi),ut/Uw)
plot(wt/(2.*pi),at/(Uw*w))
plt.xlabel('t/T')
plt.ylabel('u/Uw, a/(Uw $\omega$)')


# Check to see the Abreu time series works
# select r and phi to match curves in Figs 1 and 2
r = .75
phi = -pi/4.
Uw = 1.
T = 5.
n = 50
w = 2.*pi/T
wt = linspace( 0., 2.*pi, n) # phase
f = sqrt( 1. - r**2 )
numer = sin(wt) + ( r*sin(phi)/(1.+sqrt(1.-r**2)) )
denom = (1.-r*cos(wt+phi))
ut = Uw*f*numer/denom
numer2 = cos(wt)-r*cos(phi)-r**2/(1.+sqrt(1.-r**2))*sin(phi)*sin(wt+phi)
at = Uw*w*f*numer2/denom**2

plot(wt/(2.*pi),ut/Uw)
plot(wt/(2.*pi),at/(Uw*w))
plt.xlabel('t/T')
plt.ylabel('u/Uw, a/(Uw $\omega$)')


# <h3>Plot results of GPS base/rover test by Mark Silver</h3>
# Three laps around the block.
# 

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

fn = r"C:\\crs\\proj\\gps\\results.csv"
csv = np.genfromtxt (fn, delimiter=",")
x = csv[:,1]
y = csv[:,2]
z = csv[:,3]

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='darkgray', marker='.')

ax.set_xlabel('Easting (m?)')
ax.set_ylabel('Northing (m?)')
ax.set_zlabel('Z (m?)')

plt.show()
plt.savefig('C:\\crs\\proj\\gps\\results.png',transparent=True,bbox_inches='tight')


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
get_ipython().magic('matplotlib inline')

def qkhfs( w, h ):
    """
    Quick iterative calculation of kh in gravity-wave dispersion relationship
    kh = qkhfs(w, h )
    
    Input
        w - angular wave frequency = 2*pi/T where T = wave period [1/s]
        h - water depth [m]
    Returns
        kh - wavenumber * depth [m]

    Orbital velocities from kh are accurate to 3e-12 !

    RL Soulsby (2006) \"Simplified calculation of wave orbital velocities\"
    HR Wallingford Report TR 155, February 2006
    Eqns. 12a - 14
    """
    g = 9.81
    x = w**2.0 *h/g
    y = np.sqrt(x) * (x<1.) + x *(x>=1.)
    # This appalling bit of code is faster than a loop in Matlab and Fortran
    # but have not tested speed in Python.
    t = np.tanh( y )
    y = y-( (y*t -x)/(t+y*(1.0-t**2.0)))
    t = np.tanh( y )
    y = y-( (y*t -x)/(t+y*(1.0-t**2.0)))
    t = np.tanh( y )
    y = y-( (y*t -x)/(t+y*(1.0-t**2.0)))
    kh = y
    return kh

def ursell( aw, k, h ):
    """
    Calculate Ursell number
    Ur = ursell( aw, k, h)
        
    The Ursell number is a measure of asymmetry based on Stokes 2nd order
    wave equations and strictly speaking applies to shallow water (kh<1).
    It proportiona to the ratio of the 2nd order wave height term over
    the 1st order term.
    
    Input:
        aw = wave amplitude (H/2) [m/s]
        k  = wave number []
        h  = water depth [m]
        
    Returns:
        Ursell number
        
    Reussink et al. Eqn 6.
    """
    return (3./4.)*aw*k/(k*h)**3.

def urms_25( Hs, Tp, h ):
    """
    Calculate u_rms for JONSWAP spectr
    urms = urms_28( Hs, Tp, h )
    
    Equation 25 in Soulsby, R. L. (2006), "Simplified calculation of 
    wave orbital velocities". Report TR 155, Release 1.0, Feb. 2006.
    HR Wallingford.
    
    """
    Tz = Tp/1.28           # can't remember where I found this
    Tz = 0.777*Tp          # very similar: Tucker & Pitt, p. 103 
    Tn = ( h/g )**(1./2.)                      # Eqn 8
    t = Tn /Tz                                 # Eqn 27
    A = (6500.+(0.56 + 15.54*t)**6)**(1./6.)   # Eqn 26
    urms_25 = (0.25*Hs/Tn) /((1.+A*t**2)**3)   # Eqn 25
    return urms_25

g = 9.81
rhos = 2650.
rho = 1040.
es = 0.01
Cs = 0.01
B = 0.
ws= 0.033
n = 10
h = np.linspace(5,50,n)
H = 3.
T = 10.
w = 2.*np.pi/T
kh = qkhfs(w,h)
k = kh/h
L = np.pi*2./k
aw = H/2.
Ur = ursell(aw, k, h)
u0 = np.pi*H/(T*np.sinh(kh))
# the sqrt(2) factor converts from RMS to wave-orbital amplitude:
u0s = np.sqrt(2.)*urms_25(H, T, h)
u1 = 3.*np.pi**2*H**2/(4.*T*L*np.sinh(kh)**2)
u1s = u1 # for now
u2 = 3.*np.pi**2*H**2/(4.*T*L*np.sinh(kh)**4)
u2s = u2 # for now
K = 16*es*Cs*rho/(15.*np.pi*(rhos-rho)*g)
qs = K*(u0**3/ws)*(-5*u1-3*u2+B*u0**2/ws)
qss = K*(u0s**3/ws)*(-5*u1s-3*u2s+B*u0s**2/ws)
print qss/qs
# Equilibrium profile
B0 =   (ws/u0**2)*(5*u1+3*u2)
B0s = (ws/u0s**2)*(5*u1s+3*u2s)
if n<11:
    print 'h=',h
    print 'kh=',kh
    print 'Ur=',Ur
    print 'u0=',u0
    print 'u0s=',u0s
    print 'u1=',u1
    print 'u2=',u2
    print 'qs=',qs
    print 'B0=',B0
    print 'B0s=',B0s


# compare orbital velocities
fig = plt.figure()
plt.plot(h,u0,label = 'linear')
plt.plot(h,u0s, label = 'JONSWAP')
#plt.ylim(ymax=0., ymin=-50)
plt.ylabel('u0 (m/s)')
plt.xlabel('Depth (m)')
plt.xlim(xmax=50., xmin=0)
legend = plt.legend(loc='upper right', shadow=True)


# reproduce Fig. 1. This does not exactly match. Maybe needs different ws (in K)?
fig = plt.figure()
plt.plot(-np.log10(-qs),-h,label = 'qs linear')
plt.plot(-np.log10(-qss),-h, label = 'qs JONSWAP')
plt.ylim(ymax=0., ymin=-50)
plt.xlabel('log10 qs (m2/s)')
plt.ylabel('Depth (m)')
legend = plt.legend(loc='upper right', shadow=True)


# calculate geomorphic wave
T = np.arange(6.,16.,2.)
H = np.arange(1.,5.5, .5)
h = 50.
M = np.zeros((len(H),len(T)))
print np.shape(H),np.shape(T),np.shape(M)
for i in xrange(len(H)):
    for j in xrange(len(T)):
        w = 2.*np.pi/T[j]
        kh = qkhfs(w,h)
        k = kh/h
        L = np.pi*2./k
        #M[i,j]=(H[i]**3/((T[j]**3)*np.sinh(kh)**3))*( (-15.*np.pi**2*H[i]**2)/(4.*T[j]*L*np.sinh(kh)**2) + \
        M[i,j]= (H[i]**5)/((T[j]**3)*(np.sinh(kh)**3))
        
H, T = np.meshgrid(T, H)
fig = plt.figure()
plt.contour(H,T,np.log10(M))
#plt.contour(H,T,M)
plt.title('Contour Plot of H^5/(T^3 sinh^3(kh))')
plt.ylabel('H (m)')
plt.xlabel('T (s)')


# The equilibrium profiles are calculated using a simple integration...I think it is ok if enough points are included.
# The functions and variables modified with "s" are Sherwood modifications to the equations. Reducing the u1 term (u1s)
# by adding undertow has the counterintuitive result of flattening and even reversing the slope, because slope is the only term balancing the transport terms to acheive equilibrium.
# 

# calculate equilibrium profiles

def StokesQ(h, T, H):
    """
    Calculate Stokes volume transport
    Assume onshore transport, so cos(theta) term = 1
    """
    g = 9.81
    w = 2.*np.pi/T
    kh = qkhfs(w,h)
    k = kh/h
    L = np.pi*2./k
    c = L/T
    Qw = g*H**2/(16.*c)
    return Qw

def u1_Stokes(h, T, H):
    """
    Assume a log profile and calc u1 at top of WBL
    """
    Qw = StokesQ(h, T, H)
    z0 = 0.001
    ubar = Qw/h
    ustr = ubar*0.41/(np.log(0.6*h/z0))
    # estimate the top of the WBL. Results are sensitive to this and z0
    # this should really be calculated with knowledge of tauw and D50
    zw = 0.01*T/(2.*np.pi)
    u = (ustr/0.41)*np.log(zw/z0)
    return u

def B0_func(h, T, H, ws):
    """
    Calculate equilibrium slope
    """
    w = 2.*np.pi/T
    kh = qkhfs(w,h)
    k = kh/h
    L = np.pi*2./k
    u0 = np.pi*H/(T*np.sinh(kh))
    u1 = 3.*np.pi**2*H**2/(4.*T*L*np.sinh(kh)**2)
    u2 = 3.*np.pi**2*H**2/(4.*T*L*np.sinh(kh)**4)
    B0 = (ws/u0**2)*(5.*u1 + 3.*u2)
    return B0

def B0s_func(h, T, H, ws ):
    """
    Calculate equilibrium slope with diff. terms
    """
    w = 2.*np.pi/T
    kh = qkhfs(w,h)
    k = kh/h
    L = np.pi*2./k
    u0s = np.sqrt(2.)*urms_25(H, T, h)
    u1s = 3.*np.pi**2*H**2/(4.*T*L*np.sinh(kh)**2)
    u1S = u1_Stokes(h, T, H)
    #print "u1s, u1S=",u1s,u1S
    u2s = 3.*np.pi**2*H**2/(4.*T*L*np.sinh(kh)**4)
    B0s = (ws/u0s**2)*(5.*(u1s) + 3.*u2s)
    return B0s

dz = 2.
z = np.arange(1.,100.+dz,dz)
x = np.zeros_like(z)
xs = np.zeros_like(z)
x[0] = 0.
xs[0]= 0.
for i in xrange(1,len(z)):
    # trapezoidal rule
    x[i] = x[i-1] + (z[i]-z[i-1])*0.5*(1./B0_func(z[i-1],T,H,ws)+1./B0_func(z[i],T,H,ws))
    xs[i] = xs[i-1] + (z[i]-z[i-1])*0.5*(1./B0s_func(z[i-1],T,H,ws)+1./B0s_func(z[i],T,H,ws))
    #print "i=",i, z[i], x[i], xs[i]

    
fig = plt.figure()
plt.plot(x, -z,label ='linear')
plt.plot(xs,-z,label='JONSWAP')
plt.xlabel('Offshore distance (m')
plt.ylabel('Depth (m)')
legend = plt.legend(loc='upper right', shadow=True)


whos


# ## Calculations from EuroTop
# 
# http://www.overtopping-manual.com/eurotop.pdf
# 
# ### Some definitions
# Significant wave height `Hm0`: $H_{m0} = 4m_0^{1/2}$
# 
# Average period `Tm` $T_m$
# 
# Spectral period `Tmo`: $T_{m-1,0} = m_{-1}/m_0$
# 
# Peak period `Tp`: $T_p = 1.1T_{m-1,0}$
# 
# Wavelength `L0`: $L_0$
# 
# Deepwater wavelength `Lm0`: $L_{m-1,0} = gT_{m-1,0}^2/2\pi$  
# 
# Wave steepness `s0`: $s_0 = H_{m0}/L_0$
#   - $s_0 = 0.01$ indicates swell
#   - $s_0 = 0.04 to 0.6$ typical wind sea
#   
# Slope `alpha` and `tanalpha`: $=\tan(\alpha)$  
# 
# Iribarren number `em0` (breaker parameter, surf similarity number) $\xi_{m-1,0} =\tan(\alpha)/(H_{m0}/L_{m-1,0})^{1/2}$
#   - $0.2<\xi_{m-1,0}$ spilling waves
#   - $0.2 < \xi_{m-1,0} < 2-3$ plunging waves
#   - $\xi_{m-1,0} \approx 2-3$ collapsing waves
#   - $\xi_{m-1,0} > 2-3$ surging waves
# 

import numpy as np

def q_overtop_EOT():
    """Principal wave overtopping formula (4.1)"""
    g=9.86
    a=1.
    b=1.
    Rc=1.
    Hm0 = 1.
    qND=np.sqrt(g*Hm0**3)   # dimensionless  discharge
    RcHm0 = Rc/Hm0          # relative freeboard
    q = qND*a*np.exp(-b*RcHm0)
    return q

def runup_EOT(Hm0=1.,Tm0=8.,tanalpha=.2,yb=1.,yf=1.,yB=1.):
    g = 9.86
    c1 = 1.65
    c2 = 4.
    c3 = 1.5
    L0 = g*Tm0**2./(2*np.pi)
    em0 = tanalpha/np.sqrt(Hm0/L0)
    Ru2Hm0 = np.max(c1*yb*yf*yB*em0,yb*yf*yB*(c2-c3/np.sqrt(em0)))
    print Ru2Hm0
    return Ru2


print q_overtop_EOT()
print runup_EOT()





# ### Reading exif info.
# A more readable version of the exif for a .JPG file uses imagemagick:
#   ```$ identify -verbose DSC_4357.JPG```
#   
# 

import piexif
#jpg_name = "/home/csherwood/crs/proj/2015-12-09_survey/overwash_channel/DSC_4339.JPG"
jpg_name = "/home/csherwood/crs/proj/2016_CACO/test_images/IMG_0087.JPG"
exif_dict = piexif.load(jpg_name)
for ifd in ("0th", "Exif", "GPS", "1st"):
    print ifd
    for tag in exif_dict[ifd]:
        print(piexif.TAGS[ifd][tag]["name"], exif_dict[ifd][tag])


# munt the GPS data, just for fun...
print exif_dict['GPS']
for tag in exif_dict['GPS']:
        print(piexif.TAGS['GPS'][tag]["name"], exif_dict['GPS'][tag])


# ## Proj4 - These definitions seem to work, but produce no difference between wgs84 and nad83
# 

import mpl_toolkits.basemap.pyproj as pyproj # Import the pyproj module
# Define a projection with Proj4 notation, in this case an Icelandic grid
isn2004=pyproj.Proj("+proj=lcc +lat_1=64.25 +lat_2=65.75 +lat_0=65 +lon_0=-19 +x_0=1700000 +y_0=300000 +no_defs +a=6378137 +rf=298.257222101 +to_meter=1")
 # Define some common projections using EPSG codes
wgs84=pyproj.Proj("+init=EPSG:4326") # LatLon with WGS84 datum used by GPS units and Google Earth
wgs84b = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
nad83=pyproj.Proj("+init=EPSG:4269")
nad83b=pyproj.Proj("+proj=lcc +lat_1=42.68333333333333 +lat_2=41.71666666666667 +lat_0=41 +lon_0=-71.5 +x_0=200000 +y_0=750000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +preserve_units")
nad83c=pyproj.Proj("+proj=lcc +lat_1=42.68333333333333 +lat_2=41.71666666666667 +lat_0=41 +lon_0=-71.5 +x_0=200000 +y_0=750000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +to_meter=0 +no_defs +preserve_units")
pyproj.transform(wgs84,nad83c,-70.483290627,41.766497572)


# <h3>First-Order Transformation Equations</h3> This routine solves transformation equations (equivalent to radioactive decay equations). Analytical solutions were found by Bateman (1910), but the numerical solutions are more flexible (and arguably more elegant).
# 
# The coupled set of ordinary differential equations is:
# 
# $$\begin{matrix} \frac{d{{C}_{DDE}}}{dt}=-{{\lambda }_{DDE\to DDMU}}{{C}_{DDE}}-{{\lambda }_{Loss}}{{C}_{DDE}}\\frac{d{{C}_{DDMU}}}{dt}=-{{\lambda }_{DDMU\to DDNS}}{{C}_{DDMU}}+{{\lambda }_{DDE\to DDMU}}{{C}_{DDE}}-{{\lambda }_{Loss}}{{C}_{DDMU}}\\frac{d{{C}_{DDNS}}}{dt}=-{{\lambda}_{DDNS\to ?}}{{C}_{DDNS}}+{{\lambda}_{DDMU\to DDNS}}{{C}_{DDMU}}-{{\lambda }_{Loss}}{{C}_{DDNS}}\ \end{matrix}$$
# 
# where ${{\lambda}_{X\to Y}}$ is transformation rate from compound ${X}$ to compound ${Y}$. The ${{\lambda}_{Loss}}$ term represents physical removal of the compounds by, for example, resuspension and desorption. In this example, it is applied equally to all compounds, and can be set to zero to represent a closed system.
# 

from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML
from scipy.integrate import odeint
from pylab import *

global lam

def dcdt(c,t):
    dfdt = np.zeros(4)
    dfdt[0] = c[0]* -lam[0]               - c[0]*lam[3]
    dfdt[1] = c[1]* -lam[1] + c[0]*lam[0] - c[1]*lam[3] 
    dfdt[2] = c[2]* -lam[2] + c[1]*lam[1] - c[2]*lam[3]
    dfdt[3] =                 c[2]*lam[2] - c[3]*lam[3]
    return dfdt

def on_button_clicked(b):
    print("Calculate button clicked.")
    global lam
    lam = array([float(L0.value),float(L1.value),                 float(L2.value),float(L3.value)])
    C0 = array([.68, .23, .06, 0.])
    t = linspace(0.0,100.,50)
    C = odeint(dcdt,C0,t)

    fig = plt.figure(figsize=(6,5))
    plot(t+float(DS.value),C[:,0],label='DDE')
    plot(t+float(DS.value),C[:,1],label='DDMU')
    plot(t+float(DS.value),C[:,2],label='DDNS')
    plot(t+float(DS.value),C[:,3],label='?')
    plt.legend()
    plt.ylabel('Inventory')
    
DS = widgets.TextWidget(description = r'Start year',value='1992')
L0 = widgets.TextWidget(description = r'DDE  -> DDMU',value='0.052')
L1 = widgets.TextWidget(description = r'DDMU -> DDNS',value='0.07')
L2 = widgets.TextWidget(description = r'DDNS ->  ?  ',value='0.161')
L3 = widgets.TextWidget(description = r'DDX  -> lost',value='0.00')
B  = widgets.ButtonWidget(description = r'Calculate!')

display(DS,L0,L1,L2,L3,B)
B.on_click(on_button_clicked)





# <h2>Calculate air density from temperature, pressure, and relative humidity</h2>
# http://wahiduddin.net/calc/density_altitude.htm
# 

def pvs(T):
    # Calculate saturation water vapor pressure (hPa==millibar)
    # T = air temperature at dewpoint (degrees Celsius)
    # (per Herman Wobus according to several web sources)
    eso = 6.1078
    c0 = 0.99999683
    c1 = -0.90826951e-2
    c2 = 0.78736169e-4
    c3 = -0.61117958e-6
    c4 = 0.43884187e-8
    c5 = -0.29883885e-10
    c6 = 0.21874425e-12
    c7 = -0.17892321e-14
    c8 = 0.11112018e-16
    c9 = -0.30994571e-19
    p = c0 +T*(c1+T*(c2+T*(c3+T*(c4+T*(c5+T*(c6+T*(c7+T*(c8+T*(c9)))))))))
    return eso/(p**8)
def pv(T,p,RH):
    # Actual water vapor pressure for T (deg C), p (millibars) at relative humidity RH (%)
    return pvs(T)*(RH/100.)
def rho(T,p,RH):
    # Air density at temperature T (deg C), pressure p (millibars), and relative humidty RH (%)
    return (p*100.)/ (287.05*(T+273.15)) *(1.-(0.378*pv(T,p,RH))/(p*100.))
T = 20.
p = 1013.25
RH = 40.
print pvs(T)
print pv(T,p,RH)
print "Basement w/ dehumidifier: ",rho(17,p,40.)," kg/m3"
print "Upstairs when muggy:",rho(27.,p,80.)," kg/m3"
print "Dry air is denser than moist air!"


# Investigate uncertainty in point cloud estimates.
# 
# 1) Uncertainty reconstruction of the high res cloud from reconstructed geometry of the scene
# 2) Uncertainty in the tie points
# 3) Uncertainty in location of the GCPs relative to benchmark - based on stake errors
# 4) Uncertainty in location of the benchmark - based on OPUS report
# 

import numpy as np

# CACO 30 March 2016 results
    
# OPUS solution results, now based on the "precise" results
dzO = 0.015
dxdyO = np.sqrt( 0.002**2 + 0.005**2)

# Uncertainty in survey results - this is the RMS differences between reference markers and stake measurements
# for 5, 5, and 6 stakes on RM1, RM2, and RM3.
dzS = 0.007
dxdyS = np.sqrt(0.0144**2+0.0223**2)

dxdySO=np.sqrt(dxdyO**2+dxdyS**2)
dzSO=np.sqrt(dzO**2+dzS**2)

# Marker error from Photoscan
dzP = 0.0086
dxdyP = np.sqrt(0.0141**2+0.0122**2)

# Some kind of error estimate from Photoscan
# Pixel size on ground: 7.19 cm/pix in DEM, 5 cm/pix in ortho
# Reprojection error is 0.3 pix...assuming pix = 4 cm, then
rp_error = 0.3 # pix
px_size = 4.    # cm/pix
rp_error_m = rp_error*px_size/100.

print "OPUS solution = ",dxdyO, dzO
print "stake errors = ",dxdyS, dzS
print "combined survey errors = ",dxdySO,dzSO
print "marker error in Photoscan = ",dxdyP,dzP
print "sum of GCP undertainty = ",np.sqrt(dxdySO**2+dxdyP**2), np.sqrt(dzSO**2+dzP**2) 
print "rp_error_m = ",rp_error_m
print "sum of these = ",np.sqrt(dxdySO**2+dxdyP**2+rp_error_m**2), np.sqrt(dzSO**2+dzP**2+rp_error_m**2) 





# # Access data from the NECOFS (New England Coastal Ocean Forecast System) via OPeNDAP
# 

# Demonstration using the NetCDF4-Python library to access velocity data from a triangular grid ocean model (FVCOM) via OPeNDAP, specifying the desired URL, time, layer and lat/lon region of interest.  The resulting plot of forecast velocity vectors over color-shaded bathymetry is useful for a variety of recreational and scientific purposes. 
# 
# NECOFS (Northeastern Coastal Ocean Forecast System) is run by groups at the University of Massachusetts Dartmouth and the Woods Hole Oceanographic Institution, led by Drs. C. Chen, R. C. Beardsley, G. Cowles and B. Rothschild. Funding is provided to run the model by the NOAA-led Integrated Ocean Observing System and the State of Massachusetts.
# 
# NECOFS is a coupled numerical model that uses nested weather models, a coastal ocean circulation model, and a wave model. The ocean model is a volume-mesh model with horizontal resolution that is finer in complicated regions. It is layered (not depth-averaged) and includes the effects of tides, winds, and varying water densities caused by temperature and salinity changes.
# 
# * Model description: http://fvcom.smast.umassd.edu/research_projects/NECOFS/model_system.html
# * THREDDS server with other forecast and archive products: http://www.smast.umassd.edu:8080/thredds/catalog.html
# 

from pylab import *
get_ipython().magic('matplotlib inline')
import matplotlib.tri as Tri
import netCDF4
import datetime as dt


# DAP Data URL
# MassBay GRID for Boston Harbor, Cape Ann
url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc'
# GOM3 GRID (for Naragansett Bay, Woods Hole)
# url='http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc'
# Open DAP
nc = netCDF4.Dataset(url).variables
nc.keys()


# take a look at the "metadata" for the variable "u"
print nc['u']


shape(nc['temp'])


shape(nc['nv'])


# Desired time for snapshot
# ....right now (or some number of hours from now) ...
#start = dt.datetime.utcnow() + dt.timedelta(hours=18)
# ... or specific time (UTC)
start = dt.datetime(2015,7,25,13,0,0)+dt.timedelta(hours=0-5)


# Get desired time step  
time_var = nc['time']
itime = netCDF4.date2index(start,time_var,select='nearest')

# Get lon,lat coordinates for nodes (depth)
lat = nc['lat'][:]
lon = nc['lon'][:]
# Get lon,lat coordinates for cell centers (depth)
latc = nc['latc'][:]
lonc = nc['lonc'][:]
# Get Connectivity array
nv = nc['nv'][:].T - 1 
# Get depth
h = nc['h'][:]  # depth 


dtime = netCDF4.num2date(time_var[itime],time_var.units)
daystr = dtime.strftime('%Y-%b-%d %H:%M')
print daystr


tri = Tri.Triangulation(lon,lat, triangles=nv)


# get current at layer [0 = surface, -1 = bottom]
ilayer = 0
u = nc['u'][itime, ilayer, :]
v = nc['v'][itime, ilayer, :]


#woods hole
levels=arange(-30,2,1)
ax = [-70.7, -70.6, 41.48, 41.55]
maxvel = 1.0
subsample = 2


#boston harbor
levels=arange(-34,2,1)   # depth contours to plot
ax= [-70.97, -70.82, 42.25, 42.35] # 
maxvel = 0.5
subsample = 3


#entrance Narragansett Bay
levels=arange(-34,2,1)   # depth contours to plot
ax= [-71.48, -71.32, 41.42, 41.52] # 
maxvel = 0.5
subsample = 1


# Cape Ann
levels=arange(-34,2,1)   # depth contours to plot
ax= [-70.7, -70.58, 42.575, 42.695]
maxvel = 1.
subsample = 1


# find velocity points in bounding box
ind = argwhere((lonc >= ax[0]) & (lonc <= ax[1]) & (latc >= ax[2]) & (latc <= ax[3]))


np.random.shuffle(ind)
Nvec = int(len(ind) / subsample)
idv = ind[:Nvec]


# tricontourf plot of water depth with vectors on top
figure(figsize=(18,10))
subplot(111,aspect=(1.0/cos(mean(lat)*pi/180.0)))
tricontourf(tri, -h,levels=levels,shading='faceted',cmap=plt.cm.gist_earth)
axis(ax)
gca().patch.set_facecolor('0.5')
cbar=colorbar()
cbar.set_label('Water Depth (m)', rotation=-90)
Q = quiver(lonc[idv],latc[idv],2.24*u[idv],2.24*v[idv],scale=20)
maxstr='%3.1f mph' % maxvel
qk = quiverkey(Q,0.92,0.08,maxvel,maxstr,labelpos='W')
title('NECOFS Velocity, Layer %d, %s UTC' % (ilayer, daystr));
figname = r'NECOFS_Velocity'+daystr+'.png'
print figname
savefig('NECOFS_CapeAnn_example.png')


v


