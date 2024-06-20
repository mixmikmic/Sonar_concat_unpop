__depends__ = ['../outputs/llc_kuroshio_timeseries.nc']
__dest__ = ['../writeup/figs/fig2.pdf']


import datetime

import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from netCDF4 import Dataset


c1 = 'slateblue'
c2 = 'tomato'
c3 = 'k'
c4 = 'indigo'
plt.rcParams['lines.linewidth'] = 1.5
ap = .75

plt.style.use('seaborn-colorblind')

def leg_width(lg,fs):
    """"  Sets the linewidth of each legend object """
    for legobj in lg.legendHandles:
        legobj.set_linewidth(fs)
        


llc = Dataset(__depends__[0])


def parse_time(times):
    """
    
    Converts an array of strings that defines
    the LLC outputs into datatime arrays,
    e.g., '20110306T010000' --> datetime.datetime(2011, 3, 6, 1, 0)      
    
    Input
    ------
    times: array of strings that define LLC model time
    
    Output
    ------
    time: array of datetime associated with times
    
    """
    time = []
    for i in range(times.size):
        yr =  times[i][:4]
        mo =  times[i][4:6]
        day = times[i][6:8]
        hr =  times[i][9:11]
        time.append(datetime.datetime(int(yr),int(mo),int(day),int(hr)))  
    return np.array(time)


stats2 = np.load("../outputs/VelStats_anomaly_24h_4320_0_spectral.npz")
stats2_2160 = np.load("../outputs/VelStats_anomaly_24h_spectral.npz")


time2160 = parse_time(llc['2160']['hourly']['time'][:])
timed2160 = time2160[::24]

time4320 = parse_time(llc['4320']['hourly']['time'][:])
timed4320 = time4320[::24]

time43202 = parse_time(stats2['time'])
timed43202 = time43202[::24]

time21602 = parse_time(stats2_2160['time'])
timed21602 = time21602[::24]


timeaviso = []
for i in range(llc['aviso']['time'][:].size):
    timeaviso.append(datetime.datetime.strptime(llc['aviso']['time'][i],'%d%m%Y'))


# ## Figure 2: time-series of RMS vertical vorticity, RMS laretal rate of strain, and  RMS horizontal divergence
# 

fig = plt.figure(figsize=(12,9))

ax = fig.add_subplot(311)
plt.plot(time2160,llc['2160']['hourly/vorticity'][:],color='g',label=r'1/24$^\circ$, hourly') 
plt.plot(timed2160,llc['2160']['daily-averaged/vorticity'][:],'--',color='g',label='1/24$^\circ$, daily-averaged')
plt.plot(time4320,llc['4320']['hourly/vorticity'][:],color=c1,label=r'1/48$^\circ$, hourly')
plt.plot(timed4320,llc['4320']['daily-averaged/vorticity'][:],'--',color=c1,label='1/48$^\circ$, daily-averaged')
plt.plot(time2160,llc['2160']['smoothed100km/vorticity'][:],color='tomato',alpha=ap,label='1/24$^\circ$, 100-km-smoothed')
plt.plot(timeaviso,llc['aviso']['vorticity'][:],color='k',alpha=ap,label='AVISO gridded 1/4$^\circ$')

plt.text(timed2160[-90], .45, "Vorticity", size=20, rotation=0.,
         ha="center", va="center",
           bbox = dict(boxstyle="round",ec='k',fc='w'))
plt.xticks([])
plt.ylabel(r'RMS vorticity $\zeta/f$')
plt.text(timed2160[5], .555,'(a)',fontsize=18)
ax.xaxis.tick_top()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.yticks([ .1, .2 ,.3, .4, .5])
lg = ax.legend(loc=(-.015,-3.3), ncol=3, fancybox=True,frameon=True, shadow=True)
leg_width(lg,fs=6)


ax = fig.add_subplot(312)
plt.plot(time2160,llc['2160']['hourly/strain'][:],color='g',label=r'1/24$^\circ$, hourly') 
plt.plot(timed2160,llc['2160']['daily-averaged/strain'][:],'--',color='g',label='1/24$^\circ$, daily-averaged')
plt.plot(time4320,llc['4320']['hourly/strain'][:],color=c1,label=r'1/48$^\circ$, hourly')
plt.plot(timed4320,llc['4320']['daily-averaged/strain'][:],'--',color=c1,label='1/48$^\circ$, daily-averaged')
plt.plot(time2160,llc['2160']['smoothed100km/strain'][:],color='tomato',alpha=ap,label='1/24$^\circ$, 100-km-smoothed')
plt.plot(timeaviso,llc['aviso']['strain'][:],color='k',alpha=ap,label='AVISO')

plt.text(timed2160[-90], .45, "Strain", size=20, rotation=0.,
         ha="center", va="center",
           bbox = dict(boxstyle="round",ec='k',fc='w'))
plt.xticks([])
plt.ylabel(r'RMS strain $\alpha/f$')
plt.text(timed2160[5], .555,'(b)',fontsize=18)
ax.xaxis.tick_top()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks([ .1, .2 ,.3, .4, .5])


ax = fig.add_subplot(313)
plt.plot(time2160,llc['2160']['hourly/divergence'][:],color='g',label=r'1/24$^\circ$, hourly') 
plt.plot(timed2160,llc['2160']['daily-averaged/divergence'][:],'--',color='g',label='1/24$^\circ$, daily-averaged')
plt.plot(time4320,llc['4320']['hourly/divergence'][:],color=c1,label=r'1/48$^\circ$, hourly')
plt.plot(timed4320,llc['4320']['daily-averaged/divergence'][:],'--',color=c1,label='1/48$^\circ$, daily-averaged')
plt.plot(time2160,llc['2160']['smoothed100km/divergence'][:],color='tomato',alpha=ap,label='1/24$^\circ$, 100-km-smoothed')
plt.plot(timeaviso,llc['aviso']['divergence'][:],color='k',alpha=ap,label='AVISO')

plt.yticks([0.,.1,.2,.3])

plt.text(timed2160[-90], .25, "Divergence", size=20, rotation=0.,
         ha="center", va="center",
           bbox = dict(boxstyle="round",ec='k',fc='w'))
plt.ylabel(r'RMS divergence $\delta/f$')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.text(timed2160[5], .32,'(c)',fontsize=18)

plt.gcf().autofmt_xdate()

#plt.savefig(__dest__[0],bbox_inches='tight')


fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(211)
plt.plot(time2160,llc['2160']['hourly/vorticity'][:],color='g',label=r'1/24$^\circ$, hourly') 
plt.plot(timed2160,llc['2160']['daily-averaged/vorticity'][:],'--',color='g',label='1/24$^\circ$, daily-averaged')
plt.plot(time4320,llc['4320']['hourly/vorticity'][:],color=c1,label=r'1/48$^\circ$, hourly')
plt.plot(timed4320,llc['4320']['daily-averaged/vorticity'][:],'--',color=c1,label='1/48$^\circ$, daily-averaged')
plt.plot(time2160,llc['2160']['smoothed100km/vorticity'][:],color='tomato',alpha=ap,label='1/24$^\circ$, 100-km-smoothed')
plt.plot(timeaviso,llc['aviso']['vorticity'][:],color='k',alpha=ap,label='AVISO gridded 1/4$^\circ$')


plt.text(timed2160[-90], .45, "Vorticity", size=20, rotation=0.,
         ha="center", va="center",
           bbox = dict(boxstyle="round",ec='k',fc='w'))
plt.xticks([])
plt.ylabel(r'RMS vorticity $\zeta/f$')
#plt.text(timed2160[5], .555,'(a)',fontsize=18)
ax.xaxis.tick_top()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.yticks([ .1, .2 ,.3, .4, .5])
lg = ax.legend(loc=(-.015,-1.9), ncol=3, fancybox=True,frameon=True, shadow=True)
leg_width(lg,fs=6)


ax = fig.add_subplot(212)
plt.plot(time2160,llc['2160']['hourly/divergence'][:],color='g',label=r'1/24$^\circ$, hourly') 
plt.plot(timed2160,llc['2160']['daily-averaged/divergence'][:],'--',color='g',label='1/24$^\circ$, daily-averaged')
plt.plot(time4320,llc['4320']['hourly/divergence'][:],color=c1,label=r'1/48$^\circ$, hourly')
plt.plot(timed4320,llc['4320']['daily-averaged/divergence'][:],'--',color=c1,label='1/48$^\circ$, daily-averaged')
plt.plot(time2160,llc['2160']['smoothed100km/divergence'][:],color='tomato',alpha=ap,label='1/24$^\circ$, 100-km-smoothed')
plt.plot(timeaviso,llc['aviso']['divergence'][:],color='k',alpha=ap,label='AVISO')

plt.yticks([0.,.1,.2,.3])

plt.text(timed2160[-90], .25, "Divergence", size=20, rotation=0.,
         ha="center", va="center",
           bbox = dict(boxstyle="round",ec='k',fc='w'))
plt.ylabel(r'RMS divergence $\delta/f$')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#plt.text(timed2160[5], .32,'(c)',fontsize=18)

plt.gcf().autofmt_xdate()

plt.savefig("/Users/crocha/Desktop/fig2.png",bbox_inches='tight')


__depends__ = ['../outputs/llc_4320_kuroshio_pdfs.nc']
__dest__ = ['../writeup/figs/fig3.pdf']


import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from matplotlib.colors import LogNorm

from netCDF4 import Dataset


plt.rcParams['lines.linewidth'] = 2


pdfs  = Dataset(__depends__[0])


cpdf = np.logspace(-5,1.,7)


# ## Figure 3: Joint probability density distributions of vorticity vs. strain rate and vorticity vs. laplacian of SSHa
# 

fig = plt.figure(figsize=(12,4))


ax = fig.add_subplot(241)
plt.contourf(pdfs['vorticity'][:],pdfs['strain'][:],pdfs['april/hourly']['pdf_vorticity_strain'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][100:],pdfs['vorticity'][100:],'k--')
plt.plot(pdfs['vorticity'][:100],-pdfs['vorticity'][:100],'k--')
plt.xlim(-4.,4.)
plt.ylim(0,4.)
plt.xticks([])
plt.yticks([])
plt.ylim(0.,4.)
plt.ylabel(r'Strain $\alpha/f$')
ticks=[0,2,4]
plt.yticks(ticks)
plt.text(-4.,4.15,'(a)',fontsize=14)
plt.title('Hourly',fontsize=11)

ax = fig.add_subplot(242)
plt.contourf(pdfs['vorticity'][:],pdfs['strain'][:],pdfs['april/daily-averaged']['pdf_vorticity_strain'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][100:],pdfs['vorticity'][100:],'k--')
plt.plot(pdfs['vorticity'][:100],-pdfs['vorticity'][:100],'k--')
plt.xlim(-4.,4.)
plt.ylim(0,4.)
plt.xticks([])
plt.yticks([])
plt.text(-4.,4.15,'(b)',fontsize=14)
plt.title('Daily-averaged',fontsize=11)

plt.text(-6,5.5,'April',fontsize=14)

ax = fig.add_subplot(2,4,3)
plt.contourf(pdfs['vorticity'][:],pdfs['strain'][:],pdfs['october/hourly']['pdf_vorticity_strain'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][100:],pdfs['vorticity'][100:],'k--')
plt.plot(pdfs['vorticity'][:100],-pdfs['vorticity'][:100],'k--')
plt.xlim(-4.,4.)
plt.ylim(0.,4.)
plt.yticks([])
xticks=[-4,-2,0,2,4]
plt.xticks([])
plt.text(-4.,4.15,'(c)',fontsize=14)
plt.title('Hourly',fontsize=11)

ax = fig.add_subplot(2,4,4)
plt.contourf(pdfs['vorticity'][:],pdfs['strain'][:],pdfs['october/daily-averaged']['pdf_vorticity_strain'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][100:],pdfs['vorticity'][100:],'k--')
plt.plot(pdfs['vorticity'][:100],-pdfs['vorticity'][:100],'k--')
plt.xlim(-4.,4.)
plt.ylim(0.,4.)
plt.xticks([])
plt.yticks([])
plt.text(-4.,4.15,'(d)',fontsize=14)
plt.title('Daily-averaged',fontsize=11)
plt.text(-6.75,5.5,'October',fontsize=14)

ax = fig.add_subplot(2,4,5)
plt.contourf(pdfs['vorticity'][:],pdfs['vorticity'][:],pdfs['april/hourly']['pdf_vorticity_lapssh'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][:],pdfs['vorticity'][:],'k--')
plt.xlim(-4.,4.)
plt.ylim(-4.,4.)
plt.ylabel(r'$(g/f^2) \, \nabla^2 \eta$')
ticks=[-4,-2,0,2,4]
plt.xticks(ticks)
plt.yticks(ticks)
plt.text(-4.,4.15,'(e)',fontsize=14)
#plt.xlabel(r'Vorticity $\zeta/f$')

ax = fig.add_subplot(2,4,6)
plt.contourf(pdfs['vorticity'][:],pdfs['vorticity'][:],pdfs['april/daily-averaged']['pdf_vorticity_lapssh'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][:],pdfs['vorticity'][:],'k--')
plt.xlim(-4.,4.)
plt.ylim(-4.,4.)
ticks=[-4,-2,0,2,4]
plt.xticks(ticks)
plt.yticks([])
plt.text(-4.,4.15,'(f)',fontsize=14)
#plt.xlabel(r'Vorticity $\zeta/f$')

ax = fig.add_subplot(2,4,7)
plt.contourf(pdfs['vorticity'][:],pdfs['vorticity'][:],pdfs['october/hourly']['pdf_vorticity_lapssh'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][:],pdfs['vorticity'][:],'k--')
plt.xlim(-4.,4.)
plt.ylim(-4.,4.)
plt.xticks(ticks)
ticks=[-4,-2,0,2,4]
plt.yticks([])
plt.text(-4.,4.15,'(g)',fontsize=14)
#plt.xlabel(r'Vorticity $\zeta/f$')

ax = fig.add_subplot(2,4,8)
cbs = plt.contourf(pdfs['vorticity'][:],pdfs['vorticity'][:],pdfs['october/daily-averaged']['pdf_vorticity_lapssh'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][:],pdfs['vorticity'][:],'k--')
plt.xlim(-4.,4.)
plt.ylim(-4.,4.)
plt.xticks(ticks)
plt.yticks([])
plt.text(-4.,4.15,'(h)',fontsize=14)
#plt.xlabel(r'Vorticity $\zeta/f$')


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.16, 0.01, 0.7])
fig.colorbar(cbs, cax=cbar_ax,label=r'Probability density',extend='both',ticks=[1.e-5,1e-4,1.e-3,1e-2,1.e-1,1,10.])

plt.savefig(__dest__[0],dpi=150,bbox_inches='tight')








__depends__ = ['../outputs/llc_kuroshio_spectra.nc']
__dest__ = ['../writeup/figs/fig5.pdf','../writeup/figs/S1.pdf']


import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from netCDF4 import Dataset


c1 = 'slateblue'
c2 = 'tomato'
c3 = 'k'
c4 = 'indigo'
plt.rcParams['lines.linewidth'] = 2.5

def leg_width(lg,fs):
    """"  Sets the linewidth of each legend object """
    for legobj in lg.legendHandles:
        legobj.set_linewidth(fs)
        
def add_second_axis(ax1):
    """ Add a x-axis at the top of the spectra figures """
    ax2 = ax1.twiny() 
    ax2.set_xscale('log')
    ax2.set_xlim(ax1.axis()[0], ax1.axis()[1])
    kp = 1./np.array([500.,250.,100.,50.,25.,10.,5.])
    lp=np.array([500,250,100,50,25,10,5])
    ax2.set_xticks(kp)
    ax2.set_xticklabels(lp)
    plt.xlabel('Wavelength [km]')
    
def set_axes(type='ke'):
    if type=='ke':
        plt.loglog(kr,12.*e2,'.5',linewidth=2)
        plt.loglog(kr,35*e3,'.5',linewidth=2)
        plt.xlim(.75e-3,1/3.)
        plt.ylim(1.e-3,1.e2)
        plt.ylabel(r'KE density [m$^2$ s$^{-2}$/cpkm]')

    elif type=='ssha':
        plt.loglog(kr,e2/.5e1,'.5',linewidth=2)
        plt.loglog(kr,3*e5/1.5e2,'.5',linewidth=2)
        plt.xlim(.75e-3,1/3.)
        plt.ylim(1.e-6,1.e2)
        plt.ylabel(r'SSH variance density [m$^2$/cpkm]')   
    
    plt.xlabel(r'Wavenumber [cpkm]')


llc = Dataset(__depends__[0])


kr = np.array([1.e-4,1.])
e2 = kr**-2/1.e4
e3 = kr**-3/1.e7
e5 = kr**-5/1.e9


# ## Figure 4: LLC4320 surface KE and SSHa variance wavenumber spectra
# 

fig = plt.figure(figsize=(13,4.))

ax = fig.add_subplot(121)

plt.loglog(llc['4320/wavenumber'],llc['4320/hourly/april']['E'],color='g',label='April, hourly')
plt.loglog(llc['4320/wavenumber'],llc['4320/daily-averaged/april']['E'],'--',color='g',label='April, daily-averaged')
plt.loglog(llc['4320/wavenumber'],llc['4320/hourly/october']['E'],color=c1,label='October, hourly')
plt.loglog(llc['4320/wavenumber'],llc['4320/daily-averaged/october']['E'],'--',color=c1,label='October, daily-averaged')
plt.loglog(kr,12.*e2,'.5',linewidth=2)
plt.loglog(kr,35*e3,'.5',linewidth=2)

plt.fill_between(llc['CIwavenumber'],llc['E_lower'],llc['E_upper'], color='.25', alpha=0.25)

plt.loglog(kr,12.*e2,'.5',linewidth=2)
plt.loglog(kr,35*e3,'.5',linewidth=2)
plt.xlim(.75e-3,1/3.)
plt.ylim(1.e-3,1.e2)
plt.xlabel(r'Wavenumber [cpkm]')
plt.ylabel(r'KE density [m$^2$ s$^{-2}$/cpkm]')
 
plt.text(8.e-2, 30, "Surface KE", size=16, rotation=0.,
         ha="center", va="center",
         bbox = dict(boxstyle="round",ec='k',fc='w'))
plt.text(.65e-3,400.,'(a)',fontsize=14)
add_second_axis(ax)

plt.ylim(1.e-4,1.e2)
lg = ax.legend(loc=(-0.095,-.4), ncol=4, fancybox=True,frameon=True, shadow=True)
leg_width(lg,fs=6)

plt.xlim(.75e-3,1/3.)
plt.text(1.e-3,.65e-3,'95%',fontsize=14)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.45, hspace=None)

ax = fig.add_subplot(122)


plt.loglog(llc['4320/wavenumber'],llc['4320/hourly/april']['ESSHa'],color='g',label='April, hourly')
plt.loglog(llc['4320/wavenumber'],llc['4320/daily-averaged/april']['ESSHa'],'--',color='g',label='April, daily-averaged')
plt.loglog(llc['4320/wavenumber'],llc['4320/hourly/october']['ESSHa'],color=c1,label='October, hourly')
plt.loglog(llc['4320/wavenumber'],llc['4320/daily-averaged/october']['ESSHa'],'--',color=c1,label='October, daily-averaged')


plt.fill_between(llc['CIwavenumber'],llc['ESSHa_lower'],llc['ESSHa_upper'], color='.25', alpha=0.25)


plt.text(1./2.8,1.5e-4,'-2',fontsize=14)
plt.text(1./5.5,.3e-7,'-5',fontsize=14)

plt.loglog(kr,e2/.5e1,'.5',linewidth=2)
plt.loglog(kr,3*e5/1.5e2,'.5',linewidth=2)
plt.xlim(.75e-3,1/3.)
plt.ylim(1.e-6,1.e2)
plt.xlabel(r'Wavenumber [cpkm]')
plt.ylabel(r'SSH variance density [m$^2$/cpkm]')
plt.text(3.e-2, 14, "SSH variance", size=16, rotation=0.,
         ha="center", va="center",
         bbox = dict(boxstyle="round",ec='k',fc='w'))
plt.ylim(1.e-7,1.e2)
plt.xlim(.75e-3,1/3.)
plt.text(.65e-3,700.,'(b)',fontsize=14)
plt.text(1.e-3,.5e-6,'95%',fontsize=14)
plt.yticks([1.e-6,1.e-4,1.e-2,1.e0,1e2])
add_second_axis(ax)

plt.savefig(__dest__[0],bbox_inches='tight')


# ## Figure S1: A comparison between LLC2160 and LLC4320 spectra
# 

fig = plt.figure(figsize=(13,10))

ax1 = fig.add_subplot(221)

plt.loglog(llc['2160/wavenumber'],llc['2160/hourly/april']['E'],color='g',label=r'1/24$^\circ$, hourly')
plt.loglog(llc['2160/wavenumber'],llc['2160/daily-averaged/april']['E'],'--',color='g',label=r'1/24$^\circ$, daily-averaged')
plt.loglog(llc['4320/wavenumber'],llc['4320/hourly/april']['E'],color=c1,label=r'1/48$^\circ$, hourly')
plt.loglog(llc['4320/wavenumber'],llc['4320/daily-averaged/april']['E'],'--',color=c1,label=r'1/48$^\circ$, daily-averaged')
set_axes(type='ke')
plt.text(4.5e-2, 30, "Surface KE, April", size=16, rotation=0.,
         ha="center", va="center",
         bbox = dict(boxstyle="round",ec='k',fc='w'))
plt.xticks([])
plt.xlabel('')

plt.fill_between(llc['CIwavenumber'],llc['E_lower'],llc['E_upper'], color='.25', alpha=0.25)
plt.text(1.e-3,.5e-6,'95%',fontsize=14)
plt.text(1./2.8,.09e-1,'-2',fontsize=14)
plt.text(1./2.8,.09e-3,'-3',fontsize=14)

plt.ylim(1.e-4,1.e2)

lg = ax1.legend(loc=(-0.05,-1.6), ncol=4, fancybox=True,frameon=True, shadow=True)
leg_width(lg,fs=6)

plt.xlim(.75e-3,1/3.)
plt.text(1.e-3,.65e-3,'95%',fontsize=14)

add_second_axis(ax1)
plt.text(.84e-3,200.,'(a)',fontsize=17)

ax2 = fig.add_subplot(222)
plt.loglog(llc['2160/wavenumber'],llc['2160/hourly/october']['E'],color='g',label=r'1/24$^\circ$, hourly')
plt.loglog(llc['2160/wavenumber'],llc['2160/daily-averaged/october']['E'],'--',color='g',label=r'1/24$^\circ$, daily-averaged')
plt.loglog(llc['4320/wavenumber'],llc['4320/hourly/october']['E'],color=c1,label=r'1/48$^\circ$, hourly')
plt.loglog(llc['4320/wavenumber'],llc['4320/daily-averaged/october']['E'],'--',color=c1,label=r'1/48$^\circ$, daily-averaged')
set_axes(type='ke')
plt.text(4.e-2, 30, "Surface KE, October", size=16, rotation=0.,
         ha="center", va="center",
         bbox = dict(boxstyle="round",ec='k',fc='w'))
plt.text(.84e-3,200.,'(b)',fontsize=17)

plt.xticks([])
plt.xlabel('')
plt.yticks([])
plt.ylabel('')

plt.fill_between(llc['CIwavenumber'],llc['E_lower'],llc['E_upper'], color='.25', alpha=0.25)
plt.text(1.e-3,.5e-6,'95%',fontsize=14)
plt.text(1./2.8,.09e-1,'-2',fontsize=14)
plt.text(1./2.8,.09e-3,'-3',fontsize=14)

plt.ylim(1.e-4,1.e2)

plt.xlim(.75e-3,1/3.)
plt.text(1.e-3,.65e-3,'95%',fontsize=14)


add_second_axis(ax2)

ax1 = fig.add_subplot(223)
plt.loglog(llc['2160/wavenumber'],llc['2160/hourly/april']['ESSHa'],color='g',label=r'1/24$^\circ$, hourly')
plt.loglog(llc['2160/wavenumber'],llc['2160/daily-averaged/april']['ESSHa'],'--',color='g',label=r'1/24$^\circ$, daily-averaged')
plt.loglog(llc['4320/wavenumber'],llc['4320/hourly/april']['ESSHa'],color=c1,label=r'1/48$^\circ$, hourly')
plt.loglog(llc['4320/wavenumber'],llc['4320/daily-averaged/april']['ESSHa'],'--',color=c1,label=r'1/48$^\circ$, daily-averaged')
set_axes(type='ssha')
plt.text(4.e-2, 15, "SSH variance, April", size=16, rotation=0.,
         ha="center", va="center",
         bbox = dict(boxstyle="round",ec='k',fc='w'))
plt.text(.84e-3,200.,'(c)',fontsize=17)

plt.fill_between(llc['CIwavenumber'],llc['ESSHa_lower'],llc['ESSHa_upper'], color='.25', alpha=0.25)

plt.text(1.e-3,.5e-6,'95%',fontsize=14)
plt.text(1./2.8,1.5e-4,'-2',fontsize=14)
plt.text(1./5.5,.3e-7,'-5',fontsize=14)
plt.ylim(1.e-7,1.e2)
plt.xlim(.75e-3,1/3.)
plt.yticks([1.e-6,1.e-4,1.e-2,1.e0,1e2])

ax1 = fig.add_subplot(224)
plt.loglog(llc['2160/wavenumber'],llc['2160/hourly/october']['ESSHa'],color='g',label=r'1/24$^\circ$, hourly')
plt.loglog(llc['2160/wavenumber'],llc['2160/daily-averaged/october']['ESSHa'],'--',color='g',label=r'1/24$^\circ$, daily-averaged')
plt.loglog(llc['4320/wavenumber'],llc['4320/hourly/october']['ESSHa'],color=c1,label='October, hourly')
plt.loglog(llc['4320/wavenumber'],llc['4320/daily-averaged/october']['ESSHa'],'--',color=c1,label='October, daily-averaged')

set_axes(type='ssha')

plt.fill_between(llc['CIwavenumber'],llc['ESSHa_lower'],llc['ESSHa_upper'], color='.25', alpha=0.25)
plt.text(1.e-3,.5e-6,'95%',fontsize=14)
plt.text(1./2.8,1.5e-4,'-2',fontsize=14)
plt.text(1./5.5,.3e-7,'-5',fontsize=14)

plt.text(3.e-2, 15, "SSH variance, October", size=16, rotation=0.,
         ha="center", va="center",
         bbox = dict(boxstyle="round",ec='k',fc='w'))
plt.text(.84e-3,200.,'(d)',fontsize=17)

plt.yticks([])
plt.ylabel('')
plt.ylim(1.e-7,1.e2)
plt.xlim(.75e-3,1/3.)

plt.savefig(__dest__[1],bbox_inches='tight')





__depends__ = ['../data/aviso/mdt_cnes_cls2013_global.nc','../data/llc/2160/grid/grid2160.npz',
              '../data/llc/4320/Snapshot_4320_april.npz','../data/llc/4320/Snapshot_4320_october.npz']
__dest__ = ['../writeup/figs/fig1_1.pdf','../writeup/figs/fig1_2.pdf']


import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import cmocean

from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset


# ## Figure 1a: Kuroshio Extension AVISO mean dynamic topography and ETOPO5 bottom topography on a LCC projection 
# 

# define the Kuroshio Extension map class
m = Basemap(width=12000000/2,height=9000000/3.,projection='lcc',
            resolution=None,lat_1=25.,lat_2=30,lat_0=33,lon_0=155.)


mdt = Dataset(__depends__[0])


lona,lata = np.meshgrid(mdt['lon'][:], mdt['lat'][:])
xaviso,yaviso = m(lona,lata)
adt = mdt['mdt'][:][0]
adt.min(),adt.max()
cadt = np.arange(-1.5,1.5,.1)


# the grid that defines the Kuroshio Extension subregion
grid = np.load(__depends__[1])
xi,yi = m(grid['lon'], grid['lat'])
x = np.array([xi[0,0],xi[-1,0],xi[-1,-1],xi[0,-1],xi[0,0]])
y = np.array([yi[0,0],yi[-1,0],yi[-1,-1],yi[0,-1],yi[0,0]])


fig = plt.figure(figsize = (10,4))
m.etopo()
m.drawparallels(np.arange(10.,50.,10.),labels=[1,0,0,0],fontsize=14)
m.drawmeridians(np.arange(-140.,181.,10.),labels=[0,0,0,1],fontsize=14)
m.contour(xaviso,yaviso,adt,cadt,colors='w',linewidths=1.)
m.plot(x,y,'.5',linewidth=4)
xt,yt = m(119,43)
plt.text(xt,yt,'(a)',fontsize=15)
plt.savefig(__dest__[0])


# ## Figures 1b-e: horizontal maps of vertical vorticity and potential density sections in different seasons
# 

m2 = Basemap(projection='merc',llcrnrlat=25.,urcrnrlat=40.,            llcrnrlon=155.,urcrnrlon=175.,lat_ts=30.,resolution='c')


snap_april = np.load(__depends__[2])
snap_october = np.load(__depends__[3])
xs,ys = m2(snap_april['lon'], snap_october['lat'])


fig = plt.figure(figsize = (12,9))


dec = 2 # use every other data point for vorticity map (file is huge otherwise...)

plt.subplot(223)
ctv = np.linspace(-1.,1.,25)
m2.contourf(xs[::dec,::dec],ys[::dec,::dec],snap_april['vort'][::dec,::dec],ctv,vmin=-1.,vmax=1.,extend='both',cmap=cmocean.cm.balance)
m2.drawcoastlines()
m2.fillcontinents(color='0.5',lake_color=None)
m2.drawparallels(np.arange(-90.,91.,5.),labels=[1,0,0,0],fontsize=14)
m2.drawmeridians(np.arange(-180.,181.,5.),labels=[0,0,0,1],fontsize=14)
xt,yt = m2(155,40.5)
plt.text(xt,yt,'(d)',fontsize=15)
xt,yt = m2(166,40.5)
plt.text(xt,yt,'April 15, 2011',fontsize=15)


plt.subplot(224)
ctv = np.linspace(-1,1.,25)
cs = m2.contourf(xs[::dec,::dec],ys[::dec,::dec],snap_october['vort'][::dec,::dec],
                 ctv,vmin=-1.,vmax=1.,extend='both',cmap=cmocean.cm.balance)
m2.drawcoastlines()
m2.fillcontinents(color='0.5',lake_color=None)
m2.drawparallels(np.arange(-90.,91.,5.),labels=[0,1,0,0],fontsize=14)
m2.drawmeridians(np.arange(-180.,181.,5.),labels=[0,0,0,1],fontsize=14)
xt,yt = m2(155,40.5)
plt.text(xt,yt,'(e)',fontsize=15)
xt,yt = m2(164.,40.5)
plt.text(xt,yt,'October 15, 2011',fontsize=15)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.14, 0.02, 0.3])
fig.colorbar(cs, cax=cbar_ax,label=r'Relative vorticity $\zeta/f$')

plt.subplots_adjust(left=None, bottom=None, right=None,
                    top=None, hspace=.35, wspace=None)

cs = np.array([22.,22.5,23.,23.5,24.,24.5,25.,25.5,26.,26.5])

plt.subplot(221)
plt.contourf(snap_april['latd'],-snap_april['z'][:],
             snap_april['dens']-1000,cs,vmin=22.,
             vmax=26.5,cmap=cmocean.cm.dense,extend='both')
plt.contour(snap_april['latd'],-snap_april['z'][:],
            snap_april['dens']-1000,cs,colors='k')

plt.plot(snap_april['latd'],snap_april['mld'],linewidth=2,color='w')

plt.ylim(450,0)
plt.ylabel('Depth [m]')
#plt.title('Early Spring')
xticks = [25,30,35,40]
xticklabels = [r'25$\!^\circ$N',r'30$\!^\circ$N',r'35$\!^\circ$N',r'40$\!^\circ$N']
plt.xticks(xticks,xticklabels)

plt.text(25,-10,'(b)',fontsize=15)
plt.text(33.5,-20,'April 15, 2011',fontsize=15)

plt.subplot(222)
cps = plt.contourf(snap_october['latd'],-snap_october['z'][:],
                   snap_october['dens']-1000,cs,vmin=22.,
                   vmax=26.5,cmap=cmocean.cm.dense,extend='both')
plt.contour(snap_october['latd'],-snap_october['z'][:],
            snap_october['dens']-1000,cs,colors='k')

plt.plot(snap_october['latd'],snap_october['mld'],linewidth=2,color='w')

plt.ylim(450,0)
plt.yticks([])
plt.xticks(xticks,xticklabels)

plt.text(25,-10,'(c)',fontsize=15)
plt.text(32.5,-20,'October 15, 2011',fontsize=15)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.59, 0.02, 0.3])
fig.colorbar(cps, cax=cbar_ax,label=r'Potential density, $\sigma_0$ [kg m$^{-3}]$')

plt.savefig(__dest__[1],dpi=80)





__depends__ = []
__dest__ = []


# #  Processing/plotting code
# 

# ### Processing
#  - [LLC velocity gradient and spectra](./LLCProcessing.ipynb)
# 
# ### Figures
#  - [Figure 1](./Figure1.ipynb)
#  - [Figure 2](./Figure2.ipynb)
#  - [Figure 3](./Figure3.ipynb)
#  - [Figure 4+](./Figure4.ipynb)
#  
# Figure files are available [here](../writeup/figs/).
# 




