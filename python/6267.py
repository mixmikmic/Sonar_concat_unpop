get_ipython().magic('matplotlib inline')


from grid.gmt import GMTGrid
from grid.shake import ShakeGrid
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path
import matplotlib


# The Grid class hierarchy can be used for reading, writing and manipulating various kinds 
# of 2D grid formats (GMT grids), or multi-layer 2D grid formats (ShakeMap).
# 

#######MODIFY THIS TO REFLECT THE LOCATION OF A GLOBAL GRID ON YOUR SYSTEM####
popfile = os.path.join(os.path.expanduser('~'),'pager','data','lspop2012.grd')
vs30file = os.path.join(os.path.expanduser('~'),'secondary','data','global_vs30.grd')
shakefile = os.path.join(os.path.expanduser('~'),'data','shakemaps','nepal2011.xml')
##############################################################################


# The grid classes can be used to get spatial information about the file before opening it. 
# 

fgeodict = GMTGrid.getFileGeoDict(vs30file)
bounds = (fgeodict['xmin'],fgeodict['xmax'],fgeodict['ymin'],fgeodict['ymax'])
print 'The file spans from %.3f to %.3f in longitude, and %.3f to %.3f in latitude.' % bounds


#41.798959,-125.178223 ulcorner of calif
#32.543919,-112.950439 lrcorner of calif
sdict = {'xmin':-125.178223,'xmax':-112.950439,'ymin':32.543919,'ymax':41.798959,'xdim':0.008,'ydim':0.008}
grid = GMTGrid.load(popfile,samplegeodict=sdict)
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(16,8));
data = grid.getData().copy()
data[data == -9999] = np.nan
data = np.ma.array (data, mask=np.isnan(data))
cmap = matplotlib.cm.bone_r
cmap.set_bad('b',1.)
im = plt.imshow(data,cmap=cmap,vmin=0,vmax=10000)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05);
plt.colorbar(im, cax=cax);
ax.set_title('Population Grid of California');


# Let's test reading across the 180/-180 meridian
# 

#LL: -18.606228,175.891113
#UR: -14.990556,-178.439941
sdict = {'xmin':175.891113,'xmax':-178.439941,'ymin':-18.606228,'ymax':-14.990556}
grid = GMTGrid.load(popfile,samplegeodict=sdict)
xmin,xmax,ymin,ymax = grid.getBounds()
print xmin,xmax,ymin,ymax
m1y,m1x = grid.getRowCol(ymin,180.0)
m2y,m2x = grid.getRowCol(ymax,180.0)
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(16,8));
data = grid.getData().copy()
data[data == -9999] = np.nan
data = np.ma.array (data, mask=np.isnan(data))
cmap = matplotlib.cm.bone_r
cmap.set_bad('b',1.)
im = plt.imshow(data,cmap=cmap,vmin=0,vmax=10000)
axlim = plt.axis()
plt.hold(True)
plt.plot([m1x,m2x],[m1y,m2y],'r')
plt.axis(axlim)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05);
plt.colorbar(im, cax=cax);
ax.set_title('Population Grid around Fiji');


# Let's use a ShakeMap grid to cut out a piece of the population grid, then resample the ShakeMap to the population data.
# 

# grid._data = np.array([])
# del grid
fdict = ShakeGrid.getFileGeoDict(shakefile) #get the bounds of the ShakeMap
newdict = GMTGrid.getBoundsWithin(popfile,fdict) #get some bounds guaranteed to be inside the ShakeMap
shake = ShakeGrid.load(shakefile) #load the ShakeMap
popgrid = GMTGrid.load(popfile,samplegeodict=newdict) #load without any resampling or anything
xmin,xmax,ymin,ymax = popgrid.getBounds()
shake.interpolateToGrid(popgrid.getGeoDict())
popdict = popgrid.getGeoDict()
shakedict = shake.getGeoDict()
popdata = popgrid.getData().copy()
mmidata = shake.getLayer('mmi').getData()
popdata[popdata == -9999] = np.nan
popdata = np.ma.array(popdata, mask=np.isnan(popdata))
fig,axeslist = plt.subplots(nrows=1,ncols=2,figsize=(20,16))


#plot population
m1y,m1x = popgrid.getRowCol(ymin,180.0)
m2y,m2x = popgrid.getRowCol(ymax,180.0)
plt.sca(axeslist[0])
cmap = matplotlib.cm.bone_r
im = plt.imshow(popdata,cmap=cmap,vmin=0,vmax=10000)
axlim = plt.axis()
plt.hold(True)
plt.plot([m1x,m2x],[m1y,m2y],'r')
plt.axis(axlim)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(axeslist[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05);
plt.colorbar(im, cax=cax1);
axeslist[0].set_title('Population Grid around Nepal');

#plot pga
plt.sca(axeslist[1]);
cmap = matplotlib.cm.OrRd
im = plt.imshow(mmidata,cmap=cmap)
axlim = plt.axis()
plt.hold(True)
plt.plot([m1x,m2x],[m1y,m2y],'r')
plt.axis(axlim)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(axeslist[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05);
plt.colorbar(im, cax=cax2);
axeslist[1].set_title('MMI Grid around Nepal');


# Implement the PAGER exposure algorithm.
# 

if np.any(mmidata > 9):
    mmidata[mmidata > 9.0] = 9.0
expolist = np.zeros((5))
for mmi in range(5,10):
    immi = (mmidata > mmi-0.5) & (mmidata <= mmi+0.5)
    expolist[mmi-5] = np.sum(popdata[immi])
print expolist


get_ipython().magic('matplotlib inline')


import matplotlib.pyplot as plt
import numpy as np
from grid.shake import ShakeGrid
from mpl_toolkits.basemap import Basemap
from collections import OrderedDict
from datetime import datetime
import os.path


# The Grid class hierarchy can be used for reading, writing and manipulating various kinds of 2D grid formats (GMT grids), or multi-layer 2D grid formats (ShakeMap).
# 

#######MODIFY THIS TO REFLECT THE LOCATION OF A GLOBAL GRID ON YOUR SYSTEM####
shakefile = '/Users/mhearne/data/shakemaps/northridge.xml'
##############################################################################


def map2DGrid(ax,grid,tstr,isLeft=False):
    """
    grid is a Grid2D object 
    """
    xmin,xmax,ymin,ymax = grid.getBounds()
    pdata = grid.getData()
    nr,nc = pdata.shape
    lonrange = np.linspace(xmin,xmax,num=nc)
    latrange = np.linspace(ymin,ymax,num=nr)
    lon,lat = np.meshgrid(lonrange,latrange)
    latmean = np.mean([ymin,ymax])
    lonmean = np.mean([xmin,xmax])
    m = Basemap(llcrnrlon=xmin,llcrnrlat=ymin,urcrnrlon=xmax,urcrnrlat=ymax,            rsphere=(6378137.00,6356752.3142),            resolution='i',area_thresh=1000.,projection='lcc',            lat_1=latmean,lon_0=lonmean,ax=ax)
    # draw coastlines and political boundaries.
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    lons = np.arange(xmin,xmax,1.0)
    lats = np.arange(ymin,ymax,1.0)
    if isLeft:
        labels = labels=[1,0,0,0]
    else:
        labels = labels=[0,0,0,0]
    m.drawparallels(lats,labels=labels,color='white',fmt='%.1f') # draw parallels
    m.drawmeridians(lons,labels=[0,0,0,1],color='white',fmt='%.1f') # draw meridians
    pmesh = m.pcolormesh(lon,lat,np.flipud(grid.getData()),latlon=True)
    plt.hold(True)
    ax.set_title(tstr)
    m.colorbar(pmesh)


#ignore warnings that can reveal directory structure
import warnings
warnings.simplefilter("ignore")
#
shakegrid = ShakeGrid.load(shakefile)
pgagrid = shakegrid.getLayer('pga')
pgvgrid = shakegrid.getLayer('pgv')
mmigrid = shakegrid.getLayer('mmi')
fig,(ax0,ax1,ax2) = plt.subplots(nrows=1,ncols=3,figsize=(12,6))
fig.tight_layout()
map2DGrid(ax0,pgagrid,'Full PGA',isLeft=True)
map2DGrid(ax1,pgvgrid,'Full PGV')
map2DGrid(ax2,mmigrid,'Full MMI')
print pgagrid.getGeoDict()


geodict = ShakeGrid.getFileGeoDict(shakefile)
#bring in the shakemap by a half dimension (quarter on each side)
lonrange = geodict['xmax'] - geodict['xmin']
latrange = geodict['ymax'] - geodict['ymin']
geodict['xmin'] = geodict['xmin'] + lonrange/4.0
geodict['xmax'] = geodict['xmax'] - lonrange/4.0
geodict['ymin'] = geodict['ymin'] + latrange/4.0
geodict['ymax'] = geodict['ymax'] - latrange/4.0
shakegrid = ShakeGrid.load(shakefile,samplegeodict=geodict)
pgagrid = shakegrid.getLayer('pga')
pgvgrid = shakegrid.getLayer('pgv')
mmigrid = shakegrid.getLayer('mmi')
fig,(ax0,ax1,ax2) = plt.subplots(nrows=1,ncols=3,figsize=(12,6))
fig.tight_layout()
map2DGrid(ax0,pgagrid,'Trimmed PGA',isLeft=True)
map2DGrid(ax1,pgvgrid,'Trimmed PGV')
map2DGrid(ax2,mmigrid,'Trimmed MMI')
print pgagrid.getGeoDict()


fdict = ShakeGrid.getFileGeoDict(shakefile)
newdict = {'xmin':-120.0,
           'xmax':-118.0,
           'ymin':33.0,
           'ymax':35.0,
           'xdim':fdict['xdim'],
           'ydim':fdict['ydim']}
shakegrid = ShakeGrid.load(shakefile,samplegeodict=newdict)
pgagrid = shakegrid.getLayer('pga')
pgvgrid = shakegrid.getLayer('pgv')
mmigrid = shakegrid.getLayer('mmi')
fig,(ax0,ax1,ax2) = plt.subplots(nrows=1,ncols=3,figsize=(12,6))
fig.tight_layout()
map2DGrid(ax0,pgagrid,'Partial PGA',isLeft=True)
map2DGrid(ax1,pgvgrid,'Partial PGV')
map2DGrid(ax2,mmigrid,'Partial MMI')


print pgagrid.getGeoDict()


# Creating a ShakeMap
# 

pga = np.arange(0,16,dtype=np.float32).reshape(4,4)
pgv = np.arange(1,17,dtype=np.float32).reshape(4,4)
mmi = np.arange(2,18,dtype=np.float32).reshape(4,4)
geodict = {'xmin':0.5,'ymax':3.5,'ymin':0.5,'xmax':3.5,'xdim':1.0,'ydim':1.0,'nrows':4,'ncols':4}
layers = OrderedDict()
layers['pga'] = pga
layers['pgv'] = pgv
layers['mmi'] = mmi
shakeDict = {'event_id':'usabcd1234',
             'shakemap_id':'usabcd1234',
             'shakemap_version':1,
             'code_version':'4.0',
             'process_timestamp':datetime.utcnow(),
             'shakemap_originator':'us',
             'map_status':'RELEASED',
             'shakemap_event_type':'ACTUAL'}
eventDict = {'event_id':'usabcd1234',
             'magnitude':7.6,
             'depth':1.4,
             'lat':2.0,
             'lon':2.0,
             'event_timestamp':datetime.utcnow(),
             'event_network':'us',
             'event_description':'sample event'}
uncDict = {'pga':(0.0,0),
           'pgv':(0.0,0),
           'mmi':(0.0,0)}
shake = ShakeGrid(layers,geodict,eventDict,shakeDict,uncDict)
shake.save('grid.xml',version=1)
shake2 = ShakeGrid.load('grid.xml')
os.remove('grid.xml')





get_ipython().magic('matplotlib inline')


from grid.gdal import GDALGrid
from grid.shake import ShakeGrid
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path
import matplotlib


# The Grid class hierarchy can be used for reading, writing and manipulating various kinds 
# of 2D grid formats (GMT grids), or multi-layer 2D grid formats (ShakeMap).
# 

#######MODIFY THIS TO REFLECT THE LOCATION OF A GLOBAL GRID ON YOUR SYSTEM####
popfile = os.path.join(os.path.expanduser('~'),'pager','data','lspop2012.flt')
shakefile = os.path.join(os.path.expanduser('~'),'data','shakemaps','nepal2011.xml')
##############################################################################


# The grid classes can be used to get spatial information about the file before opening it. 
# 

fgeodict,xvar,yvar = GDALGrid.getFileGeoDict(popfile)
print fgeodict
bounds = (fgeodict['xmin'],fgeodict['xmax'],fgeodict['ymin'],fgeodict['ymax'])
print 'The file spans from %.3f to %.3f in longitude, and %.3f to %.3f in latitude.' % bounds


#41.798959,-125.178223 ulcorner of calif
#32.543919,-112.950439 lrcorner of calif
sdict = {'xmin':-125.178223,'xmax':-112.950439,'ymin':32.543919,'ymax':41.798959,'xdim':0.008,'ydim':0.008}
grid = GDALGrid.load(popfile,samplegeodict=sdict)
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(16,8));
data = grid.getData().copy()
data[data == -9999] = np.nan
data = np.ma.array (data, mask=np.isnan(data))
cmap = matplotlib.cm.bone_r
cmap.set_bad('b',1.)
im = plt.imshow(data,cmap=cmap,vmin=0,vmax=10000)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05);
plt.colorbar(im, cax=cax);
ax.set_title('Population Grid of California');


grid.getData().shape


# Let's test reading across the 180/-180 meridian
# 

#LL: -18.606228,175.891113
#UR: -14.990556,-178.439941
sdict = {'xmin':175.891113,'xmax':-178.439941,'ymin':-18.606228,'ymax':-14.990556}
grid = GDALGrid.load(popfile,samplegeodict=sdict)
xmin,xmax,ymin,ymax = grid.getBounds()
print xmin,xmax,ymin,ymax
m1y,m1x = grid.getRowCol(ymin,180.0)
m2y,m2x = grid.getRowCol(ymax,180.0)
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(16,8));
data = grid.getData().copy()
data = np.ma.array (data, mask=np.isnan(data))
cmap = matplotlib.cm.bone_r
cmap.set_bad('b',1.)
im = plt.imshow(data,cmap=cmap,vmin=0,vmax=10000)
axlim = plt.axis()
plt.hold(True)
plt.plot([m1x,m2x],[m1y,m2y],'r')
plt.axis(axlim)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05);
plt.colorbar(im, cax=cax);
ax.set_title('Population Grid around Fiji');


# Let's use a ShakeMap grid to cut out a piece of the population grid, then resample the ShakeMap to the population data.
# 

# grid._data = np.array([])
# del grid
fdict = ShakeGrid.getFileGeoDict(shakefile) #get the bounds of the ShakeMap
newdict = GDALGrid.getBoundsWithin(popfile,fdict) #get some bounds guaranteed to be inside the ShakeMap
shake = ShakeGrid.load(shakefile) #load the ShakeMap
popgrid = GDALGrid.load(popfile,samplegeodict=newdict) #load without any resampling or anything
xmin,xmax,ymin,ymax = popgrid.getBounds()
shake.interpolateToGrid(popgrid.getGeoDict())
popdict = popgrid.getGeoDict()
shakedict = shake.getGeoDict()
popdata = popgrid.getData().copy()
mmidata = shake.getLayer('mmi').getData()
popdata[popdata == -9999] = np.nan
popdata = np.ma.array(popdata, mask=np.isnan(popdata))
fig,axeslist = plt.subplots(nrows=1,ncols=2,figsize=(20,16))


#plot population
m1y,m1x = popgrid.getRowCol(ymin,180.0)
m2y,m2x = popgrid.getRowCol(ymax,180.0)
plt.sca(axeslist[0])
cmap = matplotlib.cm.bone_r
im = plt.imshow(popdata,cmap=cmap,vmin=0,vmax=10000)
axlim = plt.axis()
plt.hold(True)
plt.plot([m1x,m2x],[m1y,m2y],'r')
plt.axis(axlim)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(axeslist[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05);
plt.colorbar(im, cax=cax1);
axeslist[0].set_title('Population Grid around Nepal');

#plot pga
plt.sca(axeslist[1]);
cmap = matplotlib.cm.OrRd
im = plt.imshow(mmidata,cmap=cmap)
axlim = plt.axis()
plt.hold(True)
plt.plot([m1x,m2x],[m1y,m2y],'r')
plt.axis(axlim)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(axeslist[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05);
plt.colorbar(im, cax=cax2);
axeslist[1].set_title('MMI Grid around Nepal');


# Implement the PAGER exposure algorithm.
# 

#suppressing warnings that can reveal directory structure
import warnings
warnings.simplefilter("ignore")
#
if np.any(mmidata > 9):
    mmidata[mmidata > 9.0] = 9.0
expolist = np.zeros((5))
for mmi in range(5,10):
    immi = (mmidata > mmi-0.5) & (mmidata <= mmi+0.5)
    expolist[mmi-5] = np.sum(popdata[immi])
print expolist





