# Let's try to make a notebook that looks at the PSF.
# 

from lsst.daf.persistence import Butler


butler = Butler('/home/shared/twinkles/output_data_v2')


subset = butler.subset('calexp', filter='i')


dataId = subset.cache[6]


my_calexp = butler.get('calexp', **dataId)


psf = my_calexp.getPsf()


shape = psf.computeShape()


shape.getDeterminantRadius()


from lsst.afw.geom import Point2D
point = Point2D(50.1, 160.2)


shape = psf.computeShape(point)


shape.getDeterminantRadius()


shapes = []
for x in range(100):
    for y in range(100):
        point = Point2D(x*40., y*40.)
        shapes.append(psf.computeShape(point).getDeterminantRadius())


from matplotlib import pylab as plt


plt.hist(shapes)


src = butler.get('src', **dataId)


import numpy
star_idx = numpy.where(src.get('base_ClassificationExtendedness_value') == 0)


import math
star_shapes = []
psf_shapes = []
for x, y in zip(src.getX()[star_idx[0]], src.getY()[star_idx[0]]):
    point = Point2D(x, y)
    psf_shapes.append(psf.computeShape(point).getDeterminantRadius())
for xx, yy in zip(src.get('base_SdssShape_psf_xx')[star_idx[0]], src.get('base_SdssShape_psf_yy')[star_idx[0]]):
    star_shapes.append(math.sqrt(xx + yy))
star_shapes = numpy.array(star_shapes)
psf_shapes = numpy.array(psf_shapes)


good_star_idx = numpy.where(numpy.isfinite(star_shapes))[0]


plt.hist(psf_shapes[good_star_idx])


plt.hist(star_shapes[good_star_idx])


plt.scatter(star_shapes[good_star_idx], star_shapes[good_star_idx]/psf_shapes[good_star_idx]*(math.sqrt(2)/2), alpha=0.5)
plt.ylim(1.00075, 1.00175)





import numpy as np

import lsst.daf.persistence as dafPersist
import lsst.afw.geom as afwGeom


butler = dafPersist.Butler('/home/shared/twinkles/output_data_v2')


subset = butler.subset('src')
dataid = subset.cache[4] # Random choice
my_src = butler.get('src', dataId=dataid)
my_calexp = butler.get('calexp', dataId=dataid)
my_wcs = my_calexp.getWcs()


# Pick a bright star that was using to calibrate photometry
selection = my_src['calib_photometry_used']
index = np.argmax(np.ma.masked_array(my_src.getPsfFlux(), ~selection))


ra_target, dec_target = my_src['coord_ra'][index], my_src['coord_dec'][index] # Radians
radec = afwGeom.SpherePoint(ra_target, dec_target, afwGeom.radians)

xy = my_wcs.skyToPixel(radec)

print('%10s%15.6f%15.6f'%('sdss:', 
                          my_src['base_SdssCentroid_x'][index], 
                          my_src['base_SdssCentroid_y'][index]))
print('%10s%15.6f%15.6f'%('naive:', 
                          my_src['base_NaiveCentroid_x'][index], 
                          my_src['base_NaiveCentroid_y'][index])) 
print('%10s%15.6f%15.6f'%('gauss:', 
                          my_src['base_GaussianCentroid_x'][index], 
                          my_src['base_GaussianCentroid_y'][index])) 
print('%10s%15.6f%15.6f'%('radec:', 
                          xy.getX(), 
                          xy.getY()))


# Is it surprising that the pixel coordinates converted from `coord_ra and coord_dec
# 

# # Example notebook using a simple positional matching
# 

import numpy
from matplotlib import pylab as plt


from lsst.daf.persistence import Butler
from lsst.afw.image import abMagFromFlux, fluxFromABMag
from lsst.afw.table import MultiMatch
from lsst.meas.astrom import DirectMatchTask


butler = Butler('/home/shared/twinkles/output_data_v2')
subset = butler.subset('src', filter='r')


matched_cat = None
calexps = {}
for data_ref in subset:
    data_id = data_ref.dataId
    src_cat = data_ref.get('src')
    calexps[data_id['visit']] = data_ref.get('calexp')
    if matched_cat is None:
        id_fmt = {'visit':numpy.int64}
        matched_cat = MultiMatch(src_cat.schema, id_fmt)
    matched_cat.add(src_cat, data_id)
final_catalog = matched_cat.finish()


# Experimental cell
for ii, data_red in enumerate(subset):
    print(ii)
subset.__sizeof__()


object_ids = final_catalog.get('object')
unique_object_ids = set(object_ids)


object_count = {}
avg_flux = {}
stdev_flux = {}
avg_snr = {}
for obj_id in unique_object_ids:
    idx = numpy.where(final_catalog.get('object')==obj_id)[0]
    flux_inst = final_catalog.get('base_PsfFlux_flux')[idx]
    flux_inst_err = final_catalog.get('base_PsfFlux_fluxSigma')[idx]
    flag_gen = final_catalog.get('base_PsfFlux_flag')[idx]
    flag_edge = final_catalog.get('base_PsfFlux_flag_edge')[idx]
    flag_nogood = final_catalog.get('base_PsfFlux_flag_noGoodPixels')[idx]
    visit = final_catalog.get('visit')[idx]
    flux = []
    flux_err = []
    for f, f_err, v, fl1, fl2, fl3 in zip(flux_inst, flux_inst_err, visit, flag_gen, flag_edge, flag_nogood):
        if f > 0. and not (fl1|fl2|fl3):
            calib = calexps[v].getCalib()
            flux.append(fluxFromABMag(calib.getMagnitude(f)))
            flux_err.append(fluxFromABMag(calib.getMagnitude(f_err)))
    flux = numpy.array(flux)
    flux_err = numpy.array(flux_err)
    object_count[obj_id] = len(flux)
    avg_flux[obj_id] = numpy.average(flux)
    stdev_flux[obj_id] = numpy.std(flux)
    avg_snr[obj_id] = numpy.average(flux/flux_err)


matcher = DirectMatchTask(butler=butler)


matches = matcher.run(final_catalog, filterName='r').matches


ref_mags = {}
for match in matches:
    object_id = match.second.get('object')
    ref_mags[object_id] = abMagFromFlux(match.first.get('r_flux'))


mags = []
g_flux = []
g_flux_std = []
g_snr = []
ids = []
for obj_id in unique_object_ids:
    if object_count[obj_id] > 8:
        g_flux.append(avg_flux[obj_id])
        g_flux_std.append(stdev_flux[obj_id])
        g_snr.append(avg_snr[obj_id])
        mags.append(abMagFromFlux(avg_flux[obj_id]))
        ids.append(obj_id)
g_flux = numpy.array(g_flux)
g_flux_std = numpy.array(g_flux_std)


for i, m in zip(ids, mags):
    if i in ref_mags:
        plt.scatter(m, ref_mags[i]/m, color='b', alpha=0.5)
plt.ylim(0.98, 1.02) # there is one significant outlier


plt.scatter(mags, g_snr)


plt.scatter(ids, g_snr/(g_flux/g_flux_std))


idx = numpy.where(g_snr/(g_flux/g_flux_std) > 80)[0]


outlier_ids = [ids[el] for el in idx]


outlier1_idx = numpy.where(final_catalog.get('object')==outlier_ids[0])[0]
outlier2_idx = numpy.where(final_catalog.get('object')==outlier_ids[1])[0]


len(final_catalog)





# # Selecting a source and creating a cutout
# 
# The goal is to select a single (from individual visit) source based on it's properties and create a postage stamp image
# 
# Based partly on 
# * https://pipelines.lsst.io/getting-started/index.html#getting-started-tutorials 
# * https://github.com/RobertLuptonTheGood/notebooks/blob/master/Demos/Colour%20Images.ipynb
# 

import numpy as np

import lsst.daf.persistence as dafPersist
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.afw.image as afwImage

from astropy.visualization import ZScaleInterval


# 25 Apr 2018: Something unexpected happens here where when the kernel restarts, the `plt.rcParams` are not respected
# 

# Set plotting defaults
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)
zscale = ZScaleInterval()


butler = dafPersist.Butler('/home/shared/twinkles/output_data_v2')


# Show the available dataref keys for a dataset type 
# 
# Note that the `queryMetadata` and `subset` functions have similar functionality. `subset` assembles the datarefs, whereas `queryMetadata` returns a list of tuples 
# 

# Display the available keys
print(butler.getKeys('calexp'))
#print(dir(butler))
#butler.queryMetadata('calexp', butler.getKeys('calexp')) # Warning may not return in same order as getKeys()


# At this point, I was tempted to name a variable "filter", but this is already a reserved word in python. Using "band" instead.
# 

# Count the number of images in each filter
# Is queryMetadata faster than subset?
visit_array, band_array = map(np.array, zip(*butler.queryMetadata('calexp', ['visit', 'filter'])))
for band in np.unique(band_array):
    print(band, np.sum(band_array == band))


subset = butler.subset('src')
dataid = subset.cache[4] # A random choice of image
#print(subset.cache[4])
#print(dir(butler))
#help(butler.get)
#my_src = butler.get('src', dataId={'visit': 234})
my_src = butler.get('src', dataId=dataid)
my_calexp = butler.get('calexp', dataId=dataid)
my_wcs = my_calexp.getWcs()
my_calib = my_calexp.getCalib()
my_calib.setThrowOnNegativeFlux(False) # For magnitudes


# Access schema for the source catalog to find columns of interest
# 

#my_src.schema.getNames()
#my_src.schema # To see slots


# The cell below takes advantage of slots defined in the schema.
# 
# The `getMagnitude` function below can return some `nan` and `inf` values. In the category of "best practices", we are suggesting that these _not_ be replaced with sentinel values, but rather that
# 

# This is just a demonstration of the slot functionality
np.testing.assert_equal(my_calib.getMagnitude(my_src['base_PsfFlux_flux']),
                        my_calib.getMagnitude(my_src.getPsfFlux()))

psf_mag = my_calib.getMagnitude(my_src.getPsfFlux())
#psf_mag = np.where(np.isfinite(psf_mag), psf_mag, 9999.) # Don't set sentinel values!

cm_mag = my_calib.getMagnitude(my_src.getModelFlux())
cm_mag = np.where(np.isfinite(cm_mag), cm_mag, 9999.) # Don't set sentinel values

# If you have nan or inf values in your array, use the range argument to avoid searching for min and max
plt.figure()
#plt.yscale('log', nonposy='clip')
plt.hist(psf_mag, bins=np.arange(15., 26., 0.25), range=(15., 26.))
#plt.hist(np.nan_to_num(psf_mag), bins=np.arange(15., 26., 0.25)) # Alertnative
plt.xlabel('PSF Magnitude')
plt.ylabel('Counts')

plt.figure()
plt.scatter(psf_mag, psf_mag - cm_mag, c=my_src['base_ClassificationExtendedness_value'], cmap='coolwarm')
plt.colorbar().set_label('Classification')
plt.xlim(15., 26.)
plt.ylim(-1., 2.)
plt.xlabel('PSF Magnitude')
plt.ylabel('PSF - Model Magnitude')


# Now select a source based on it's properties, e.g., pick a bright star
# 

# Pick a bright star candidates
#mask = ~np.isfinite(psf_mag) | (my_src['base_ClassificationExtendedness_value'] == 1)
#index = np.argmin(np.ma.masked_array(psf_mag, mask))

# Pick a bright star that was using to calibrate photometry
selection = my_src['calib_photometry_used']
index = np.argmin(np.ma.masked_array(psf_mag, ~selection))

print(psf_mag[index])
print(index)

ra_target, dec_target = my_src['coord_ra'][index], my_src['coord_dec'][index] # Radians
#print(dir(afwCoord))
#print(dir(afwGeom))
#print(help(afwGeom.Point2D))
#coord = afwCoord.Coord(ra_target * afwGeom.degrees, dec_target * afwGeom.degrees)
#coord = afwGeom.Point2D(ra_target * afwGeom.degrees, dec_target * afwGeom.degrees)
radec = afwGeom.SpherePoint(ra_target, dec_target, afwGeom.radians) # Is this really the preferred way to do this?

#xy = afwGeom.PointI(my_wcs.skyToPixel(radec)) # This converts to integer
#xy = afwGeom.Point2D(my_wcs.skyToPixel(radec))
xy = my_wcs.skyToPixel(radec)
print(my_src['base_SdssCentroid_x'][index], my_src['base_SdssCentroid_y'][index])
print(xy.getX(), xy.getY())
#print(xy)
#dir(my_wcs)
#xy = my_wcs.skyToPixel(radec)

print(my_wcs.skyToPixel(radec).getX())
print(my_src.getX()[index])

# Equivalence check
assert my_src.getX()[index] == my_src['base_SdssCentroid_x'][index]


# Trying to isolate some behavior here that I don't understand
ra_target, dec_target = my_src['coord_ra'][index], my_src['coord_dec'][index] # Radians
radec = afwGeom.SpherePoint(ra_target, dec_target, afwGeom.radians)
xy = my_wcs.skyToPixel(radec)
print(my_wcs.skyToPixel(radec).getX())
print(my_src.getX()[index])


# ### Challenge to the reader:
# * Use callback with bokeh to interactively select a source
# 

# Probably this cell should go away
#my_calexp = butler.get('calexp', dataId={'visit': 234})
#subset = butler.subset('md')
#subset = butler.subset('wcs')
#subset.cache
#my_wcs = butler.get('wcs', dataId={'visit': 234})


cutoutSize = afwGeom.ExtentI(100, 100)
#my_wcs.skyToPixel(coord)
xy = afwGeom.PointI(my_wcs.skyToPixel(radec))
bbox = afwGeom.BoxI(xy - cutoutSize//2, cutoutSize)
#print(bbox)
#print(dir(my_calexp))
#print(help(butler.get))
#my_calexp.getBBox()

# Full frame image
image = butler.get('calexp', immediate=True, dataId=dataid) #.getMaskedImage()

# Postage stamp image only
cutout_image = butler.get('calexp_sub', bbox=bbox, immediate=True, dataId=dataid).getMaskedImage()


#import lsst.afw.display as afwDisplay
#print(dir(afwDisplay))


vmin, vmax = zscale.get_limits(image.image.array)
plt.imshow(image.image.array, vmin=vmin, vmax=vmax, cmap='binary')
plt.colorbar()
#dir(xy)
plt.scatter(xy.getX(), xy.getY(), color='none', edgecolor='red', s=200)
#my_calexp.image.array


# Demonstration of equivalency
my_calexp_cutout = my_calexp.Factory(my_calexp, bbox, afwImage.LOCAL)
assert np.all(my_calexp_cutout.image.array == cutout_image.image.array)

print(cutout_image.getDimensions())
vmin, vmax = zscale.get_limits(cutout_image.image.array)
plt.imshow(cutout_image.image.array, vmin=vmin, vmax=vmax, cmap='binary')

# Does the cutout_image have a wcs? It does not appear to...
plt.scatter(xy.getX() - cutout_image.getX0(), xy.getY() - cutout_image.getY0(), c='none', edgecolor='red', s=200)


# ### Challenge to the reader:
# * Can you plot the image with projection (including rotation) for equatorial coordinates with RA and Dec labeled?




