# This is a fairly complex assemblage of python code that produces animations of tree diffusion over a landscape using matplotlib. A large part of it is the "[baltic](https://github.com/blab/baltic)" lightweight tree parser that has evolved from a [linked list script](http://stackoverflow.com/questions/280243/python-linked-list/280286#280286) on stackoverflow. An even bigger part is tinkering with matplotlib to produce pretty figures, which can be removed. To begin there are several things that you will need to provide to use this script (or modify the code to proceed):
# - *Maximum clade credibility (MCC) tree with phylogeography annotations*. This one is a must, since we're animating a tree and the tree should say where each lineage is inferred to be.
# - *A map*. We went for GeoJSON files with administrative divisions of West African countries. These were additionally edited so that each division could be related to a location in the MCC tree. If you have another file format that contains coordinates that can define location polygons and that can be mapped to locations defined in the tree you can use that instead.
# - *Location points* (optional). These are the population centroides and also define the points between which lineages travel. These points also grow and shrink depending on how many lineages are co-circulating within a location. You could use the mean of X and Y coordinates of a location to define these points instead, which might [land you in water](https://en.wikipedia.org/wiki/Coastline_paradox). Alternatively you could find [convex hulls](https://en.wikipedia.org/wiki/Convex_hull) of each location and use the mean of their X and Y coordinates to define the points of travel.
# - *Case numbers* (optional). This is used to colour locations based on how many cases exist at the time point. With some modification of the code could be used to show anything.
# - *Standardised names* (optional). This maps the "standardised" names of each location back to the original name with diacritic marks. Just trying to be considerate towards another language.
# - *Plotting assist* (optional). This encodes information (vertical and horizontal text alignment) to position location names so they don't obscure each other. Ideally should be done automatically using something like [Coulomb's law](https://en.wikipedia.org/wiki/Coulomb%27s_law).
# 
# The animation works by going through each epi week in small slices, interpolating case numbers between each epi week report. This is animated in each frame as a change in colour intensity. At each time slice the tree is also cut up and lineages that exist at that time point are animated if they are travelling from one location to another, otherwise they just count towards the circle size of each population centroid. The lineages that are travelling are animated using [Bezier curves](https://en.wikipedia.org/wiki/B%C3%A9zier_curve) with a single control point placed at a given distance perpendicular to the line connecting points A and B between which lineages are travelling. Each Bezier line is defined by a number of segments which decrease in size towards the origin of the travelling lineage, giving them the comet-like appearance in the animation. 
# 
# There are two advantages to using Bezier curves:
# - Two lineages travelling from A to B and from B to A will not obscure each other. One will arc up, the other one will arc down.
# - The control point for the Bezier curve can be positioned at any distance from the direct line between points A and B. That means you can set the control point to be at a distance inversely proportional to the distance between points A and B, such that a lineage traveling far travels in nearly straight lines, but arc heavily when traveling between close points, ensuring locations close to each other exchanging viruses are not missed by the viewer. 
# 
# Also enjoy the music. If you like it - please consider buying the album (the dude's really friendly and appreciates the support).
# 

from IPython.display import HTML
license='<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">Gytis Dudas</span> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.'
HTML(license)


fr='<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">The movie from my presentation at <a href="https://twitter.com/hashtag/VGE16?src=hash">#VGE16</a> can be seen here: <a href="https://t.co/SjSwOYAWVM">https://t.co/SjSwOYAWVM</a></p>&mdash; evoGytis (@evogytis) <a href="https://twitter.com/evogytis/status/741215926482132992">June 10, 2016</a></blockquote><script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>'
HTML(fr)


# ### Import relevant information
# Population centroid locations, administrative division maps, case numbers, additional files for pretty plotting.
# 

import matplotlib as mpl ## matplotlib should not be set to inline mode to accelerate animation rendering and save memory
mpl.use('Agg') ## recommended backend for animations
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
import matplotlib.patheffects as path_effects
import matplotlib.animation as animation
from IPython.display import clear_output
from IPython.display import HTML

import numpy as np
import pandas as pd
from scipy.special import binom

from ebov_data import * ## load library

if locations: ## if data loaded previously - do nothing
    pass
else: ## call status, set countries, colour maps and load data
    status()
    setFocusCountries(focus=['SLE','LBR','GIN'])
    setColourMaps()
    loadData()

typeface='Helvetica Neue' ## set default matplotlib font and font size
mpl.rcParams['font.weight']=300
mpl.rcParams['axes.labelweight']=300
mpl.rcParams['font.family']=typeface
mpl.rcParams['font.size']=22

countryCaseCounts={} ## Output case numbers for each country separately

maxByCountry={}
for country in required_countries:
    countryCaseCounts[country]=np.vstack([[cases_byLocation[x][y] for y in dates] for x in cases_byLocation.keys() if country==location_to_country[x]]) ## stack case counts across locations
    countryCaseCounts[country]=np.sum(countryCaseCounts[country],axis=0)

    maxByCountry[country]=max([sum(cases_byLocation[x].values()) for x in cases_byLocation.keys() if location_to_country[x]==country]) ## find the location with the highest cumulative case numbers within each country
    if maxByCountry[country]==0.0:
        maxByCountry[country]=0.01

totalCaseCounts={loc:sum(cases_byLocation[loc].values()) for loc in cases_byLocation.keys()}
print '\n\nhighest case count in country:\n%s'%('\n'.join(['%s: %s'%(x,maxByCountry[x]) for x in maxByCountry.keys()]))

print '\ncase report dates: %s'%('\t'.join(dates))
print '\nnumber of districts in case report: %s'%(len([x for x in cases_byLocation.keys() if sum(cases_byLocation[x].values())>=1.0]))
print '\ndate of most recent report: %s'%(dates[-1])
frame='<iframe style="border: 0; width: 400px; height: 345px;" src="https://bandcamp.com/EmbeddedPlayer/album=2074815275/size=large/bgcol=ffffff/linkcol=63b2cc/artwork=small/track=42188979/transparent=true/" seamless><a href="http://obsrr.bandcamp.com/album/nustebusiam-neb-ti">NUSTEBUSIAM NEBUTI by OBSRR</a></iframe>'

print 'Done!'
HTML(frame)


# ### First test - plotting cumulative case numbers per country.
# 

fig,ax = plt.subplots(figsize=(20,20),facecolor='w') ## start figure

for i,loc in enumerate(locations): ## iterate over locations
    country=location_to_country[loc]
    countryColour=colours[country] ## fetch colour map for country
    
    if country in required_countries:
        if totalCaseCounts.has_key(loc):
            caseFrac=totalCaseCounts[loc]/float(maxByCountry[country]) ## get the location's case numbers as fraction of the highest cumulative case numbers in the country
        else:
            caseFrac=0.0

        c=countryColour(caseFrac) ## get colour

        for part in location_points[loc]: ## plot location borders and polygons
            xs=column(part,0)
            ys=column(part,1)
            ax.plot(xs,ys,color='grey',lw=1,zorder=0) ## mpl won't draw polygons unless there's something plotted

        ax.add_collection(PatchCollection(polygons[loc],facecolor=c,edgecolor='grey',lw=1,zorder=1)) ## add polygon

        lon,lat=popCentres[loc] ## plot population centres
        ax.scatter(lon,lat,80,facecolor=c,edgecolor=desaturate(countryColour(1-caseFrac),1.0),lw=2,zorder=2)

        vas=['bottom','top'] ## define available text alignments and corrections for text positions
        has=['left','right']
        corrections=[0.02,-0.02]

        h=1 ## set default text alignment (right, top)
        v=1
        if textCorrection.has_key(loc): ## check if custom text positions are available
            h,v=textCorrection[loc]
        
        ax.text(lon+corrections[h],lat+corrections[v]*1.5,r'%s'%map_to_actual[loc],size=20,va=vas[v],ha=has[h],
                alpha=0.8,path_effects=[path_effects.Stroke(linewidth=4, foreground='white'),
                                        path_effects.Stroke(linewidth=0.5, foreground='black')]) ## plot district names at population centres, with corrections so as not to obscure it

ycoord=np.mean([4.3,12.7]) ## this plots a distance bar to indicate distances in the map (set at 100km)
legend_y=12.0
legend_x1=-15
legend_x2=-14.08059

ax.plot([legend_x1,legend_x2],[legend_y,legend_y],color='k',lw=6) ## plot scale bar and label
ax.text(np.mean([legend_x1,legend_x2]),legend_y+0.04,'%.0f km'%metricDistance((legend_x1,legend_y),(legend_x2,legend_y)),size=36,va='bottom',ha='center')
    

colorbarTextSize=30 ## add colourbars - colours are log-normalized
colorbarTickLabelSize=24
colorbarWidth=0.02
colorbarHeight=0.4
colorbarIncrement=0.08

ax2 = fig.add_axes([colorbarIncrement, 0.1, colorbarWidth, colorbarHeight])
mpl.colorbar.ColorbarBase(ax2, cmap=colours['GIN'],norm=mpl.colors.LogNorm(vmin=1,vmax=maxByCountry['GIN']))
ax2.tick_params(labelcolor='k',size=0,labelsize=colorbarTickLabelSize)
ax2.yaxis.set_label_position('left') 
ax2.set_ylabel('Guinea',color='k',size=colorbarTextSize)

ax3 = fig.add_axes([colorbarIncrement*2, 0.1, colorbarWidth, colorbarHeight])
mpl.colorbar.ColorbarBase(ax3, cmap=colours['LBR'],norm=mpl.colors.LogNorm(vmin=1,vmax=maxByCountry['LBR']))
ax3.tick_params(labelcolor='k',size=0,labelsize=colorbarTickLabelSize)
ax3.yaxis.set_label_position('left') 
ax3.set_ylabel('Liberia',color='k',size=colorbarTextSize)

ax4 = fig.add_axes([colorbarIncrement*3, 0.1, colorbarWidth, colorbarHeight])
mpl.colorbar.ColorbarBase(ax4, cmap=colours['SLE'],norm=mpl.colors.LogNorm(vmin=1,vmax=maxByCountry['SLE']))
ax4.tick_params(labelcolor='k',size=0,labelsize=colorbarTickLabelSize)
ax4.yaxis.set_label_position('left') 
ax4.set_ylabel('Sierra Leone',color='k',size=colorbarTextSize)

ax.set_aspect('equal') ## equal aspect, since we're near the equator
ax.spines['top'].set_visible(False) ## suppress axes and their labels
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(size=0)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_ylim(ylimits) ## set axis limits
ax.set_xlim(xlimits)

for local_border in global_border:
    ax.plot(column(local_border,0),column(local_border,1),lw=1,ls='-',color='k') ## add international borders
    
plt.savefig(local_output+'EBOV_countryCases.png',dpi=300,bbox_inches='tight') ## save figure
plt.show()


print required_countries
print locations
print [location_to_country[x] for x in locations]
print totalCaseCounts


# ### Second test - plotting cumulative case numbers for the entire epidemic.
# 

fig,ax = plt.subplots(figsize=(20,20),facecolor='w') ## start figure

for i,loc in enumerate(locations):
    countryColour=mpl.cm.get_cmap('viridis') ## get correct colourmap
    
    if totalCaseCounts.has_key(loc):
        caseFrac=totalCaseCounts[loc]/float(totalCaseCounts['WesternUrban'])
    else:
        caseFrac=0.0
        
    c=countryColour(caseFrac)
    country=location_to_country[loc]
    if country in required_countries:
        lon,lat=popCentres[loc] ## plot population centres
        ax.scatter(lon,lat,50,facecolor='w',zorder=6)

        ax.add_collection(PatchCollection(polygons[loc],facecolor=c,edgecolor='w',lw=1,zorder=3))

        vas=['bottom','top'] ## define available text alignments and corrections for text positions
        has=['left','right']
        corrections=[0.02,-0.02]

        h=1 ## set default text alignment (right, top)
        v=1
        ## check if custom text positions are available
        if textCorrection.has_key(loc):
            h,v=textCorrection[loc]

        ## plot district names at population centres, with corrections so as not to obscure it
        ax.text(lon+corrections[h],lat+corrections[v]*1.5,r'%s'%map_to_actual[loc],size=20,va=vas[v],ha=has[h],alpha=1,path_effects=[path_effects.Stroke(linewidth=4, foreground='black'),path_effects.Stroke(linewidth=1, foreground='lightgrey')],zorder=10)

    
ycoord=np.mean([4.3,12.7])
legend_y=12.0
if len(required_countries)>3:
    legend_x1=-15.7
    legend_x2=-14.78059
else:
    legend_x1=-15.0
    legend_x2=-14.08059

ax.plot([legend_x1,legend_x2],[legend_y,legend_y],color='k',lw=6)
ax.text(np.mean([legend_x1,legend_x2]),legend_y+0.04,'%.0f km'%metricDistance((legend_x1,legend_y),(legend_x2,legend_y)),size=36,va='bottom',ha='center')

## add colourbars - colours are log-normalized
colorbarTextSize=30
colorbarTickLabelSize=24
colorbarWidth=0.02
colorbarHeight=0.4
colorbarIncrement=0.08

ax2 = fig.add_axes([colorbarIncrement*2, 0.1, colorbarWidth, colorbarHeight])
mpl.colorbar.ColorbarBase(ax2, cmap=mpl.cm.get_cmap('viridis'),norm=mpl.colors.LogNorm(vmin=1,vmax=float(max(totalCaseCounts.values()))))
ax2.tick_params(labelcolor='k',size=0,labelsize=colorbarTickLabelSize)
ax2.yaxis.set_label_position('left') 
ax2.set_ylabel('Total',color='k',size=colorbarTextSize)

ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(size=0)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_ylim(ylimits)
ax.set_xlim(xlimits)

for local_border in global_border:
    ax.plot(column(local_border,0),column(local_border,1),lw=2,color='k',zorder=5)
    ax.plot(column(local_border,0),column(local_border,1),lw=4,color='w',zorder=4)

plt.savefig(local_output+'EBOV_totalCases.png',dpi=300,bbox_inches='tight')
plt.show()


# ### Import MCC tree
# This bit uses baltic to import the MCC tree.
# 

#import baltic as bt ## import baltic (https://github.com/blab/baltic)
import imp
bt = imp.load_source('baltic', '/Users/evogytis/Documents/BLAB_baltic/baltic.py')

tree_path=path_to_dropbox+'Sequences/Aug2016_1610_genomes/Joint/HomogenousGLM/All_1610/Run1/Makona_1610_cds_ig.GLM.MCC.tree'

ll=bt.loadNexus(tree_path)

print 'Done!'


t0 = time.time() ## time how long animation takes

maxByCountryTemporal={country:max([max(cases_byLocation[z].values()) for z in cases_byLocation.keys() if location_to_country[z]==country]) for country in required_countries} ## get the highest case count at any point in the country's history - used to normalize colours later

smooth=2 ## smooth defines how many gridpoints will be established between each epiweek (50 used in the final version)

dpi=50 ## dots per inch for each .png (90 used in the final version)

Bezier_smooth=5 ## how many segments Bezier lines will have (15 used in the final version)

tracking_length=21 ## number of days over which to plot the lineage
depth=tracking_length/365.0

# locTrait='location' ## name of locations in the tree
locTrait='location.states'
print len([x for x in ll.Objects if x.traits.has_key(locTrait)==False])

dates2=dates[:dates.index('2015-11-02')+1] ## this helps with debugging - you can define some subset of dates from the case numbers that will animate only a fraction of the entire spreadsheet

Nframes=len(dates2)*smooth ## number of frames animation will have
print 'Number of frames to animate: %d'%(Nframes)

animation_duration=70 ## define how long the animation should be in seconds, it will work out the appropriate FPS
fps=int((Nframes)/animation_duration)

height_normalization=create_normalization([decimalDate('2013-12-01'),decimalDate(dates2[-1])],0.0,1.0) ## Bezier line control points are also positioned based on the time of the animation


### < RANDOMIZE LOCATIONS - randomise location labels, removes country border induced structure
# temp_location_to_country={}
# temp_popCentres={}
# reassign=[x for x in popCentres.keys() if location_to_country[x] in ['SLE','LBR','GIN']]

# for loc in popCentres.keys():
#     if location_to_country[loc] in ['SLE','LBR','GIN']:
#         randomChoice=np.random.randint(len(reassign)) ## get random index
#         assignedLoc=reassign[randomChoice]
#         newCountry=location_to_country[assignedLoc]
#         temp_popCentres[loc]=popCentres[reassign.pop(randomChoice)] ## assign new coordinates to loc, remove from list
#         temp_location_to_country[loc]=newCountry
    
# popCentres=temp_popCentres
# location_to_country=temp_location_to_country
### RANDOMIZE LOCATIONS />


global travelers ## the animation will need to have information to traveling lineages
travelers=[x for x in ll.Objects if x.parent!=ll.root and x.traits[locTrait]!=x.parent.traits[locTrait]] ## find lineages that have travelled - they're what's going to be animated
print '\nNumber of travelling lineages: %d (%.3f%% of all lineages)'%(len(travelers),len(travelers)/float(len(ll.Objects))*100)

def convertDate(x,start,end):
    """ Converts calendar dates between given formats """
    return dt.datetime.strftime(dt.datetime.strptime(x,start),end)

def animate(frame):
    tr=(frame%smooth)/float(smooth) ## tr is a fraction of smoothing
    
    t=int(frame/smooth) ## t is index of time slice

    #### Primary plotting (map)
    ax1.lines=[line for line in ax1.lines if '_border' in line.get_label()] ## reset lines (except borders) and texts in the plot
    ax1.texts=[]
    
    if len(dates2)-1>t: ## get epi week of next frame
        next_time=decimalDate(dates2[t+1])
    else:
        next_time=decimalDate(dates2[t])
    
    current_time=decimalDate(dates2[t]) ## get epi week of current frame

    delta_time=next_time-current_time ## find interval step size

    ax1.text(0.05,0.1,'Epi week: %s\nDecimal time: %.3f'%(convertDate(dates2[t],'%Y-%m-%d','%Y-%b-%d'),decimalDate(dates2[t])+(delta_time*tr)),size=40,transform=ax1.transAxes) ## add text to indicate current time point
    
    ax1.text(0.05,0.0,'@evogytis',size=28,ha='left',va='bottom',transform=ax1.transAxes)
    
    exists=[k for k in ll.Objects if k.parent!=ll.root and k.parent.absoluteTime<=current_time+(delta_time*tr)<=k.absoluteTime] ## identify lineages that exist at current timeslice

    lineage_locations=[c.traits[locTrait] for c in exists if c.traits[locTrait]!='Not Available'] ## identify locations where lineages are present
    presence=unique(lineage_locations)

    circle=[c.set_radius(0) for c in ax1.patches if '_circle' in c.get_label()] ## reset circle sizes

    for region in presence: ## iterate through every region where a lineage exists
        size=lineage_locations.count(region) ## count how many other lineages there are

        circle=[c for c in ax1.patches if c.get_label()=='%s_circle'%(region)][0] ## fetch circle at the location

        circle.set_radius(0.02+size*0.003) ## update its radius

    
    cur_slice=current_time+(delta_time*tr)

    for k in travelers: ## iterate through travelling lineages
        transition_time=(k.absoluteTime+k.parent.absoluteTime)/2.0 ## branch begins travelling mid-branch

        if cur_slice-depth<transition_time<cur_slice+depth: ## make sure transition is within period of animation
            frac=1-(transition_time-cur_slice)/float(depth) ## frac will go from 0.0 to 2.0

            ori=k.parent.traits[locTrait] ## fetch locations
            des=k.traits[locTrait]  

            pointA=popCentres[ori] ## find coordinates of start and end locations
            beginX,beginY=pointA
            pointB=popCentres[des]
            endX,endY=pointB

            fc='k' ## colour line black unless it's travelling between countries

            origin_country=location_to_country[ori] ## get countries for start and end points
            destination_country=location_to_country[des]
            
            if origin_country!=destination_country: ## if lineage travelling internationally - colour it by origin country
                countryColour=colours[origin_country]
                fc=countryColour(0.6)

            distance=math.sqrt(math.pow(beginX-endX,2)+math.pow(beginY-endY,2)) ## calculate distance between locations

            normalized_height=height_normalization(cur_slice) ## normalize time of lineage

            adjust_d=-1+(1-normalized_height)+1/float(distance)**0.15+0.5 ## adjust Bezier line control point distance
            n=Bezier_control(pointA,pointB,adjust_d) ## find the coordinates of a point n that is at a distance adjust_d, perpendicular to the mid-point between points A and B

            bezier_start=frac-0.5 ## Bezier line begins at half a fraction along the path
            bezier_end=frac

            if bezier_start<0.0: ## if Bezier line begins outside the interval - make sure it stays within interval
                bezier_start=0.0
            if bezier_end>1.0:
                bezier_end=1.0

            bezier_line=Bezier([pointA,n,pointB],bezier_start,bezier_end,num=Bezier_smooth) ## get Bezier line points

            if bezier_start<1.0: ## only plot if line begins before destination
                for q in range(len(bezier_line)-1): ## iterate through Bezier line segments with fading alpha and reducing width
                    x1,y1=bezier_line[q]
                    x2,y2=bezier_line[q+1]

                    segL=(q+1)/float(len(bezier_line)) ## fraction along length of Bezier line
                    
                    ax1.plot([x1,x2],[y1,y2],lw=7*segL,alpha=1,color=fc,zorder=99,solid_capstyle='round') ## plot actual lineage

                    ax1.plot([x1,x2],[y1,y2],lw=10*segL,alpha=1,color='w',zorder=98,solid_capstyle='round') ## plot underlying white background to help lineages stand out

    for i,loc in enumerate(locations):  ##plot new districts
        country=location_to_country[loc]
        countryColour=colours[country]
        c=countryColour(0)
        
        if country in required_countries:
            if len(dates2)-1>t:
                nex_cases=cases_byLocation[loc][dates2[t+1]]
            else:
                nex_cases=cases_byLocation[loc][dates2[t]]

            cur_cases=cases_byLocation[loc][dates2[t]]

            country_max=1+float(maxByCountryTemporal[country]) ## get the maximum number of cases seen in the country at any point

            col=1+cur_cases+(nex_cases-cur_cases)*tr ## interpolate between current and next cases (add one so that single cases show up after log normalization)     

            country_max=float(maxByCountryTemporal[country]) ## find out what fraction of the maximum number of cases was reported
            c=countryColour(np.log10(col)/np.log10(country_max))
            
            polygons=[p for p in ax1.patches if p.get_label()=='%s_polygon'%(loc)]
            for polygon in polygons:
                polygon.set_facecolor(c) ## change the colour of locations based on cases
    
    frame+=1 ## next frame
    
    update=10 ## update progress bar every X frames
    
    #### Secondary plotting (tree)
    Ls2=[x for x in ax2.lines if 'Colour' not in str(x.get_label())] ## fetch all the lines with labels in tree plot
    partials=[x for x in ax2.lines if 'partial' in str(x.get_label())]
    finished_lines=[x for x in ax2.lines if 'finished' in str(x.get_label())]
    finished_points=[x for x in ax2.collections if 'finished' in str(x.get_label())]
    
    finished_labels=[str(x.get_label()) for x in finished_lines]+[str(x.get_label()) for x in finished_points]
    partial_labels=[str(x.get_label()) for x in partials]
    
    if frame%update==0: ## progress bar
        clear_output()
        timeElapsed=(time.time() - t0)/60.0
        progress=int((frame*(50/float(Nframes))))
        percentage=frame/float(Nframes)*100
        rate=timeElapsed/float(frame)
        ETA=rate*(Nframes-frame)
        sys.stdout.write("[%-50s] %6.2f%%  frame: %5d %10s  time: %5.2f min  ETA: %5.2f min (%6.5f s/operation) %s %s %s" % ('='*progress,percentage,frame,dates2[t],timeElapsed,ETA,rate,len(partials),len(finished_lines),len(finished_points)))
        sys.stdout.flush()

        
    ####
    ## COMMENT this bit out if you don't want the tree to appear out of the time arrow
    ####
    for ap in ll.Objects:
        idx='%s'%(ap.index)
        xp=ap.parent.absoluteTime

        x=ap.absoluteTime
        y=ap.y

        location=ap.traits[locTrait]
        country=location_to_country[location]
        cmap=colours[country]
        c=cmap(normalized_coords[location])
        
        if xp<=cur_slice<x: ## branch is intersected
            if 'partial_%s'%(idx) in partial_labels: ## if branch was drawn before
                l=[w for w in partials if 'partial_%s'%(idx)==str(w.get_label())][-1]
                l.set_data([xp,cur_slice],[y,y])
            else: ## branch is intersected, but not drawn before
                ax2.plot([xp,cur_slice],[y,y],lw=branchWidth,color=c,zorder=99,label='partial_%s'%(ap.index))
                
        if x<=cur_slice: ## time arrow passed branch - add it to finished class
            if 'partial_%s'%(idx) in partial_labels:
                l=[w for w in partials if 'partial_%s'%(idx)==str(w.get_label())][-1]
                l.set_data([xp,x],[y,y])
                l.set_label('finished_%s'%(idx))
                finished_labels.append('finished_%s'%(idx))
                
            if 'finished_%s'%(idx) not in finished_labels:
                ax2.plot([xp,x],[y,y],lw=branchWidth,color=c,zorder=99,label='finished_%s'%(ap.index))
                
            if 'partial_%s'%(idx) in partial_labels or 'finished_%s'%(idx) not in finished_labels:
                if isinstance(ap,bt.leaf):
                    ax2.scatter(x,y,s=tipSize,facecolor=c,edgecolor='none',zorder=102,label='finished_%s'%(ap.index))
                    ax2.scatter(x,y,s=tipSize+30,facecolor='k',edgecolor='none',zorder=101,label='finished_%s'%(ap.index))
                elif isinstance(ap,bt.node):
                    yl=ap.children[0].y
                    yr=ap.children[-1].y
                    ax2.plot([x,x],[yl,yr],lw=branchWidth,color=c,zorder=99,label='finished_%s'%(ap.index))
    ####
    ## COMMENT this bit out if you don't want the tree to appear out of the time arrow
    ####
                
    for l in Ls2:
        if 'time' in l.get_label():
            l.set_data([cur_slice,cur_slice],[0,1]) ## adjust time arrow
            
        #### 
        ## UNCOMMENT this bit if you'd like lineages to be coloured over time
        ####
#         else:
#             ## fetch all line data
#             d_xs,d_ys=l.get_data()
            
#             ## extract x coordinate
#             start,end=d_xs
            
#             ## if time arrow passed end point of line - delete line
#             if end<cur_slice:
#                 ax2.lines.remove(l)
                
#             ## if time arrow passed start of line - adjust start of line
#             elif start<cur_slice:
#                 l.set_data([cur_slice,end],d_ys)
    
#     ## iterate over collections (scatter points) in tree plot
#     Ps2=[x for x in ax2.collections if 'Colour' not in str(x.get_label())]
    
#     for p in Ps2:
#         ## fetch coordinates
#         coords=p.get_offsets()
#         ## only alter points with 1 coordinate
#         if len(coords)==1:
#             ## remove black and white point if time arrow has passed
#             if coords[0][0]<=float(cur_slice):
#                 ax2.collections.remove(p)
        #### 
        ## UNCOMMENT this bit if you'd like lineages to be coloured over time
        ####
    
    ### Tertiary plotting (cases)
    Ls3=[x for x in ax3.lines if 'Colour' not in str(x.get_label())] ## fetch all the lines with labels in cases plot
    
    for l in Ls3:
        if 'time' in l.get_label():
            l.set_data([cur_slice,cur_slice],[0,1]) ## adjust time arrow
        else:
            d=l.get_xydata() ## fetch all line data
            
            for e in range(len(d)-1): ## iterate over points
                x_now=d[:,0][e] ## get coordinates of current and next positions
                x_nex=d[:,0][e+1]

                y_now=d[:,1][e]
                y_nex=d[:,1][e+1]
                
                if x_now<cur_slice: ## if beginning of line passed time arrow
                    d[:,0][e]=cur_slice # adjust coordinate so it's sitting on top of time arrow
                    d[:,1][e]=y_now+((y_nex-y_now)/(x_nex-x_now))*(cur_slice-x_now) 

## This part will initialise the map, case numbers, and tree (in grey, if so set up)
plt.clf() 
plt.cla()
plt.figure(figsize=(32,18),facecolor='w') ## start figure

gs = gridspec.GridSpec(2, 2,width_ratios=[18,14],height_ratios=[14,4],hspace=0.05555,wspace=0.05882) ## define subplots

ax1 = plt.subplot(gs[0:, 0]) ## ax1 is map
ax2 = plt.subplot(gs[0, 1]) ## ax2 is tree
ax3 = plt.subplot(gs[1, 1]) ## ax3 is cases

for l,local_border in enumerate(global_border): ## plot the international borders
    ax1.plot(column(local_border,0),column(local_border,1),lw=5,color='w',zorder=96,label='%d_border_bg'%(l))
    ax1.plot(column(local_border,0),column(local_border,1),lw=2,color='k',zorder=97,label='%d_border'%(l))
    
for i,loc in enumerate(locations): ## iterate over locations, plot the initial setup
    country=location_to_country[loc]
    countryColour=colours[country]
    
    c=countryColour(0) ## zero cases colour

    if country in required_countries:
        N_lineages=plt.Circle(popCentres[loc],radius=0,label='%s_circle'%(loc),facecolor='indianred',edgecolor='k',lw=1,zorder=100) ## add circle that tracks the number of lineages at location with radius 0 to begin with
        ax1.add_patch(N_lineages)

        for part in location_points[loc]: ## plot every part of each location (islands, etc)
            poly=plt.Polygon(part,facecolor=c,edgecolor='grey',lw=1,label='%s_polygon'%(loc),closed=True,zorder=95)
            ax1.add_patch(poly)

ax1.spines['top'].set_visible(False) ## remove borders and axis labels
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(size=0)
ax1.set_xticklabels([])
ax1.set_yticklabels([])

ax1.set_ylim(ylimits) ## set plot limits
ax1.set_xlim(xlimits)

xlabels=['2013-%02d-01'%x for x in range(12,13)] ## setup time labels
xlabels+=['2014-%02d-01'%x for x in range(1,13)]
xlabels+=['2015-%02d-01'%x for x in range(1,13)]
xlabels+=['2016-%02d-01'%x for x in range(1,3)]

################
## Secondary plot begins - CASES
################
for c,country in enumerate(required_countries): ## iterate through countries
    greyColour=mpl.cm.Greys
    countryColour=colours[country]
    xs=[decimalDate(x) for x in dates] ## get time points based on epiweeks
    ys=[sum([cases_byLocation[loc][epiweek] for loc in locations if location_to_country[loc]==country]) for epiweek in dates] ## get cases in country at each epiweek
    
    grey_colour=greyColour((required_countries.index(country)+1)/float(len(required_countries)+2))
    
    ax3.plot(xs,ys,lw=3.3,color=grey_colour,zorder=2,label='BW') ## plot the same cases, one in full colour and one in grey on top to obscure colour
    ax3.plot(xs,ys,lw=3,color=countryColour(0.6),zorder=1,label='Colour')
    
ax3.axvline(decimalDate(dates[0]),color='k',lw=3,label='time',zorder=100) ## add time arrow to indicate current time

ax3.set_xticks([decimalDate(x)+1/24.0 for x in xlabels]) ## add ticks, tick labels and month markers
ax3.set_xticklabels([convertDate(x,'%Y-%m-%d','%b\n%Y') if x.split('-')[1]=='01' else convertDate(x,'%Y-%m-%d','%b') for x in xlabels])
[ax3.axvspan(decimalDate(xlabels[x]),decimalDate(xlabels[x])+1/12.,facecolor='k',edgecolor='none',alpha=0.04) for x in range(0,len(xlabels),2)]

ax3.xaxis.tick_bottom() ## make cases plot pretty
ax3.yaxis.tick_left()
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax3.yaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
ax3.set_xlim(decimalDate('2013-12-01'),decimalDate(dates2[-1]))
ax3.set_ylim(0,700)

ax3.tick_params(which='both',direction='out')
ax3.tick_params(axis='x',size=0,labelsize=18)
ax3.tick_params(axis='y',which='major',size=8,labelsize=30)
ax3.tick_params(axis='y',which='minor',size=5)
ax3.set_xticklabels([])
################
## Secondary plot ends - CASES
################


################
## Tertiary plot begins - TREE
################
tipSize=20
branchWidth=2

posteriorCutoff=0.0

####
## UNCOMMENT if you'd like the tree to be plotted in grey initially and get coloured over time
####
## iterate over objects in tree
# for k in ll.Objects:
#     location=k.traits[locTrait]
#     country=location_to_country[location]
#     cmap=colours[country]
#     c=cmap(normalized_coords[location])
    
#     countryColour=mpl.cm.Greys
#     grey_colour=countryColour((required_countries.index(country)+1)/float(len(required_countries)+2))
    
#     y=k.y
#     yp=k.parent.y
    
#     x=k.absoluteTime
#     xp=k.parent.absoluteTime
    
#     if isinstance(k,bt.leaf):
#         ## plot BW tree on top
#         ax2.scatter(x,y,s=tipSize,facecolor=grey_colour,edgecolor='none',zorder=102,label='LeafBW_%d'%(k.index))
#         ax2.scatter(x,y,s=tipSize+30,facecolor='k',edgecolor='k',zorder=100,label='Colour')
#         ax2.plot([xp,x],[y,y],color=grey_colour,lw=branchWidth,zorder=99,label='LeafBranchBW_%d'%(k.index))
        
#         ## plot colour tree underneath
#         ax2.scatter(x,y,s=tipSize,facecolor=c,edgecolor='none',zorder=101,label='LeafColour_%d'%(k.index))
#         ax2.plot([xp,x],[y,y],color=c,lw=branchWidth,zorder=98,label='LeafBranchColour_%d'%(k.index))
        
#     elif isinstance(k,bt.node):
#         yl=k.children[0].y
#         yr=k.children[-1].y
        
#         if xp==0.0:
#             xp=x

#         ls='-'
#         if k.traits['posterior']<posteriorCutoff:
#             ls='--'
            
#         ax2.plot([xp,x],[y,y],color=grey_colour,lw=branchWidth,ls=ls,zorder=99,label='NodeBranchBW_%d'%(k.index))
#         ax2.plot([x,x],[yl,yr],color=grey_colour,lw=branchWidth,ls=ls,zorder=99,label='NodeHbarBW_%d'%(k.index))
        
#         ax2.plot([xp,x],[y,y],color=c,lw=branchWidth,ls=ls,zorder=98,label='NodeBranchColour_%d'%(k.index))
#         ax2.plot([x,x],[yl,yr],color=c,lw=branchWidth,ls=ls,zorder=98,label='NodeHbarColour_%d'%(k.index))
####
## UNCOMMENT if you'd like the tree to be plotted in grey initially and get coloured over time
####

ax2.axvline(decimalDate(dates[0]),color='k',lw=3,label='time',zorder=200) ## add time arrow to indicate current time

ax2.set_xticks([decimalDate(x)+1/24.0 for x in xlabels]) ## add ticks, tick labels and month markers
ax2.set_xticklabels([convertDate(x,'%Y-%m-%d','%b\n%Y') if x.split('-')[1]=='01' else convertDate(x,'%Y-%m-%d','%b') for x in xlabels])
[ax2.axvspan(decimalDate(xlabels[x]),decimalDate(xlabels[x])+1/12.,facecolor='k',edgecolor='none',alpha=0.04) for x in range(0,len(xlabels),2)]

ax2.xaxis.tick_bottom() ## make tree plot pretty
ax2.yaxis.tick_left()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

ax2.tick_params(axis='x',size=0)
ax2.tick_params(axis='y',size=0)
ax2.set_xticklabels([])
ax2.set_yticklabels([])

ax2.set_xlim(decimalDate('2013-12-01'),decimalDate(dates2[-1]))
ax2.set_ylim(-5,len(ll.Objects)/2.0+6)
################
## Tertiary plot ends - TREE
################

for i in range(0,Nframes): ## iterate through each frame
    animate(i) ## animate will modify the map, tree and cases
    plt.savefig(local_output+'EBOV_animation/ani_frame_%05d.png'%(i), format='png',bbox_inches='tight',dpi=dpi) ## save individual frames for stitching up using 3rd party software (e.g. FFMpeg)
    
print '\n\nDONE!'

## Expect a HUGE slow down around August 2014 (about 0.02 s/frame) due to lots of EBOV movement 
print '\nTime taken: %.2f minutes'%((time.time() - t0)/60.0)

fps=int((Nframes)/animation_duration)
print 'Recommended fps to get animation %d seconds long: %d'%(animation_duration,fps)
plt.show()
## ffmpeg was used to stitch the resulting frames together with the following command:
## ffmpeg -framerate 70 -i ani_frame_%05d.png -s:v 2560x1440 -c:v libx264 -profile:v high -pix_fmt yuv420p EBOV_animation.HD.264.mp4


# ## Try different functions for positioning the Bezier line control point.
# 

get_ipython().magic('matplotlib inline')

## start figure
fig,ax = plt.subplots(figsize=(20,20),facecolor='w')

## iterate over locations, plot the initial setup
# for i,loc in enumerate(locations):
   
## how many segments Bezier lines will have
Bezier_smooth=15
    
print popCentres.keys()
['Gbarpolu', 
 'Lola', 
 'Tonkolili', 
 'Gaoual', 
 'Macenta', 
 'Mali', 
 'Gueckedou', 
 'Boke', 
 'Telimele', 
 'Kerouane', 
 'Kindia', 
 'Tougue', 
 'Mandiana', 
 'Maryland', 
 'Koubia', 
 'Forecariah', 
 'Beyla', 
 'Labe', 
 'Bo', 
 'Pujehun', 
 'Coyah', 
 'Kouroussa', 
 'Lofa', 
 'Yamou', 
 'Nzerekore', 
 'Kambia', 
 'Boffa', 
 'Koundara', 
 'GrandCapeMount', 
 'Fria', 
 'Sinoe', 
 'Kenema', 
 'Mamou', 
 'GrandGedeh', 
 'PortLoko', 
 'Koinadugu', 
 'Kailahun', 
 'Nimba', 
 'Moyamba', 
 'RiverCess', 
 'Bombali', 
 'Faranah', 
 'GrandBassa', 
 'Montserrado', 
 'Pita', 
 'Lelouma', 
 'Bong', 
 'WesternRural', 
 'Siguiri', 
 'Dalaba', 
 'Dabola', 
 'Dinguiraye', 
 'Kono', 
 'WesternUrban', 
 'Margibi', 
 'Kankan', 
 'Bomi', 
 'GrandKru', 
 'Conakry', 
 'Dubreka', 
 'Bonthe', 
 'Kissidougou', 
 'RiverGee']

xs=['WesternUrban']

## plot a random proportion of all lines
plottingFraction=0.9

for x in xs:
    countryColour=colours[location_to_country[x]]

    for y in popCentres.keys():
        fc='k'
        if location_to_country[x]!=location_to_country[y]:
            fc=colours[location_to_country[x]](0.6)

        if x!=y and np.random.random()<=plottingFraction:
            pointA=popCentres[x]
            beginX,beginY=pointA

            pointB=popCentres[y]
            endX,endY=pointB

            ## calculate distance between locations
            distance=math.sqrt(math.pow(beginX-endX,2)+math.pow(beginY-endY,2))
            
            #############
            ## this controls the distance at which the Bezier line control point will be placed
            #############
            #adjust_d=1-1/float(distance)**0.1-0.5
            adjust_d=-1+0.1/float(distance)**0.15+0.5
            #adjust_d=0.1+np.e**(1-distance**0.8)
            ## find the coordinates of a point n that is at a distance adjust_d, perpendicular to the mid-point between points A and B
            n=Bezier_control(pointA,pointB,adjust_d)

            ## Bezier line begins at half a fraction along the path
            bezier_start=0.0
            bezier_end=1.0

            ## get Bezier line points
            bezier_line=Bezier([pointA,n,pointB],bezier_start,bezier_end,num=Bezier_smooth)

            ## iterate through Bezier line segments with fading alpha and reducing width
            for q in range(len(bezier_line)-1):
                x1,y1=bezier_line[q]
                x2,y2=bezier_line[q+1]

                ## fraction along length of Bezier line
                segL=(q+1)/float(len(bezier_line))

                ## plot actual lineage
                ax.plot([x1,x2],[y1,y2],lw=7*segL,alpha=1,color=fc,zorder=99,solid_capstyle='round')

                ## plot underlying white background to help lineages stand out
                ax.plot([x1,x2],[y1,y2],lw=9*segL,alpha=1,color='w',zorder=98,solid_capstyle='round')

## plot all locations
for x in popCentres.keys():
    
    countryColour=colours[location_to_country[x]]
    c=countryColour(0)
    
    ## add circle with radius 0 to begin with
    # it tracks the number of lineages at location
    N_lineages=plt.Circle(popCentres[x],radius=0.03,label='%s_circle'%(x),facecolor='indianred',edgecolor='k',lw=1,zorder=100)
    ax.add_patch(N_lineages)

    ## plot every part of each location (islands, etc)
    for part in location_points['%s'%(x)]:
        poly=plt.Polygon(part,facecolor=c,edgecolor='grey',lw=1,label='%s_polygon'%(x),closed=True)
        ax.add_patch(poly)

## plot the international borders
for l,local_border in enumerate(global_border):
    ax.plot(column(local_border,0),column(local_border,1),lw=2,color='k',zorder=97,label='%d_border'%(l))

        
## make map pretty
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(size=0)
ax.set_xticklabels([])
ax.set_yticklabels([])
#ax.text(0.9,0.9,'%s'%(dates3[0]),size=30,transform=ax.transAxes)
ax.set_ylim(ylimits)
ax.set_xlim(xlimits)

plt.show()


x1,y1=popCentres['WesternUrban']
x2,y2=popCentres['WesternRural']
distances=[]
for i in popCentres.keys():
    for j in popCentres.keys():
        if i!=j:
            x1,y1=popCentres[i]
            x2,y2=popCentres[j]
            distance=math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))
            distances.append(distance)

print 'shortest distance: %.4f\nlargest distance: %.4f'%(min(distances),max(distances))
distances=sorted(distances)
## start figure
fig,ax = plt.subplots(figsize=(20,10),facecolor='w')

ax.hist(distances,bins=100,edgecolor='none',facecolor='steelblue')

ax.tick_params(labelsize=26)
ax.set_xlabel('distance',size=28)
ax.set_ylabel('frequency',size=28)
plt.show()


## start figure
fig,ax = plt.subplots(figsize=(20,10),facecolor='w')

ys=[-1+0.1+1/float(distance)**0.15+0.5 for distance in distances]

print 'distribution of control point distances'
ax.hist(ys,bins=100,edgecolor='none',facecolor='steelblue')
ax.set_xlabel('control point distance',size=28)
ax.set_ylabel('frequency',size=28)
ax.tick_params(labelsize=26)
plt.show()

print 'relationship between actual distance and control point distance'
## start figure
fig,ax = plt.subplots(figsize=(20,10),facecolor='w')

ax.plot(distances,ys,color='indianred',alpha=1,lw=3)
ax.set_xlabel('actual distance',size=28)
ax.set_ylabel('control point distance',size=28)
#ax.set_xlim(left=0.0)
#ax.set_ylim(bottom=0.0)
ax.tick_params(labelsize=26)
plt.show()


get_ipython().magic('matplotlib inline')
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import HTML

from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import copy

from ebov_data import *

if locations:
    pass
else:
    status()
    setFocusCountries(['SLE','LBR','GIN'])
    setColourMaps()
    loadData()

typeface='Helvetica Neue' ## set default matplotlib font and font size
#typeface='Helvetica'
mpl.rcParams['font.weight']=300
mpl.rcParams['axes.labelweight']=300
mpl.rcParams['font.family']=typeface
mpl.rcParams['font.size']=22
mpl.rcParams['pdf.fonttype']=42

path='<iframe style="border: 0; width: 400px; height: 460px;" src="https://bandcamp.com/EmbeddedPlayer/album=1406422575/size=large/bgcol=333333/linkcol=e99708/artwork=small/track=2364395925/transparent=true/" seamless><a href="http://romowerikoito.bandcamp.com/album/und-ina">Undeina by Romowe Rikoito</a></iframe>'
totalCaseCounts={x:sum(cases_byLocation[x].values()) for x in cases_byLocation.keys()}

countryCaseCounts={}
maxByCountry={}
for country in ['SLE','GIN','LBR']:
    countryCaseCounts[country]=np.vstack([[cases_byLocation[x][y] for y in dates] for x in cases_byLocation.keys() if location_to_country[x]==country])
    countryCaseCounts[country]=np.sum(countryCaseCounts[country],axis=0)
    maxByCountry[country]=max([totalCaseCounts[z] for z in totalCaseCounts.keys() if location_to_country[z]==country])

print '\nhighest case count in district:\n%s'%('\n'.join(['%s: %s'%(x,maxByCountry[x]) for x in maxByCountry.keys()]))

print '\ncase report dates: %s'%('\t'.join(dates))
print '\nnumber of districts in case report: %s'%(len(totalCaseCounts.keys()))
print '\ndate of most recent report: %s'%(dates[-1])

print 'Done!'
HTML(path)


#import baltic as bt ## use baltic, available at https://github.com/blab/baltic
import imp
bt = imp.load_source('baltic', '/Users/evogytis/Documents/BLAB_baltic/baltic.py')


tree_path=path_to_dropbox+'Sequences/Aug2016_1610_genomes/Joint/HomogenousGLM/All_1610/Run1/Makona_1610_cds_ig.GLM.MCC.tree' 

ll=bt.loadNexus(tree_path)

for k in ll.Objects: ## MCC tree in figtree format, with actual tip names in the string
    if k.branchType=='leaf':
        k.name=k.numName

print 'Done!'


# ## Plot MCC tree with a map of West Africa
# 

# fig,ax = plt.subplots(figsize=(15,320),facecolor='w') ## tree expanded vertically to allow outputing sequence labels
fig,ax = plt.subplots(figsize=(15,15),facecolor='w')

## name of districts in tree
# traitName='location'
traitName='location.states'

tipSize=20 ## tip circle radius
branchWidth=2 ## line width for branches

posteriorCutoff=0.0 ## posterior cutoff if collapsing tree

plot_tree=ll ## reference
# plot_tree=ll.collapseNodes('posterior',posteriorCutoff) ## collapse nodes lower than a given level of support

for k in plot_tree.Objects: ## iterate over branches in the tree
    location=k.traits[traitName] ## get inferred location of branch
    country=location_to_country[location] ## find country of location
    cmap=colours[country] ## fetch colour map for country
    c=cmap(normalized_coords[location]) ## get colour of location
    y=k.y ## get y coordinates
    yp=k.parent.y ## get parent's y coordinate
    
    x=k.absoluteTime ## x coordinate is absolute time
    xp=k.parent.absoluteTime ## get parent's absolute time
    
    if isinstance(k,bt.leaf): ## if tip...
        ax.scatter(x,y,s=tipSize,facecolor=c,edgecolor='none',zorder=100) ## put a circle at each tip
        ax.scatter(x,y,s=tipSize+30,facecolor='k',edgecolor='none',zorder=99)
        #ax.text(x+5/365.0,y,'%s'%(k.name),size=12,zorder=101,ha='left',va='center') ## uncomment to add tip labels (only use if tree is set up with a lot of vertical space)
        
    elif isinstance(k,bt.node): ## if node...
        yl=k.children[0].y ## get y coordinates of first and last child
        yr=k.children[-1].y
        
        if xp==0.0:
            xp=x

        ls='-'
        if k.traits['posterior']<posteriorCutoff: ## change to dotted line if posterior probability too low
            ls='--'
        ax.plot([x,x],[yl,yr],color=c,lw=branchWidth,ls=ls,zorder=98) ## plot vertical bar connecting node to both its offspring
        
    ax.plot([x,xp],[y,y],color=c,lw=branchWidth,zorder=98) ## plot horizontal branch to parent
    
ax.xaxis.tick_bottom() ## tick bottom
ax.yaxis.tick_left() ## tick left

xDates=['2013-%02d-01'%x for x in range(11,13)] ## create a timeline centered on each month
xDates+=['2014-%02d-01'%x for x in range(1,13)]
xDates+=['2015-%02d-01'%x for x in range(1,12)]

[ax.axvspan(decimalDate(xDates[x]),decimalDate(xDates[x])+1/float(12),facecolor='k',edgecolor='none',alpha=0.04) for x in range(0,len(xDates),2)] ## grey vertical bar every second month
ax.set_xticks([decimalDate(x)+1/24.0 for x in xDates]) ## x ticks in the middle of each month
ax.set_xticklabels([convertDate(x,'%Y-%m-%d','%b\n%Y') if x.split('-')[1]=='01' else convertDate(x,'%Y-%m-%d','%b') for x in xDates]) ## labels in mmm format unless January: then do YYYY-mmm

ax.spines['top'].set_visible(False) ## make axes invisible
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.tick_params(axis='x',labelsize=20,size=0) ## no axis labels visible except for timeline
ax.tick_params(axis='y',size=0)
ax.set_yticklabels([])

ax.set_xlim(decimalDate('2013-12-01'),decimalDate('2015-11-01')) ## bounds on axis limits
ax.set_ylim(-4,ll.ySpan+5)

# plt.savefig(local_output+'tree.png',dpi=200,bbox_inches='tight') ## save to file
# plt.savefig(local_output+'tree.pdf',dpi=200,bbox_inches='tight')
plt.show()

# ### Plot map
fig,ax2 = plt.subplots(figsize=(20,20),facecolor='w')

for i,loc in enumerate(locations): ## iterate over locations
    country=location_to_country[loc] ## identify country
    
    if country in required_countries: ## if country is to be plotted
        countryColour=colours[country] ## get colour map
        c=countryColour(normalized_coords[loc]) ## get colour based on location

        hatch=''
        if sum(cases_byLocation[loc].values())==0: ## hatching changes if location has no cases
            hatch='/'

        ax2.add_collection(PatchCollection(polygons[loc],facecolor=c,edgecolor='w',lw=1,zorder=1,hatch=hatch)) ## add location polygon collection

        lon,lat=popCentres[loc] ## get longitude/latitude of location's population centroid

        size=sum([k.length for k in ll.Objects if k.traits[traitName]==loc])**2 ## size of population centroid points depends on how long Ebola is inferred to have been in location

        ec='k'
        if size==0: ## white outline if lineages were never present
            ec='w'

        size=30+size
        ax2.scatter(lon,lat,size,facecolor=c,edgecolor=ec,lw=1,zorder=99,alpha=0.5) ## point at population centroid

        vas=['bottom','top'] ## define available text alignments and corrections for text positions
        has=['left','right']
        corrections=[0.01,-0.01]

        h=1 ## set default text alignment (right, top)
        v=1
        if textCorrection.has_key(loc): ## check if custom text positions are available
            h,v=textCorrection[loc]

        effects=[path_effects.Stroke(linewidth=4, foreground='white'),
                 path_effects.Stroke(linewidth=0.5, foreground='black')] ## black text, white outline

        ax2.text(lon+corrections[h],lat+corrections[v]*1.5,r'%s'%map_to_actual[loc],size=20,
                 va=vas[v],ha=has[h],alpha=1.0,path_effects=effects,zorder=100) ## plot district names at population centres, with corrections so as not to obscure it

for l,local_border in enumerate(global_border): ## plot the international borders
    ax2.plot(column(local_border,0),column(local_border,1),lw=2,color='w',zorder=97,label='%d_border'%(l))

ax2.set_aspect(1) ## we're close to the equator - enforce 1:1 aspect for longitude and latitude
ax2.spines['top'].set_visible(False) ## hide axes
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

ax2.tick_params(axis='x',size=0) ## no visible axis labels
ax2.tick_params(axis='y',size=0)
ax2.set_xticklabels([])
ax2.set_yticklabels([])

ax2.set_ylim(ylimits) ## bounds on plot
ax2.set_xlim(xlimits)

# plt.savefig(local_output+'tree_legend.png',dpi=300,bbox_inches='tight') ## save to file
# plt.savefig(local_output+'tree_legend.pdf',dpi=300,bbox_inches='tight')

plt.show()


fig,ax = plt.subplots(figsize=(20,20),facecolor='w')

traitName='location.states' ## name of locations trait in tree

travel_lineages=sorted([k for k in ll.Objects if k.parent!=ll.root and k.traits[traitName]!=k.parent.traits[traitName]],key=lambda x:x.absoluteTime) ## only interested in lineages travelling

xDates=['2013-%02d-01'%x for x in range(11,13)] ## create timeline
xDates+=['2014-%02d-01'%x for x in range(1,13)]
xDates+=['2015-%02d-01'%x for x in range(1,11)]

heights=[k.absoluteTime for k in travel_lineages] ## get absolute times of each branch in the tree
height_normalization=create_normalization([decimalDate(xDates[0]),decimalDate(xDates[-1])],0.0,1.0) ## create a normalization based on timeline, where earliest day is 0.0 and latest is 1.0

cmap=mpl.cm.get_cmap('viridis') ## colour map

for k in travel_lineages: ## iterate through lineages which have switched location
    locA=k.traits[traitName] ## get location of current lineage
    locB=k.parent.traits[traitName] ## get location of where it came from
    
    oriX,oriY=popCentres[locA] ## get population centroid coordinates
    desX,desY=popCentres[locB]
    
    normalized_height=height_normalization(k.absoluteTime) ## normalize heights of lineages
    normalized_parent_height=height_normalization(k.parent.absoluteTime)
    
    distance=math.sqrt(math.pow(oriX-desX,2)+math.pow(oriY-desY,2)) ## find travelling distance
    
    adjust_d=1-2*normalized_height+1/float(distance)**0.1-0.5 ## position Bezier curve control point according to an arbitrary function
    
    n=Bezier_control((oriX,oriY),(desX,desY),adjust_d) ## control point perpendicular to midway between point A and B at a distance adjust_d
    
    curve=Bezier([(oriX,oriY),n,(desX,desY)],0.0,1.0,num=30) ## get Bezier line coordinates
    
    for i in range(len(curve)-1): ## iterate through Bezier curve coordinates, alter colour according to height
        x1,y1=curve[i]
        x2,y2=curve[i+1]
        frac=i/float(len(curve)) ## fraction along line
        
        ax.plot([x1,x2],[y1,y2],lw=1+4*(1-frac),color=cmap(normalized_parent_height+(normalized_height-normalized_parent_height)*(1-frac)),zorder=int(normalized_height*10000)) ## curve tapers and changes colour

for i,loc in enumerate(locations): ## iterate over locations
    country=location_to_country[loc] ## get country
    
    if country in required_countries: ## if country required
        countryColour=colours[country] ## get colour map
        c=countryColour(totalCaseCounts[loc]/float(maxByCountry[country])) ## colour proportional to cases

        ax.add_collection(PatchCollection(polygons[loc],facecolor=countryColour(0.1),edgecolor='grey',lw=1,zorder=1)) ## polygon colour pale

        lon,lat=popCentres[loc] ## population centroid coordinates

        size=[k.traits[traitName] for k in ll.Objects].count(loc) ## circle size proportional to branches in location
        size=50+size
        ax.scatter(lon,lat,size,facecolor=c,edgecolor=desaturate(countryColour(1-(totalCaseCounts[loc]/float(maxByCountry[country]))),1.0),lw=2,zorder=200000) ## plot circle, edge coloured inversely from main colour

ycoord=np.mean([4.3,12.7]) ## add bar to indicate distance
legend_y=12.0
legend_x1=-15
legend_x2=-14.08059

ax.plot([legend_x1,legend_x2],[legend_y,legend_y],color='k',lw=6)
ax.text(np.mean([legend_x1,legend_x2]),legend_y+0.04,'%.0f km'%metricDistance((legend_x1,legend_y),(legend_x2,legend_y)),size=36,va='bottom',ha='center')

colorbarTextSize=30 ## add colourbars
colorbarTickLabelSize=24
colorbarWidth=0.02
colorbarHeight=0.4
colorbarIncrement=0.08

ax2 = fig.add_axes([colorbarIncrement*4, 0.13, colorbarHeight, colorbarWidth]) ## add dummy axes

mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=mpl.colors.Normalize([decimalDate(xDates[0]),decimalDate(xDates[-1])]),orientation='horizontal')
ax2.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=len(xDates))) ## add colour bar to axes

xaxis_labels=['' if (int(x.split('-')[1])+2)%3!=0 else convertDate(x,'%Y-%m-%d','%b\n%Y') if x.split('-')[1]=='01' else convertDate(x,'%Y-%m-%d','%b') for x in xDates] ## month if month+2 is divisible by 3 (January, April, July, October)
    
ax2.set_xticklabels(xaxis_labels) ## set colour bar tick labels
ax2.xaxis.set_label_position('top') ## colour bar label at the top
ax2.set_xlabel('date',color='k',size=colorbarTextSize) ## colour bar label is "date"
ax2.tick_params(labelcolor='k',size=10,labelsize=colorbarTickLabelSize) ## adjust axis parameters

for l,local_border in enumerate(global_border): ## plot the international borders
    ax.plot(column(local_border,0),column(local_border,1),lw=2,color='k',zorder=97)
    ax.plot(column(local_border,0),column(local_border,1),lw=6,color='w',zorder=96)

ax.set_aspect(1) ## aspect of 1 because we're close to the equator
ax.spines['top'].set_visible(False) ## invisible axes
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(size=0) ## invisible axis labels
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_ylim(ylimits) ## bounds on plot
ax.set_xlim(xlimits)

# plt.savefig(local_output+'geoTree.png',dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'geoTree.pdf',dpi=300,bbox_inches='tight')
plt.show()


normCmap=make_cmap([mpl.cm.Greys(x) for x in np.linspace(0.4,1.0,256)])

traitName='location.states'
travel_lineages=sorted([k for k in ll.Objects if k.parent!=ll.root and k.traits[traitName]!=k.parent.traits[traitName]],key=lambda x:x.absoluteTime)

partitioning=10 ## choose how many epi weeks to lump together
dates2=dates[:dates.index('2015-08-31')] ## only plot up to end of August

dateRange=range(0,len(dates2),partitioning) ## indices for lumping epiweeks

lumpy_maxima={country:0 for country in required_countries} ## will contain maximum number of cases in any given epiweek lump, split by country
for country in required_countries: ## iterate over countries
    for d,idx in enumerate(dateRange): ## iterate over lumps of epiweeks
        epiweeks=dates2[idx:idx+partitioning] ## fetch epiweeks
        
        for loc in cases_byLocation.keys(): ## iterate over locations
            c=location_to_country[loc] ## find country
            if c==country: ## if country matches
                case_lump=sum([cases_byLocation[loc][week] for week in epiweeks]) ## lump cases across epiweeks
                if case_lump>=lumpy_maxima[country]: ## if current location has more cases across epiweeks it becomes new maximum
                    lumpy_maxima[country]=case_lump
    
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cols=3 ## number of columns
N=len(dateRange) ## number of individual maps
rows=int(N/cols)+1 ## find the number of rows required to plot all lumps

print 'Number of cells to plot: %d'%(N)

scale=5 ## fig size scalar
plt.figure(figsize=(cols*scale,rows*scale),facecolor='w')

gs = gridspec.GridSpec(rows, cols,hspace=0.1,wspace=0.0,width_ratios=[1 for c in range(cols)],height_ratios=[1 for r in range(rows)]) ## gridspec for figure grid

epiweek_stepsize=decimalDate(dates2[1])-decimalDate(dates2[0]) ## find how long an epiweek is, should be around 7/365.0

timemin='2013-12-01' ## start time
timemax='2015-11-01' ## end time

height_normalization=create_normalization([decimalDate(timemin),decimalDate(timemax)],0.0,1.0) ## create height normalization where 0.0 is earliest and 1.0 is latest

for d,idx in enumerate(dateRange): ## iterate over epiweek lumps
    epiweeks=dates2[idx:idx+partitioning] ## fetch epiweeks in lump
    
    first=epiweeks[0]
    last=epiweeks[-1]
    
    treeStart=decimalDate(epiweeks[0]) ## tree starts at the beginning of the first epiweek in lump
    treeEnd=decimalDate(epiweeks[-1])+epiweek_stepsize ## finishes at the end of the last epiweek
    
    print d,idx,epiweeks
    
    cases_in_lump={loc:sum([cases_byLocation[loc][week] for week in epiweeks]) for loc in cases_byLocation.keys()} ## identify how many cases are in lump
    
    row=int(d/cols) ## current row is *quotient* when dividing lump index by number of available columns
    
    if row==0: ## current column is *remainder* when dividing lump index by number of available columns
        col=d
    else:
        col=d%cols

    height_normalization=create_normalization([treeStart,treeEnd],0.0,1.0) ## normalize heights to be within [0.0,1.0]
    
    ax = plt.subplot(gs[row, col]) ## fetch axes at the right cell in the grid
    
    for k in travel_lineages: ## iterate over travelling lineages
        if treeStart<np.mean([k.parent.absoluteTime,k.absoluteTime])<=treeEnd: ## if travelling lineage's mid-point is within interval - plot it
            locA=k.traits[traitName] ## end location
            locB=k.parent.traits[traitName] ## start location

            oriX,oriY=popCentres[locA]
            desX,desY=popCentres[locB]

            normalized_height=height_normalization(k.absoluteTime) ## normalize time of lineage
            normalized_parent_height=height_normalization(k.parent.absoluteTime)

            distance=math.sqrt(math.pow(oriX-desX,2)+math.pow(oriY-desY,2))
            adjust_d=1-2*normalized_height+1/float(distance)**0.1-0.5
            n=Bezier_control((oriX,oriY),(desX,desY),adjust_d)
            curve=Bezier([(oriX,oriY),n,(desX,desY)],0.0,1.0,num=30)

            midpoint=np.mean([normalized_parent_height,normalized_height])
            
            for i in range(len(curve)-1): ## iterate through Bezier curve coordinates, alter colour according to height
                x1,y1=curve[i]
                x2,y2=curve[i+1]
                frac=1-(i/float(len(curve)))

                ax.plot([x1,x2],[y1,y2],lw=1+2*frac,color=normCmap(midpoint),zorder=int(normalized_height*10000))
                ax.plot([x1,x2],[y1,y2],lw=1+4*frac,color='w',zorder=int(normalized_height*10000)-1)
            
    for i,loc in enumerate(locations):
        country=location_to_country[loc]
        
        if country in required_countries:
            countryColour=colours[country]
            c=countryColour(np.log10(cases_in_lump[loc])/float(np.log10(lumpy_maxima[country])))

            ax.add_collection(PatchCollection(polygons[loc],facecolor=c,edgecolor=countryColour(1.0),lw=1,zorder=1))

            lon,lat=popCentres[loc]

            size=[k.traits[traitName] for k in ll.Objects if k.parent.absoluteTime<=treeEnd and treeStart<=k.absoluteTime].count(loc)/2.0

            if size>0:
                size+=25
            ax.scatter(lon,lat,size,facecolor=c,edgecolor='k',lw=1,zorder=200000)
        
    ax.text(0.05,0.13,'%s'%(' - '.join(unique([first.split('-')[0],last.split('-')[0]]))),size=28,transform=ax.transAxes)
    ax.text(0.05,0.05,'%s - %s'%(convertDate(first,'%Y-%m-%d','%b %d'),convertDate(last,'%Y-%m-%d','%b %d')),size=24,transform=ax.transAxes)

    for l,local_border in enumerate(global_border):
        ax.plot(column(local_border,0),column(local_border,1),lw=1,color='k',zorder=97,label='%d_border'%(l))
        ax.plot(column(local_border,0),column(local_border,1),lw=2,color='w',zorder=96)
        
    colorbarTextSize=30
    colorbarTickLabelSize=20
    colorbarWidth=0.02
    colorbarHeight=0.35

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(size=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_ylim(ylimits)
    ax.set_xlim(xlimits)

# plt.savefig(local_output+'spatial.png',dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'spatial.pdf',dpi=300,bbox_inches='tight')
plt.show()


branchWidth=2
tipSize=30

traitName='location.states'
# traitName='location'
ll.root.traits[traitName]='reservoir' ## give root node some trait value that's different from what the actual tree root has, so it registers as a switch
ll.root.y=0
ll.root.x=ll.Objects[0].absoluteTime

location_to_country['reservoir']='?'
normalized_coords['reservoir']=1.0

tree_strings={'SLE':[],'GIN':[],'LBR':[]} ## will contain tree string derived from introductions into each country
country_trees={'SLE':[],'GIN':[],'LBR':[]}
loc_trees={}

subtree_sizes={loc:[] for loc in locations}
subtree_sizes['WesternArea']=[]
subtree_lengths={loc:[] for loc in locations}
subtree_lengths['WesternArea']=[]

output_subtrees=open(local_output+'subtrees.txt','w')
print>>output_subtrees,'location\tcountry\torigin location\torigin country\tclade TMRCA\tparent TMRCA\tclade size\tpersistence\ttree string'

output_members=open(local_output+'members.txt','w')

for l in sorted(ll.Objects,key=lambda x:x.height): ## iterate over branches
    k=l
    kp=l.parent
    
    kloc=k.traits[traitName]
    if k.parent.traits.has_key(traitName): ## get current branch's country and its parent's country
        kploc=kp.traits[traitName]
        kpc=location_to_country[kploc]
    else: ## if parent doesn't have a trait - dealing with root of tree
        kploc='reservoir'
        kpc='reservoir'

    kc=location_to_country[kloc]
    
    if kloc!=kploc: ## if locations don't match
        if isinstance(k,bt.node): ## and dealing with node
            N_children=len(k.leaves)
        else: ## dealing with leaf...
            N_children=1
        
        loc_subtree=ll.subtree(k,traitName=traitName) ## extract subtree during a within-trait traversal
        if loc_subtree: ## successful extraction (not None)
            loc_leaves=[x.name for x in loc_subtree.Objects if isinstance(x,bt.leaf)] ## get leaves in resulting subtree

            print 'location: %s to %s jump (ancestor %d, %d leaves in full tree, now has %d)'%(kploc,kloc,k.index,N_children,len(loc_leaves))
            if loc_trees.has_key(kloc):
                loc_trees[kloc].append((kploc,loc_subtree))
            else:
                loc_trees[kloc]=[(kploc,loc_subtree)]
            
            subtree_sizes[kloc].append(len(loc_leaves)) ## remember subtree size
    
    if kc!=kpc: ## if countries don't match
        subtree=ll.subtree(k,traitName=traitName,converterDict=location_to_country)
        
        if subtree: ## if subtree extracted
            if isinstance(k,bt.leaf): ## if dealing with a leaf that switched
                N_children=1
            else:
                N_children=len(k.leaves)
                
            subtree_leaves=[x.numName for x in subtree.Objects if isinstance(x,bt.leaf)]
            print 'country: %s (%s) to %s (%s) jump (ancestor %d, %d leaves in full tree, now has %d)'%(kpc,kploc,kc,kloc,k.index,N_children,len(subtree_leaves))
            
            subtree_lengths[kloc].append((k.length*0.5)+max([decimalDate(x.strip("'").split('|')[-1]) for x in subtree_leaves])-k.absoluteTime) ## remember time from mid-point of branch to last tip (subtree height)
            
            subtree.singleType() ## convert to single type tree in case multitype tree was returned
            tree_strings[kc].append(subtree.toString()) ## remember subtree string, subtree object itself
            country_trees[kc].append((kploc,subtree))

            mostRecentSubtreeTip=max([decimalDate(x.strip("'").split('|')[-1]) for x in subtree_leaves])
            persistence=mostRecentSubtreeTip-k.absoluteTime
            ancestorTime=k.parent.absoluteTime
            cladeTMRCA=k.absoluteTime
            originCountry=kpc
            originLoc=kploc
            cladeSize=len(subtree_leaves)

            subtreeString=subtree.toString()

            print>>output_subtrees,'%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s'%(kc,kloc,kpc,kploc,cladeTMRCA,ancestorTime,len(subtree_leaves),persistence,subtreeString)

            print>>output_members,'%s'%(','.join(subtree_leaves))


output_subtrees.close()
output_members.close()
print 'Done!'


fig,ax = plt.subplots(figsize=(20,10),facecolor='w')

xs=sorted(subtree_sizes.keys(),key=lambda x:(location_to_country[x],np.log10(sum(subtree_sizes[x]))),reverse=True) ## sort by country, then by number of introductions
# xs=sorted(subtree_sizes.keys(),key=lambda x:(location_to_country[x],-sum(subtree_sizes[x])))

xs=[x for x in xs if sum(cases_byLocation[x].values())>0]

for l,loc in enumerate(xs):
    country=location_to_country[loc]
    c=colours[country](normalized_coords[loc])

    ys=sorted(subtree_sizes[loc],reverse=True)
    for y,val in enumerate(ys):
        ax.bar(l,val,bottom=sum(ys[:y]),facecolor=c,edgecolor='w',lw=1,align='center') ## plot stacked bars of introduction sizes for each location
        
ax.set_xticks(range(len(xs)))
ax.set_xticklabels(xs,rotation=90)

ax.set_xlim(-0.5,len(xs)-0.5)
ax.set_ylabel('number of sequences',size=28)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.yaxis.tick_left()

ax.tick_params(axis='x',size=0)
ax.tick_params(axis='y',direction='out')
plt.show()


persistence_out=open(local_output+'EBOV_1610_loc_persistence.txt','w')
print>>persistence_out,'location\tclade sizes\tN introductions\tmedian persistence (days)\tmean persistence (days)\tstandard deviation\tpersistences (years)'
for loc in subtree_lengths.keys():
    L=len(subtree_lengths[loc])
    
    cS=map(str,subtree_sizes[loc])
    med=np.median(subtree_lengths[loc])*365
    mu=np.mean(subtree_lengths[loc])*365
    std=np.std(subtree_lengths[loc])
    
    if L>0:
        print loc,cS,L,med,mu,std
    print>>persistence_out,'%s\t%s\t%d\t%s\t%s\t%s\t%s'%(loc,','.join(cS),L,med,mu,std,','.join(map(str,subtree_lengths[loc])))
#     if len(subtree_lengths[loc])>0:
#         fig,ax = plt.subplots(figsize=(5,5),facecolor='w')
#         ax.hist([365*x for x in subtree_lengths[loc]],facecolor='steelblue')
#         ax.set_xlim(0,365)
#         plt.show()
persistence_out.close()


fig = plt.figure(figsize=(20,20),facecolor='w') ## set up whole figure
gs = gridspec.GridSpec(2, 1, height_ratios=[1,0.15],wspace=0.00,hspace=0.00) ## set up grid

# fig = plt.figure(figsize=(20,100),facecolor='w') ## for locations
# gs = gridspec.GridSpec(2, 1, height_ratios=[1,0.05],wspace=0.00,hspace=0.00) ## for locations

ax = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1],sharex=ax)

# traitName='location'
traitName='location.states'
branchWidth=2
tipSize=20

closure_dates={'SLE':'2014-06-11','GIN':'2014-08-09','LBR':'2014-07-27'} ## border closure dates
country_order={'GIN':3,'SLE':2,'LBR':1} ## order of countries

map_to_actual['WesternArea']='Western Area'

effects=[path_effects.Stroke(linewidth=4, foreground='white'),path_effects.Stroke(linewidth=0.5, foreground='black')] ## black text, white outline

cumulative_y=0
for tree_country in sorted(country_trees.keys(),key=lambda x:country_order[x]): ## iterate over countries
    for t,tr in enumerate(sorted(country_trees[tree_country],key=lambda x:(-x[1].root.absoluteTime,len(x[1].Objects)))): ## iterate over subtrees within the country in order of introduction
        origin,loc_tree=tr
        
        if len([ob for ob in loc_tree.Objects if ob.branchType=='leaf'])>0: ## if there's at least one leaf in subtree
            for w in loc_tree.Objects: ## iterate over subtree branches
                location=w.traits[traitName] ## get location
                country=location_to_country[location] ## get country from location
                cmap=colours[country] ## get colour map for country
                c=cmap(normalized_coords[location]) ## get standardised colour for location

                y=w.y
                x=w.absoluteTime
                
                if y!=None:
                    if w.parent!=None:
                        xp=w.parent.absoluteTime
                        yp=w.parent.y
                    else:
                        xp=x
                        yp=y
                    
                    if isinstance(w,bt.leaf): ## leaves have circles at tips
                        ax.scatter(x,y+cumulative_y,s=tipSize,facecolor=c,edgecolor='none',zorder=100) ## plot tip circle
                        ax.scatter(x,y+cumulative_y,s=tipSize+30,facecolor='k',edgecolor='none',zorder=99)

                    elif isinstance(w,bt.node): ## nodes have vertical lines
                        yl=w.children[0].y
                        yr=w.children[-1].y

                        ax.plot([x,x],[yl+cumulative_y,yr+cumulative_y],color=c,lw=branchWidth,zorder=98) ## plot vertical bar
                    
                    ax.plot([x,xp],[y+cumulative_y,y+cumulative_y],color=c,lw=branchWidth,zorder=98) ## plot branch

            oriC=colours[location_to_country[origin]](0.3)
            if loc_tree.Objects[0].absoluteTime==None:
                oriX=loc_tree.Objects[0].absoluteTime
                oriY=loc_tree.Objects[0].y+cumulative_y
            else:
                oriX=loc_tree.Objects[0].parent.absoluteTime
                oriY=loc_tree.Objects[0].y+cumulative_y

            if origin!='reservoir': ## add text if not dealing with first intro
                ax.text(oriX-7/365.0,oriY,'%s'%(map_to_actual[origin]),ha='right',va='center',
                        size=16,path_effects=effects) ## uncomment to plot text at the beginning of the subtree to indicate its origin, only use with enough vertical space
                
#                 ax.text(oriX-7/365.0,oriY,'%s'%(map_to_actual[origin]),ha='right',va='center',
#                         size=16) ## uncomment to plot text at the beginning of the subtree to indicate its origin, only use with enough vertical space
                
            ax.scatter(oriX,oriY,150,facecolor=oriC,edgecolor='w',lw=1,zorder=200) ## circle at the base of the subtree to indicate origin
            
            ## track when switching countries
            if t==0:
                rememberCumulative_y=cumulative_y
            
            cumulative_y+=max([x.y for x in loc_tree.Objects])+50 ## increment y position
            
            if t==len(country_trees[tree_country])-1:
                c=colours[tree_country](0.5)
                label_effects=[path_effects.Stroke(linewidth=5, foreground='w'),
                 path_effects.Stroke(linewidth=1.0, foreground=c)] ## country colour text, white outline
                
                rs=np.linspace(cumulative_y,rememberCumulative_y,50) ## country span line
                for r in range(len(rs)-1):
                    ax.plot([decimalDate('2013-12-01')]*2,[rs[r],rs[r+1]],lw=6,color=colours[country](r/float(len(rs)-1)),zorder=9) ## plot segment with fading colour
                ax.text(decimalDate('2013-12-01'),
                        (cumulative_y+rememberCumulative_y)/2.0,'%s'%(translate[tree_country]),
                        size=42,rotation=90,ha='right',va='center',
                        path_effects=label_effects,zorder=10) ## name of country
                
#                 ax.text(decimalDate('2013-12-01'),
#                         (cumulative_y+rememberCumulative_y)/2.0,'%s'%(translate[tree_country]),
#                         size=42,rotation=90,ha='right',va='center',zorder=10) ## name of country
                

####################    
epoch_path=path_to_dropbox+'Sequences/Jun2016_1610_genomes/GLM/epoch/changePointEstimates.csv'

xs=[]
ymu=[]
yhi=[]
ylo=[]
ps=[]

axx=ax2.twinx() ## secondary y axis

for line in open(epoch_path,'r'): ## load epoch data
    l=line.strip('\n').split(',') 
    if l[0]=='month':
        header=l
    else:
        d=decimalDate(l[0],fmt='%b-%y')
        prob,meanC,hpLo,hpHi=map(float,l[1:])
        xs.append(d)
        ps.append(prob)
        ymu.append(meanC)
        yhi.append(hpHi)
        ylo.append(hpLo)
        
ax2.bar(xs,ps,width=1/24.0,align='center',facecolor='steelblue',lw=1,edgecolor='k')
axx.plot([x+1/24.0 for x in xs],ymu,color='indigo',lw=3)
axx.fill_between([x+1/24.0 for x in xs],ylo,yhi,facecolor='indigo',edgecolor='none',alpha=0.1)
axx.plot([x+1/24.0 for x in xs],ylo,color='indigo',ls='--',alpha=1.0,lw=2)
axx.plot([x+1/24.0 for x in xs],yhi,color='indigo',ls='--',alpha=1.0,lw=2)
ax2.set_ylim(0,1)
axx.set_ylim(0,4)

ax2.spines['top'].set_visible(False)
axx.spines['top'].set_visible(False)

ax2.tick_params(axis='x',labelsize=26,size=0)
ax2.tick_params(axis='y',labelsize=20,size=5,direction='out')
axx.tick_params(axis='x',size=0)
ax2.set_xticklabels([])
axx.tick_params(axis='y',labelsize=16,size=5,direction='out')

axx.set_ylabel('coefficient')
ax2.set_ylabel('change point probability')  

for country in required_countries: ## plot border closures in all axes
    ax2.axvline(decimalDate(closure_dates[country]),lw=3,color=colours[country](0.5),zorder=3)
    ax2.axvline(decimalDate(closure_dates[country]),lw=6,color='w',zorder=2)

    axx.axvline(decimalDate(closure_dates[country]),lw=3,color=colours[country](0.5),zorder=3)
    axx.axvline(decimalDate(closure_dates[country]),lw=6,color='w',zorder=2)

    ax.axvline(decimalDate(closure_dates[country]),lw=3,color=colours[country](0.5),zorder=3)
    ax.axvline(decimalDate(closure_dates[country]),lw=6,color='w',zorder=2)
#####################

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

every=2
xDates=['2013-%02d-01'%x for x in range(11,13)]
xDates+=['2014-%02d-01'%x for x in range(1,13)]
xDates+=['2015-%02d-01'%x for x in range(1,12)]

ax.set_xlim(decimalDate(xDates[0]),decimalDate(xDates[-1]))
ax2.set_xlim(decimalDate(xDates[0]),decimalDate(xDates[-1]))

[ax2.axvspan(decimalDate(xDates[x]),decimalDate(xDates[x])+1/float(12),facecolor='k',edgecolor='none',alpha=0.04) for x in range(0,len(xDates),2)]
[ax.axvspan(decimalDate(xDates[x]),decimalDate(xDates[x])+1/float(12),facecolor='k',edgecolor='none',alpha=0.04) for x in range(0,len(xDates),2)]
ax.set_xticks([decimalDate(x)+1/24.0 for x in xDates if (int(x.split('-')[1])-1)%2==0])
ax2.set_xticks([decimalDate(x)+1/24.0 for x in xDates if (int(x.split('-')[1])-1)%2==0])
axx.set_xticks([decimalDate(x)+1/24.0 for x in xDates if (int(x.split('-')[1])-1)%2==0])

ax2.set_xticklabels([convertDate(x,'%Y-%m-%d','%b\n%Y') if x.split('-')[1]=='01' else convertDate(x,'%Y-%m-%d','%b') for x in xDates if (int(x.split('-')[1])-1)%2==0])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.tick_params(axis='x',size=0)
ax.tick_params(axis='y',size=0)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_ylim(-30,cumulative_y+80)
# plt.savefig(local_output+'trees.png',dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'trees.pdf',dpi=300,bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(10,27),facecolor='w')
gs = gridspec.GridSpec(1, 1,wspace=0.00,hspace=0.00)
ax = plt.subplot(gs[0])

traitName='location.states'
branchWidth=2
tipSize=20

closure_dates={'SLE':'2014-06-11','GIN':'2014-08-09','LBR':'2014-07-27'}
country_order={'GIN':3,'SLE':2,'LBR':1}
earliestIntro={}

cumulative_y=0
storeDestination=''
ylabels=[]

for loc in loc_trees.keys():
    earliestIntro[loc]=min([y[1].Objects[0].absoluteTime for y in loc_trees[loc] if y[1].Objects[0].traits[traitName]==loc])

for loc in sorted(loc_trees.keys(),key=lambda x:(country_order[location_to_country[x]],-earliestIntro[x])): ## iterate over each location, sorted by country and date of first introduction

    locTree=loc_trees[loc]
    for t,tr in enumerate(locTree): ## iterate over trees
        origin,loc_tree=tr
        destination=loc_tree.Objects[0].traits[traitName]
        originTime=loc_tree.Objects[0].absoluteTime ## origin date of clade
        loc_leaves=[w for w in loc_tree.Objects if w.branchType=='leaf']
        cladeSize=float(len(loc_leaves))
        countryColour=colours[location_to_country[origin]]
        c=countryColour(normalized_coords[origin])
        lastTip=max([decimalDate(w.name.split('|')[-1]) for w in loc_leaves]) ## final tip date
        persistence=lastTip-originTime ## persistence is clade start - clade end (0 for tips)
        
        if storeDestination!=destination: ## if moving to next location - increment
            cumulative_y+=1
            ylabels.append(destination)

        radius=np.sqrt(cladeSize/np.pi)*40.0

        jitter=np.random.uniform(-0.4,0.4) ## jitter points vertically
        ax.scatter(np.mean([lastTip,originTime]),cumulative_y+jitter,s=radius,facecolor=c,edgecolor='k',zorder=10001,alpha=1.0) ## add clade size (area proportional to number of leaves in clade)
        
        ax.plot([originTime,lastTip],[cumulative_y+jitter,cumulative_y+jitter],
                lw=3,alpha=1.0,color=c,zorder=10000-cladeSize-1,solid_capstyle='round') ## add persistence line
        
        storeDestination=destination
ylabels.append(destination)

popPath=open('/Users/evogytis/Dropbox/Ebolavirus_Phylogeography/Location_GLM/sparks/sparky.csv','r')
popSizes={}
for line in popPath:
    l=line.strip('\n').split(',')
    if l[0]!='location':
        popSizes[l[0]]=float(l[-2])
popSizes['WesternArea']=popSizes['WesternUrban']+popSizes['WesternRural']
    
ax.set_yticks(range(1,len(ylabels)))
ax.set_yticklabels(ylabels)

ax.scatter([decimalDate('2013-12-01') for y in range(1,len(ylabels))],range(1,len(ylabels)),s=[10+((3.0*popSizes[loc]-3.0*min(popSizes.values()))/np.pi)**0.5 for loc in ylabels],c=[colours[location_to_country[loc]](normalized_coords[loc]) for loc in ylabels]) ## plot location pop sizes

ax.tick_params(axis='y',size=0)
ax.grid(axis='x',ls='--',color='grey',zorder=0)

[ax.axhspan(x-0.5,x+0.5,facecolor='k',edgecolor='none',alpha=0.08,zorder=0) for x in range(0,len(ylabels)+2,2)]

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

every=2
xDates=['2013-%02d-01'%x for x in range(11,13)]
xDates+=['2014-%02d-01'%x for x in range(1,13)]
xDates+=['2015-%02d-01'%x for x in range(1,12)]

ax.set_xticks([decimalDate(x) for x in xDates if (int(x.split('-')[1])-1)%2==0])
ax.set_xticklabels([convertDate(x,'%Y-%m-%d','%b\n%Y') if x.split('-')[1]=='01' else convertDate(x,'%Y-%m-%d','%b') for x in xDates if (int(x.split('-')[1])-1)%2==0])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_xlim(decimalDate(xDates[0]),decimalDate(xDates[-1]))
ax.set_ylim(0.5,len(ylabels)-0.5)

for country in required_countries:
    ax.axvline(decimalDate(closure_dates[country]),lw=3,color=colours[country](0.5),zorder=2)
    ax.axvline(decimalDate(closure_dates[country]),lw=6,color='w',zorder=1)

# plt.savefig(local_output+'persistences.png',dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'persistences.pdf',dpi=300,bbox_inches='tight')
plt.show()


fig,ax = plt.subplots(figsize=(20,100),facecolor='w')

# traitName='location'
traitName='location.states'
branchWidth=2
tipSize=20

closure_dates={'SLE':'2014-06-11','GIN':'2014-08-09','LBR':'2014-07-27'} ## border closure dates
country_order={'GIN':3,'SLE':2,'LBR':1} ## order of countries

map_to_actual['WesternArea']='Western Area'

cumulative_y=0
unroll_trees=[item for sublist in loc_trees.values() for item in sublist]

sorted_trees=sorted(unroll_trees,key=lambda x:(country_order[location_to_country[x[1].Objects[0].traits[traitName]]],-x[1].root.absoluteTime))
for subtree in sorted_trees: ## iterate over countries
    origin,loc_tree=subtree

    if len([ob for ob in loc_tree.Objects if ob.branchType=='leaf'])>0: ## if there's at least one leaf in subtree
        for w in loc_tree.Objects: ## iterate over subtree branches
            location=w.traits[traitName] ## get location
            country=location_to_country[location] ## get country from location
            cmap=colours[country] ## get colour map for country
            c=cmap(normalized_coords[location]) ## get standardised colour for location

            y=w.y
            x=w.absoluteTime

            if y!=None:
                if w.parent!=None:
                    xp=w.parent.absoluteTime
                    yp=w.parent.y
                else:
                    xp=x
                    yp=y

                if isinstance(w,bt.leaf):
                    ax.scatter(x,y+cumulative_y,s=tipSize,facecolor=c,edgecolor='none',zorder=100) ## plot tip circle
                    ax.scatter(x,y+cumulative_y,s=tipSize+30,facecolor='k',edgecolor='none',zorder=99)

                elif isinstance(w,bt.node):
                    yl=w.children[0].y
                    yr=w.children[-1].y

                    ax.plot([x,x],[yl+cumulative_y,yr+cumulative_y],color=c,lw=branchWidth,zorder=98) ## plot vertical bar

                ax.plot([x,xp],[y+cumulative_y,y+cumulative_y],color=c,lw=branchWidth,zorder=98) ## plot ancestral branch

        oriC=colours[location_to_country[origin]](0.3)
        if loc_tree.Objects[0].absoluteTime==None:
            oriX=loc_tree.Objects[0].absoluteTime
            oriY=loc_tree.Objects[0].y+cumulative_y
        else:
            oriX=loc_tree.Objects[0].parent.absoluteTime
            oriY=loc_tree.Objects[0].y+cumulative_y

        if origin!='reservoir':
            effects=[path_effects.Stroke(linewidth=4, foreground='white'),
             path_effects.Stroke(linewidth=0.5, foreground='black')] ## black text, white outline

            ax.text(oriX-7/365.0,oriY,'%s'%(map_to_actual[origin]),ha='right',va='center',size=16,path_effects=effects) ## uncomment to plot text at the beginning of the subtree to indicate its origin, only use with enough vertical space

        ax.scatter(oriX,oriY,150,facecolor=oriC,edgecolor='w',lw=1,zorder=200) ## circle at the base of the subtree to indicate origin

        if sorted_trees.index(subtree)==0:
            rememberCountry=country
            rememberCumulative_y=0
            
        if rememberCountry!=country or sorted_trees.index(subtree)==len(sorted_trees)-1:
            c=colours[rememberCountry](0.5)
            effects=[path_effects.Stroke(linewidth=5, foreground='w'),
             path_effects.Stroke(linewidth=1.0, foreground=c)] ## black text, white outline

            rs=np.linspace(cumulative_y,rememberCumulative_y,50)
            for r in range(len(rs)-1):
                ax.plot([decimalDate('2013-12-01')]*2,[rs[r],rs[r+1]],lw=6,color=colours[rememberCountry](r/float(len(rs)-1)),zorder=9)
            ax.text(decimalDate('2013-12-01'),(cumulative_y+rememberCumulative_y)/2.0,'%s'%(translate[rememberCountry]),size=42,rotation=90,ha='right',va='center',path_effects=effects,zorder=10)
            
            rememberCumulative_y=cumulative_y
            
        rememberCountry=country
        cumulative_y+=max([x.y for x in loc_tree.Objects])+50 ## increment y position

for country in required_countries: ## plot border closures in all axes
    ax.axvline(decimalDate(closure_dates[country]),lw=3,color=colours[country](0.5),zorder=3)
    ax.axvline(decimalDate(closure_dates[country]),lw=6,color='w',zorder=2)

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

every=2
xDates=['2013-%02d-01'%x for x in range(11,13)]
xDates+=['2014-%02d-01'%x for x in range(1,13)]
xDates+=['2015-%02d-01'%x for x in range(1,12)]

ax.set_xlim(decimalDate(xDates[0]),decimalDate(xDates[-1]))

[ax.axvspan(decimalDate(xDates[x]),decimalDate(xDates[x])+1/float(12),facecolor='k',edgecolor='none',alpha=0.04) for x in range(0,len(xDates),2)]
ax.set_xticks([decimalDate(x)+1/24.0 for x in xDates if (int(x.split('-')[1])-1)%2==0])
ax.set_xticklabels([convertDate(x,'%Y-%m-%d','%b\n%Y') if x.split('-')[1]=='01' else convertDate(x,'%Y-%m-%d','%b') for x in xDates if (int(x.split('-')[1])-1)%2==0])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.tick_params(axis='x',labelsize=26,size=0)
ax.tick_params(axis='y',size=0)
ax.set_yticklabels([])

ax.set_ylim(-30,cumulative_y+80)
# plt.savefig(local_output+'locTrees.png',dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'locTrees.pdf',dpi=300,bbox_inches='tight')

plt.show()


# traitName='location'
traitName='location.states'
branchWidth=2
tipSize=30

ll.root.traits[traitName]='reservoir'
ll.root.y=0
ll.root.x=ll.Objects[0].absoluteTime

location_to_country['reservoir']='?'
normalized_coords['reservoir']=1.0

for k in ll.Objects:
    #print k.traits,k.parent.traits
    kloc=k.traits[traitName]
    kploc=k.parent.traits[traitName]
    if kploc!=kloc:
        if kploc=='reservoir':
            subtree=copy.deepcopy(ll.traverseWithinTrait(k,traitName))
            
local_tree=bt.tree() ## create a new tree object where the subtree will be
local_tree.Objects=subtree ## assign objects

local_tree.root.children.append(subtree[0]) ## connect objects with the root
subtree[0].parent=local_tree.root
local_tree.root.absoluteTime=subtree[0].absoluteTime-subtree[0].length

local_tree.sortBranches() ## sort branches (also draws tree)

local_tree.root.x=local_tree.Objects[0].x
local_tree.root.y=local_tree.Objects[0].y

N_times={x:0 for x in locations}
for q in local_tree.Objects:
    if isinstance(q,bt.node):
        toReplace=[]
        for ch,child in enumerate(q.children):
            childTrait=child.traits[traitName]
            parentTrait=q.traits[traitName]
            if childTrait!=parentTrait:
                toReplace.append(child)
                
        for child in toReplace:
            childTrait=child.traits[traitName]
            print q.index,parentTrait,childTrait
            N_times[childTrait]+=1
            q.children.remove(child)

            fake_leaf=bt.leaf()
            if N_times[childTrait]==1:
                if childTrait=='Conakry':
                    fake_leaf.name='%s (%s)\nGN-1 lineage'%(childTrait,location_to_country[childTrait])
                elif childTrait=='Kailahun':
                    fake_leaf.name='%s (%s), SL lineages'%(childTrait,location_to_country[childTrait])
                else:
                    fake_leaf.name='%s (%s)'%(childTrait,location_to_country[childTrait])
            else:
                fake_leaf.name='%s#%d (%s)'%(childTrait,N_times[childTrait],location_to_country[childTrait])
            fake_leaf.numName=fake_leaf.name
            fake_leaf.traits[traitName]=childTrait
            fake_leaf.index='%s%s%s'%(np.random.random(),np.random.random(),np.random.random()) ## generate random index
            fake_leaf.length=child.length
            fake_leaf.absoluteTime=q.absoluteTime+fake_leaf.length
            fake_leaf.height=q.height+fake_leaf.length
            
            fake_leaf.parent=q
            q.children.append(fake_leaf)
            q.leaves=[]
            q.numChildren=0
            local_tree.Objects.append(fake_leaf)

output_pruned=open(local_output+'Fig2_initialStages.source.tree','w')
print>>output_pruned,'#NEXUS\nBegin trees;\ntree TREE1 = [&R] %s\nEnd;'%(local_tree.toString(traits=[traitName,'posterior','%s.set'%(traitName),'%s.set.prob'%(traitName)]))
output_pruned.close()


root_path=path_to_dropbox+'/Sequences/Aug2016_1610_genomes/Joint/HomogenousGLM/All_1610/Run1_first_100M_used_in_revision1/Makona_1610_cds_ig.GLM.tmrcas' ## load tmrca distributions

burnin=10000000
for line in open(root_path,'r'):
    l=line.strip('\n').split('\t')
    if l[0]=='state':
        header=l
        param_idx={x:i for i,x in enumerate(header)}
        params=header[1:]
        rootHeights={x:[] for x in params}
    elif float(l[0])>=burnin:
        for param in params:
            rootHeights[param].append(float(l[param_idx[param]]))

print [np.mean(x) for x in rootHeights.values()]
print [np.median(x) for x in rootHeights.values()]
print [hpd(x,0.95) for x in rootHeights.values()]

fig = plt.figure(figsize=(20, 15)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5],wspace=0.01) 

ax = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

# fig,ax = plt.subplots(figsize=(10,15),facecolor='w')
tipSize=40
branchWidth=3

local_tree.traverse_tree()
local_tree.sortBranches(descending=True)
effects=[path_effects.Stroke(linewidth=4, foreground='w'),path_effects.Stroke(linewidth=0.5, foreground='k')]

offsetTree=10
for w in local_tree.Objects:
    location=w.traits[traitName]
    country=location_to_country[location]
    cmap=colours[country]
    c=cmap(normalized_coords[location])
    y=w.y+offsetTree
    x=w.absoluteTime

    if y!=None:
        if w.parent!=None:
            xp=w.parent.absoluteTime
            yp=w.parent.y
        else:
            xp=x
            yp=y

        if isinstance(w,bt.leaf):
            ls='-'
            if w.traits[traitName]!='Gueckedou':
                s=16
                if w.name[-1]!=')':
                    s=20
#                     ax.text(x+0.005,y,'%s'%(w.name),size=s,ha='left',va='center',path_effects=effects)
                    ax.text(x+0.005,y,'%s'%(w.name),size=s,ha='left',va='center')
                else:
                    ax.text(x+0.005,y,'%s'%(w.name),size=s,ha='left',va='center')
                ls='--'
            else:
                ax.scatter(x,y,s=tipSize,facecolor=c,edgecolor='none',zorder=100)
                ax.scatter(x,y,s=tipSize*2,facecolor='k',edgecolor='none',zorder=99)
            ax.plot([x,xp],[y,y],color=c,lw=branchWidth,ls=ls,zorder=98)
            
        elif isinstance(w,bt.node):
            yl=w.children[0].y+offsetTree
            yr=w.children[-1].y+offsetTree
            
            if w.traits['posterior']>=0.5:
                ax.scatter(x,y,s=70,facecolor='w',edgecolor=c,zorder=100)
                if len(w.leaves)>=5 and w.parent.index!='Root':
                    vas=['bottom','top']
                    va=vas[w.parent.children.index(w)-1]
                    ax.text(x-2/365.0,y+((w.parent.children.index(w)-1)*0.2),'%.2f'%(w.traits['posterior']),
                            va=va,ha='right',size=16,path_effects=effects,zorder=101)
#                     ax.text(x-2/365.0,y+((w.parent.children.index(w)-1)*0.2),'%.2f'%(w.traits['posterior']),
#                             va=va,ha='right',size=16,zorder=101)
                    
            
            if w.parent==local_tree.root:
#                 location_states=w.traits['%s.states.set'%(traitName)]
#                 location_probs=w.traits['%s.states.set.prob'%(traitName)]
                location_states=w.traits['%s.set'%(traitName)]
                location_probs=w.traits['%s.set.prob'%(traitName)]
                join_probs={a:b for a,b in zip(location_states,location_probs)}
                sort_locations=sorted(join_probs.keys(),key=lambda a:join_probs[a])
                print [(a,'%.2f'%join_probs[a]) for a in sort_locations]
                width=0.05
                heightRange=yr-yl
                start=yl
                for loc in sort_locations:
                    height=heightRange*join_probs[loc]
                    #print loc,'%.2f'%(join_probs[loc])
                    country=location_to_country[loc]
                    fc=colours[country](normalized_coords[loc])
                    
                    if join_probs[loc]>=0.2:
                        ax.text(x-width*0.5,np.mean([start,start+height]),'%s'%(map_to_actual[loc]),rotation=90,zorder=101,
                                va='center',ha='center',path_effects=effects)
#                         ax.text(x-width*0.5,np.mean([start,start+height]),'%s'%(map_to_actual[loc]),rotation=90,zorder=101,
#                                 va='center',ha='center')
                    ax.add_patch(plt.Rectangle((x-width,start),width,height,facecolor=fc,edgecolor='k',lw=1,zorder=100))
                    start+=height
                
            ax.plot([x,xp],[y,y],color=c,lw=branchWidth,zorder=98)
            ax.plot([x,x],[yl,yr],color=c,lw=branchWidth,zorder=98)

for intro in rootHeights.keys():
    hpdLo,hpdHi=hpd(rootHeights[intro],0.95)

    x_grid=np.linspace(hpdLo,hpdHi,100)
    kde=gaussian_kde(rootHeights[intro],0.3)
    y_grid=kde.evaluate(x_grid)

    root_y=[(y*0.6)-1 for y in y_grid]
    
    if intro=='Root':
        c='k'
        #intro='reservoir'
    else:
        c=colours[location_to_country[intro]](normalized_coords[intro])

    if intro!='Root':
        topX,topY=[(k.absoluteTime,k.y+offsetTree) for k in local_tree.Objects if k.traits[traitName]==intro][-1]
    else:
        topX,topY=local_tree.Objects[0].absoluteTime,local_tree.Objects[0].children[0].y+offsetTree
        
    bottomX=topX
    bottomY=(kde.evaluate(bottomX)*0.6)-1
        
    ax.plot([bottomX,topX],[bottomY,topY],ls=':',color=c)
    
    ax.fill_between(x_grid,root_y,y2=-1,facecolor=c,edgecolor='none',alpha=0.4)
    ax.plot(x_grid,root_y,lw=2,color=c,ls='-')
          

every=1
xDates=['2013-%02d-01'%x for x in range(1,13)]
xDates+=['2014-%02d-01'%x for x in range(1,12)]
# xDates+=['2015-%02d-01'%x for x in range(1,12)]


[ax.axvspan(decimalDate(xDates[x]),decimalDate(xDates[x])+1/float(12),facecolor='k',edgecolor='none',alpha=0.04) for x in range(0,len(xDates),2)]
ax.set_xticks([decimalDate(x)+1/24.0 for x in xDates if (int(x.split('-')[1])-1)%every==0])
ax.set_xticklabels([convertDate(x,'%Y-%m-%d','%b\n%Y') if x.split('-')[1]=='01' else convertDate(x,'%Y-%m-%d','%b') for x in xDates if (int(x.split('-')[1])-1)%every==0])
    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.tick_params(axis='y',size=0)
ax.tick_params(axis='x',labelsize=18,size=0)
ax.set_yticklabels([])

ax.set_xlim(decimalDate('2013-12-01'),decimalDate(xDates[-1]))
ax.set_ylim(-1,(len(local_tree.Objects)+1)/2.0+offsetTree+1)

for loc in locations:
    if location_to_country[loc] in required_countries:
        countryColour=colours[location_to_country[loc]]
        c=countryColour(normalized_coords[loc])
        ax2.add_collection(PatchCollection(polygons[loc],facecolor=c,edgecolor='w',lw=1,zorder=1))


normalize_height=create_normalization([x.parent.absoluteTime for x in local_tree.Objects if len(x.parent.traits)>0 and x.traits[traitName]!=x.parent.traits[traitName]],0.0,1.0)

destinationXs=[]
destinationYs=[]
destinations=['Gueckedou']
for w in local_tree.Objects:
    if len(w.parent.traits)>0:
        locA=w.traits[traitName]
        locB=w.parent.traits[traitName]
        if locA!=locB:
            destinations.append(locA)
            country=location_to_country[location]
            cmap=colours[country]
            c=cmap(normalized_coords[location])
            y=w.y
            x=w.absoluteTime

            if y!=None:
                if w.parent!=None:
                    xp=w.parent.absoluteTime
                    yp=w.parent.y
                else:
                    xp=x
                    yp=y

            end=w.height
            start=w.parent.height

            oriX,oriY=popCentres[locB]
            desX,desY=popCentres[locA]

            destinationXs.append(desX)
            destinationYs.append(desY)
            
            ## normalize time of lineage
            normalized_height=normalize_height(w.absoluteTime)
            normalized_parent_height=normalize_height(w.parent.absoluteTime)
            #print normalized_height,normalized_parent_height
            ## define Bezier curve
            distance=math.sqrt(math.pow(oriX-desX,2)+math.pow(oriY-desY,2))

            ## adjust_d is the function that determines where the Bezier line control point will be
#             adjust_d=distance-(distance**0.5)*normalized_height
            adjust_d=0.1-2*normalized_height+1/float(distance)**0.1-0.1

            ## control point coordinate set perpendicular to midway between point A and B at a distance adjust_d
            n=Bezier_control((oriX,oriY),(desX,desY),adjust_d)

            ## get Bezier line coordinates
            curve=Bezier([(oriX,oriY),n,(desX,desY)],0.0,1.0,num=40)

            midpoint=np.mean([normalized_parent_height,normalized_height])

            ## iterate through Bezier curve coordinates, alter colour according to height
            for i in range(len(curve)-1):
                x1,y1=curve[i]
                x2,y2=curve[i+1]
                frac=(i/float(len(curve)))

                ax2.plot([x1,x2],[y1,y2],lw=4+2*frac,color=desaturate(mpl.cm.Oranges(frac),0.8),
                         zorder=int(normalized_height*10000),solid_capstyle='round')
                ax2.plot([x1,x2],[y1,y2],lw=6+4*frac,color='w',zorder=int(normalized_height*10000)-1,solid_capstyle='round')
        
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

ax2.tick_params(size=0)
ax2.set_xticklabels([])
ax2.set_yticklabels([])

## identify the longest axis
frameLength=max([max(destinationXs)-min(destinationXs),max(destinationYs)-min(destinationYs)])

for loc in destinations:
    ## define available text alignments and corrections for text positions
    vas=['bottom','top']
    has=['left','right']
    corrections=[0.01,-0.01]

    ## set default text alignment (right, top)
    h=1
    v=1
    ## check if custom text positions are available
    if textCorrection.has_key(loc):
        if loc=='Kissidougou' or loc=='Siguiri' or loc=='Gueckedou':
            h=1
            v=0
        elif loc=='Conakry':
            h=0
            v=1
        elif loc=='Macenta':
            h=0
            v=1

    effects=[path_effects.Stroke(linewidth=4, foreground='white'),path_effects.Stroke(linewidth=0.5, foreground='black')]
    
    lon,lat=popCentres[loc]
    ## plot district names at population centres, with corrections so as not to obscure it
    ax2.text(lon+corrections[h],lat+corrections[v]*1.5,r'%s'%map_to_actual[loc],size=34,
             va=vas[v],ha=has[h],alpha=1.0,path_effects=effects,zorder=100000)
#     ax2.text(lon+corrections[h],lat+corrections[v]*1.5,r'%s'%map_to_actual[loc],size=34,
#              va=vas[v],ha=has[h],alpha=1.0,zorder=100000)


ax2.set_xlim(min(destinationXs)-0.3,max(destinationXs)+0.7)
ax2.set_ylim(min(destinationYs)-2.4,max(destinationYs)+1.15)
ax2.set_aspect(1)

for l,local_border in enumerate(global_border):
    ax2.plot(column(local_border,0),column(local_border,1),lw=2,color='k',zorder=97,label='%d_border'%(l))
    ax2.plot(column(local_border,0),column(local_border,1),lw=5,color='w',zorder=96)

# plt.savefig(local_output+'InitialStages.png',dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'InitialStages.pdf',dpi=300,bbox_inches='tight')

plt.show()


output_countrees=open(local_output+'Fig3_trees.source.tree','w')
print>>output_countrees,'#NEXUS\nBegin trees;\n'

for country in sorted(country_trees.keys(),key=lambda x:country_order[x]): ## iterate over countries
    for t,tr in enumerate(sorted(country_trees[country],key=lambda x:(-x[1].root.absoluteTime,len(x[1].Objects)))): 
        origin,loc_tree=tr
        tree_string=loc_tree.toString(traits=['posterior',traitName,'%s.set'%(traitName),'%s.set.prob'%(traitName)])
        if len(loc_tree.Objects)>1:
            print>>output_countrees,'tree %s_%s = [&R] %s'%(location_to_country[origin],t+1,tree_string)
        else:
            print 'singleton tree:',tree_string

print>>output_countrees,'End;'
output_countrees.close()


branchWidth=2
tipSize=30

traitName='location.states'
# traitName='location'
ll.root.traits[traitName]='reservoir' ## give root node some trait value that's different from what the actual tree root has, so it registers as a switch
ll.root.y=0
ll.root.x=ll.Objects[0].absoluteTime

dummy=copy.deepcopy(ll)

location_to_country['reservoir']='?'
normalized_coords['reservoir']=1.0

loc_trees={}

components=[]

for l in sorted(dummy.Objects,key=lambda x:x.height): ## iterate over branches
    k=l
    kp=l.parent
    
    kloc=k.traits[traitName]
    if k.parent.traits.has_key(traitName): ## get current branch's country and its parent's country
        kploc=kp.traits[traitName]
        kpc=location_to_country[kploc]
    else:
        kploc='reservoir'
        kpc='reservoir'

    kc=location_to_country[kloc]
    if kc!=kpc:
#     if kloc!=kploc: ## if locations don't match
        proceed=False ## assume we still can't proceed forward
        
        if isinstance(k,bt.leaf): ## if dealing with a leaf - proceed
            N_children=1
            proceed=True
        else:
            N_children=len(k.leaves)
#             if [ch.traits[traitName] for ch in k.children].count(kloc)>0: ## if locations don't match
            if [location_to_country[ch.traits[traitName]] for ch in k.children].count(kc)>0: ## if locations don't match
                proceed=True
        
        subtree=copy.deepcopy(dummy.traverseWithinTrait(k,traitName,location_to_country))
        subtree_leaves=[x.name for x in subtree if isinstance(x,bt.leaf)]
        
        if len(subtree_leaves)>0 and proceed==True: ## if at least one valid tip and no hanging nodes
            print '%s (%s) to %s (%s) jump (ancestor %d, %d leaves in full tree, now has %d)'%(kpc,kploc,kc,kloc,k.index,N_children,len(subtree_leaves))
            
            orderedTips=sorted([w for w in subtree if w.branchType=='leaf'],key=lambda x:x.absoluteTime)
            lastTip=orderedTips[-1] ## identify most recent tip
            firstTip=orderedTips[0]
            cur_node=firstTip

            while cur_node: ## while not at root
                if cur_node==firstTip: ## if at the beginning of descent
                    if k.branchType=='node': ## and node
                        cladeOb=bt.clade(firstTip.numName) ## create a clade object
                        cladeOb.length=firstTip.length
                        cladeOb.index=firstTip.index
                        cladeOb.absoluteTime=firstTip.parent.absoluteTime
                        cladeOb.lastAbsoluteTime=lastTip.absoluteTime
                        cladeOb.traits=k.traits
                        cladeOb.width=np.log(len(subtree_leaves)+2)
                        cladeOb.leaves=subtree_leaves
                        cladeOb.parent=firstTip.parent
                        firstTip.parent.children.append(cladeOb) ## replace tip with clade object
                        firstTip.parent.children.remove(firstTip)
                        cur_node=cladeOb
                    
                components.append(cur_node) ## add descent line to components
                cur_node=cur_node.parent ## continue descent
            
            components=list(set(components)) ## only keep unique branches

print 'Done!'


fig,ax = plt.subplots(figsize=(15,15),facecolor='w')

traitName='location.states'

tipSize=10 ## tip circle radius
branchWidth=4 ## line width for branches

posteriorCutoff=0.0 ## posterior cutoff if collapsing tree

# plot_tree=bt.tree()
c=list(set(components))
print len(c)
d=copy.deepcopy(dummy)
for k in sorted(d.Objects,key=lambda x:-x.height):
    if k.index not in [w.index for w in c]:
        d.Objects.remove(k) ## remove branches not in components
        k.parent.children.remove(k)
        
    elif k.branchType=='leaf' and k.index in [w.index for w in c if isinstance(w,bt.clade)]:
        d.Objects.remove(k)
        r=[w for w in c if isinstance(w,bt.clade) and w.index==k.index][0] ## replace leaves with clade objects
        r.parent=k.parent
        k.parent.children.append(r)
        k.parent.children.remove(k)
        d.Objects.append(r)
                    

d.sortBranches()
plot_tree=d

for k in plot_tree.Objects: ## iterate over branches in the tree
    location=k.traits[traitName] ## get inferred location of branch
    country=location_to_country[location] ## find country of location
    cmap=colours[country] ## fetch colour map for country
#     c=cmap(normalized_coords[location]) ## get colour of location
    c=cmap(0.5) ## get colour of location
    y=k.y ## get y coordinates
    yp=k.parent.y ## get parent's y coordinate
    
    x=k.absoluteTime ## x coordinate is absolute time
    xp=k.parent.absoluteTime ## get parent's absolute time
    
    convert=lambda x:np.sqrt(x)/np.pi
    
    if k.branchType=='leaf': ## if tip...
        if isinstance(k,bt.leaf): ## if really tip
            ax.scatter(x,y,s=30+tipSize,facecolor=c,edgecolor='none',zorder=100) ## put a circle at each tip
            ax.scatter(x,y,s=30+tipSize*1.2+30,facecolor='k',edgecolor='none',zorder=99)
        else: ## if clade
#             clade=plt.Polygon(([x-0.05*k.length,y-0.0001*len(plot_tree.Objects)],[x-0.05*k.length,y+0.0001*len(plot_tree.Objects)],[k.lastAbsoluteTime,y+k.width/2.0],[k.lastAbsoluteTime,y-k.width/2.0]),facecolor=c,edgecolor='none',zorder=12)
#             ax.add_patch(clade) ## add triangle
            #ax.text(k.lastAbsoluteTime+0.01,y,len(k.leaves),va='center',ha='left')
            ax.plot([k.lastAbsoluteTime,xp],[y,y],color=c,lw=branchWidth,zorder=98)
            ax.scatter(k.lastAbsoluteTime,y,s=30+len(k.leaves)*tipSize,facecolor=c,edgecolor='none',zorder=100)
            ax.scatter(k.lastAbsoluteTime,y,s=30+len(k.leaves)*tipSize*1.2+30,facecolor='k',edgecolor='none',zorder=99)
            
    elif isinstance(k,bt.node): ## if node...
        yl=k.children[0].y ## get y coordinates of first and last child
        yr=k.children[-1].y
        
        if xp==0.0:
            xp=x

        ls='-'
        if k.traits['posterior']<posteriorCutoff: ## change to dotted line if posterior probability too low
            ls='--'
        ax.plot([x,x],[yl,yr],color=c,lw=branchWidth,ls=ls,zorder=98) ## plot vertical bar connecting node to both its offspring
        
    ax.plot([x,xp],[y,y],color=c,lw=branchWidth,zorder=98) ## plot horizontal branch to parent
    
ax.xaxis.tick_bottom() ## tick bottom
ax.yaxis.tick_left() ## tick left

xDates=['2013-%02d-01'%x for x in range(11,13)] ## create a timeline centered on each month
xDates+=['2014-%02d-01'%x for x in range(1,13)]
xDates+=['2015-%02d-01'%x for x in range(1,12)]

[ax.axvspan(decimalDate(xDates[x]),decimalDate(xDates[x])+1/float(12),facecolor='k',edgecolor='none',alpha=0.04) for x in range(0,len(xDates),2)] ## grey vertical bar every second month
ax.set_xticks([decimalDate(x)+1/24.0 for x in xDates]) ## x ticks in the middle of each month
ax.set_xticklabels([convertDate(x,'%Y-%m-%d','%b\n%Y') if x.split('-')[1]=='01' else convertDate(x,'%Y-%m-%d','%b') for x in xDates]) ## labels in mmm format unless January: then do YYYY-mmm

ax.spines['top'].set_visible(False) ## make axes invisible
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.tick_params(axis='x',labelsize=20,size=0) ## no axis labels visible except for timeline
ax.tick_params(axis='y',size=0)
ax.set_yticklabels([])

ax.set_xlim(left=decimalDate('2014-01-01')) ## bounds on axis limits
ax.set_ylim(-4,plot_tree.ySpan+5)

plt.show()


get_ipython().magic('matplotlib inline')
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import gridspec
import matplotlib.patheffects as path_effects
from IPython.display import HTML

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from ebov_data import *

if locations:
    pass
else:
    status()
    setFocusCountries(['SLE','LBR','GIN'])
    setColourMaps()
    loadData()

typeface='Helvetica Neue' ## set default matplotlib font and font size
mpl.rcParams['font.weight']=300
mpl.rcParams['axes.labelweight']=300
mpl.rcParams['font.family']=typeface
mpl.rcParams['font.size']=22

def kde_scipy( vals1, vals2, (a,b), (c,d), N ):
    """ Performs 2D kernel density estimation.
        vals1, vals2 are the values of two variables (columns)
        (a,b) interval for vals1 over which to estimate first axis KDE
        (c,d) -"-          vals2 over which to estimate second axis KDE
        N is number of equally spaced points over which to estimate KDE.
     """
    x=np.linspace(a,b,N)
    y=np.linspace(c,d,N)
    X,Y=np.meshgrid(x,y)
    positions = np.vstack([Y.ravel(), X.ravel()])

    values = np.vstack([vals1, vals2])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return [x, y, Z]

xml_path=path_to_dropbox+'Sequences/Aug2016_1610_genomes/Joint/HomogenousGLM/All_1610/Run2/Makona_1610_cds_ig.joint_GLM.xml' ## path to XML

lpaths=[path_to_dropbox+'Sequences/Aug2016_1610_genomes/Joint/HomogenousGLM/All_1610/Run%d/Makona_1610_cds_ig.GLM.log'%(run) for run in range(1,3)]
hpaths=[path_to_dropbox+'Sequences/Aug2016_1610_genomes/Joint/HomogenousGLM/All_1610/Run%d/Makona_1610_cds_ig.GLM.history.log'%(run) for run in range(1,3)]

tip_dates=[]
for line in open(xml_path,'r'): 
    cerberus=re.search('<taxon id="[A-Za-z\_\|0-9?]+\|([0-9\-]+)',line) ## collect all sequence collection dates
    if cerberus is not None:
        tip_dates.append(cerberus.group(1))

mostRecent=max(map(decimalDate,tip_dates)) ## find most recent one

matrix_index={} ## predictor matrix indexing
origin_indices={x:[] for x in locations}
destination_indices={x:[] for x in locations}

lg=len(locations)
for i in range(len(locations)):
    for j in range(i+1,len(locations)):
        
        locA=locations[i]
        countryA=location_to_country[locA]
        
        locB=locations[j]
        countryB=location_to_country[locB]
        
        f1=int((lg*(lg-1)/2) - (lg-i)*((lg-i)-1)/2 + j - i - 1) ## indexing of a flattened GLM matrix
        f2=int((lg*(lg-1)) - (lg-i)*((lg-i)-1)/2 + j - i - 1)

        matrix_index[f1]=(locations[i],locations[j])
        matrix_index[f2]=(locations[j],locations[i])
        
        origin_indices[locA].append(f1)
        origin_indices[locB].append(f2)
        
        destination_indices[locB].append(f1)
        destination_indices[locA].append(f2)

predictor_description={} ## this will contain the predictor description taken from the comment before the GLM matrix

read_matrix=False ## flag to read the matrices

predictors=[] ## this will contain the predictor name
description=''

read_loc=False ## flag to read locations

xml_districts=[] ## this will contain location names

counter=1
store=''
print 'predictors found:'
for line in open(xml_path,'r'): ## iterate through XML by line
    descriptionRegex='<!-- predictor [0-9]+\: ([A-Za-z\_\.\(\)\-0-9\, +>]+) -->'
    matrixRegex='<parameter id="([A-Za-z0-9\_\.]+)" value="'
    
    matrixID=re.search(matrixRegex,line) ## search for matrix
    valid_description=re.search(descriptionRegex,store) ## look at whether previous line was a valid description of a predictor
    if matrixID is not None and valid_description is not None:
        predictor=matrixID.group(1)
        predictors.append(predictor)
        predictor_description[predictor]=description
        print ' & %s & %s \\\\ \\hline'%(predictor,description)
        
        counter+=1

    descriptionID=re.search(descriptionRegex,line) ## search for matrix description
    if descriptionID is not None:
        description=descriptionID.group(1)
    
    if 'location.dataType' in line: ## identify when to start reading location names
        read_loc=True

    cerberus=re.search('<state code="([A-Za-z]+)"/>',line) ## log location
    if read_loc==True and cerberus is not None:
        xml_districts.append(cerberus.group(1))
    
    store=line ## remember line for next iteration
    
burnin=10000000 ## define burnin for GLM log file, identify the name of the trait
trait='location'

required_coeffs=['%s.glmCoefficients%d'%(trait,x+1) for x in range(len(predictors))] ## identify what the indicator and coefficient names in the log file will be
required_indicators=['%s.coefIndicator%d'%(trait,x+1) for x in range(len(predictors))]

GLM_coeffs={x:[] for x in predictors} ## create a dict of lists that will contain posterior samples
GLM_indicators={x:[] for x in predictors}

for log_path in lpaths:
    for line in open(log_path,'r'): ## iterate through the log file
        l=line.strip('\n').split('\t')
        if l[0]=='state':
            header=l
            indices_coeffs=[i for i,x in enumerate(header) if x in required_coeffs]
            indices_indicators=[i for i,x in enumerate(header) if x in required_indicators]
        elif '#' in line:
            pass

        elif int(l[0])>=burnin: ## start logging posterior states past the burnin
            for i,j,x in zip(indices_coeffs,indices_indicators,predictors): ## iterate through indices where indicators and coefficients of known predictors will be
                GLM_indicators[x].append(float(l[j]))
                GLM_coeffs[x].append(float(l[i]))

frame='<iframe style="border: 0; width: 400px; height: 308px;" src="https://bandcamp.com/EmbeddedPlayer/album=2789340638/size=large/bgcol=333333/linkcol=e99708/artwork=small/track=1338962038/transparent=true/" seamless><a href="http://vilkduja.bandcamp.com/album/insomnia-ep">Insomnia EP by Vilkduja</a></iframe>'

priorProbability=1-math.pow(0.5,(1/float(len(predictors)))) ##### calculates prior odds
priorOdds=float(priorProbability/float(1-priorProbability))

BFs={} ## BFs for individual indicators being on
print '\npredictor analysis:'
print '%3s%30s%5s%13s%4s%9s'%('idx','predictor','N','ln coeff','pp','BF')

GLM_conditional_coeffs={}

for i,x in enumerate(predictors):
    L=len(GLM_indicators[x])
    if L==0:
        GLM_coeffs.pop(x,None)
        GLM_indicators.pop(x,None)
    else:
        MCMClen=L
        support=np.mean(GLM_indicators[x])
        conditioned_coeff=[a for a,b in zip(GLM_coeffs[x],GLM_indicators[x]) if b==1.0]
        GLM_conditional_coeffs[x]=conditioned_coeff
        posteriorOdds=(((support-(1/float(MCMClen)))/float((1-(support-(1/float(MCMClen)))))))
        BFs[x]=posteriorOdds/float(priorOdds)
        note=' '
        if BFs[x]>3.0:
            note='*'
        print '%3s%s%30s%6d%9.2f%8.2f%11.2f'%(i+1,note,x,len(GLM_coeffs[x]),np.mean(conditioned_coeff),support,BFs[x])

#########################################################################################
#### IMPORT PREDICTOR MATRICES    #######################################################
#########################################################################################

within_country=[]
between_countries=[]

distance_matrix={x:{y:0 for y in popCentres.keys()} for x in popCentres.keys()} ## find distance in kilometres between every location population centroid
for x in popCentres.keys():
    pointA=popCentres[x]
    for y in popCentres.keys():
        pointB=popCentres[y]
        distance_matrix[x][y]=metricDistance(pointA,pointB)

all_times=[]
all_distances=[]

within={x:[] for x in required_countries}
between={x:{y:[] for y in required_countries if x!=y} for x in required_countries}
country_jump_to={x:{} for x in required_countries}
country_jump_from={x:{} for x in required_countries}

jump_matrix={x:{y:0 for y in popCentres.keys() if x!=y} for x in popCentres.keys()}
posteriors=[]
country_jump_matrix={x:{y:{} for y in required_countries} for x in required_countries}
location_jump_matrix={x:{y:{} for y in locations} for x in locations}


jumps_to={y:[] for y in popCentres.keys()}
jumps_from={y:[] for y in popCentres.keys()}

MCMClen=0

# burnin=10000000
burnin=20000000
for history_path in hpaths:
    for line in open(history_path,'r'): ## iterate through the history file
        l=line.strip('\n').split('\t')
        if '#' in line or 'state' in line:
            pass
        elif float(l[0])>=burnin:
            MCMClen+=1
            try:
                N_transitions=int(float(l[-1]))
            except:
                N_transitions=int(float(line.split(' ')[-1].strip('\n')))

            cerberus=re.findall('{[0-9]+,[0-9\.]+,[A-Za-z]+,[A-Za-z]+}',l[2]) ## fetch all transitions

            local_matrix={x:{y:0 for y in popCentres.keys() if x!=y} for x in popCentres.keys()}

            withins=0
            betweens=0

            for trans in cerberus: ## iterate over each event
                log,height,origin,destination=trans[1:-1].split(',')

                height=float(height)

                countryA=location_to_country[origin] ## get country for start and end locations
                countryB=location_to_country[destination]

                pointA=popCentres[origin]
                pointB=popCentres[destination]

                distance=distance_matrix[origin][destination]
                absoluteTime=mostRecent-height

                all_times.append(absoluteTime) ## remember time of transition

                all_distances.append(distance) ## remember distance

                jump_matrix[origin][destination]+=1 ## add to number of known transitions between locations

                jumps_to[destination].append(absoluteTime) ## add time to known jumps to and from the locations
                jumps_from[origin].append(absoluteTime)

                if countryA!=countryB: ## if jump is between countries - remember time and distance in the between category
                    betweens+=1
                    between[countryA][countryB].append((absoluteTime,distance))

                    if country_jump_from[countryA].has_key(MCMClen):
                        country_jump_from[countryA][MCMClen]+=1
                    else:
                        country_jump_from[countryA][MCMClen]=1

                    if country_jump_to[countryB].has_key(MCMClen):
                        country_jump_to[countryB][MCMClen]+=1
                    else:
                        country_jump_to[countryB][MCMClen]=1  

                elif countryA==countryB: ## otherwise - in within category
                    withins+=1
                    within[countryA].append((absoluteTime,distance))

                if location_jump_matrix[origin][destination].has_key(MCMClen):
                    location_jump_matrix[origin][destination][MCMClen].append(absoluteTime)
                else:
                    location_jump_matrix[origin][destination][MCMClen]=[absoluteTime]

                if country_jump_matrix[countryA][countryB].has_key(MCMClen):
                    country_jump_matrix[countryA][countryB][MCMClen].append(absoluteTime)
                else:
                    country_jump_matrix[countryA][countryB][MCMClen]=[absoluteTime]

            if len(cerberus)!=N_transitions: ## make sure that the number of found jumps matches what BEAST reported
                print 'Number of transitions found (%d) does not match reported number (%d) at state %s'%(len(cerberus),N_transitions,l[0])

            posteriors.append(withins/float(betweens)) ## add ratio of within to between jumps to a separate list

location_jump_matrix['WesternArea']={loc:{} for loc in locations}
for destination in locations:
    for mcmc in range(MCMClen):
        urban=[]
        rural=[]
        if location_jump_matrix['WesternRural'][destination].has_key(mcmc):
            rural=location_jump_matrix['WesternRural'][destination][mcmc]
        if location_jump_matrix['WesternUrban'][destination].has_key(mcmc):
            urban=location_jump_matrix['WesternUrban'][destination][mcmc]
            
        location_jump_matrix['WesternArea'][destination][mcmc]=urban+rural
        
        urban=[]
        rural=[]
        if location_jump_matrix[destination]['WesternRural'].has_key(mcmc):
            rural=location_jump_matrix[destination]['WesternRural'][mcmc]
        if location_jump_matrix[destination]['WesternUrban'].has_key(mcmc):
            urban=location_jump_matrix[destination]['WesternUrban'][mcmc]
        location_jump_matrix[destination]['WesternArea']={}
        location_jump_matrix[destination]['WesternArea'][mcmc]=urban+rural
        
print 'Done! (%d MCMC states loaded)'%(MCMClen)
HTML(frame)


fig = plt.figure(figsize=(15, 10)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1],wspace=0.01) ## setup figure with two columns

ax = plt.subplot(gs[0]) ## violins go into this subplot
ax2 = plt.subplot(gs[1]) ## horizontal inclusion probability bars go into this subplot

leftedge=0 ## these will provide x axis limits later on
rightedge=0

ax2.set_xlabel('inclusion probability',size=24) ## set x labels for both figures
ax.set_xlabel('coefficient',size=34)

ax2.xaxis.set_label_position('top')
ax.xaxis.set_label_position('top')
ax.xaxis.labelpad=10

plotBFs=[3,15,50] ## plot inclusion probabilities for BF=3, 15 and 50
cutoffs={}
for bf in plotBFs:
    posteriorOdds=priorOdds*bf
    cutoffs[bf]=posteriorOdds/(1+posteriorOdds)
    print '%d Bayes factor cut-off = %.4f'%(bf,cutoffs[bf])
    ax2.axvline(cutoffs[bf],color='k',lw=1,ls='--')
    ax2.text(cutoffs[bf],-0.5,'BF=%d'%(bf),size=22,ha='center',va='top',rotation=90)

predictors=sorted(GLM_coeffs.keys(),key=lambda x:(np.mean(GLM_indicators[x]),np.mean(GLM_conditional_coeffs[x]))) ## sort predictor names by support, then by coefficient

significant=[x for x in GLM_conditional_coeffs.keys() if np.mean(GLM_indicators[x])>=priorOdds*min(plotBFs)]
significant=sorted(significant,key=lambda x:(np.mean(GLM_indicators[x]),np.mean(GLM_coeffs[x])))
print significant

for i,x in enumerate(significant): ## for each predictor plot violins     
    support=np.mean(GLM_indicators[x])
    
    if support>=min(cutoffs.values()): ## if BF >minimum then plot coefficients conditional on it being turned on
        posterior_samples=[val for val,ind in zip(GLM_coeffs[x],GLM_indicators[x]) if ind==1.0] ## conditioned
#         posterior_samples=[val*ind for val,ind in zip(GLM_coeffs[x],GLM_indicators[x])] ## product of coefficient and indicator ("actual rate")
    else:
        posterior_samples=GLM_coeffs[x] ## otherwise plot all samples
        
    w=0.4 ## width of violins
    
    k1 = gaussian_kde(posterior_samples) #calculates the kernel density
    
    mu=np.mean(posterior_samples)
    m1,M1=hpd(posterior_samples,0.95) ## get HPDs
    
    if x in significant:
        #print predictor_description[x]
        posteriorOdds=(((support-(1/float(MCMClen)))/float((1-(support-(1/float(MCMClen)))))))
        BF=posteriorOdds/float(priorOdds)
        if BF>50.0:
            BF='%15s'%('>50')
        else:
            BF='%15.2f'%(BF)
        print '%s\n%40s\t%.2f [%.2f, %.2f]\t%s\t%.4f'%(predictor_description[x],x,mu,m1,M1,BF,support)
    
    if m1<=leftedge: ## define new x axis minimum if new HPD is lower than current one
        leftedge=m1
    if M1>=rightedge:
        rightedge=M1

    x1 = np.linspace(m1,M1,100)  ## create a range of 100 values between lower and upper HPDs
    v1 = k1.evaluate(x1)  ## evaluate KDEs at each of the 100 values
    v1 = v1/v1.max()*w ## rescale to fit inside defined violin width

    supportFrac=min([1.0,support/float(min(cutoffs.values()))]) ## ratio of inclusion probability and smallest BF required, limited to 1.0
    
    if M1<0.0 and m1<0.0: ## define fill colours for violins depending on whether they're inside or outside HPDs
        fcolour='#BA2F46'
    elif M1>0.0 and m1>0.0:
        fcolour=desaturate(mpl.cm.Greens(0.6),0.6)
    if support<1.0:
        fcolour='grey'
     
    ax.fill_between(x1,[i+q for q in v1],[i-q for q in v1],facecolor=desaturate(fcolour,supportFrac),edgecolor='none',alpha=supportFrac,zorder=100) ## plot violin, colour is desaturated depending on inclusion probability
    
    fcolour='k'
    ax.plot(x1,[i+w for w in v1],color=desaturate(fcolour,supportFrac),lw=2,alpha=1,zorder=100) ## plot nice edges for each violin (upper and lower)
    ax.plot(x1,[i-w for w in v1],color=desaturate(fcolour,supportFrac),lw=2,alpha=1,zorder=100)
    
    if i%2==0: ## even numbered predictor - add a grey shaded area in the background
        ax.axhspan(i-0.5,i+0.5,facecolor='k',edgecolor='none',alpha=0.05,zorder=0)
    
    ax2.barh(i,support,height=0.93,lw=2,align='center',edgecolor='none',
             facecolor=desaturate('steelblue',supportFrac)) ## plot inclusion probability

ylabels=[]
for pred in significant: ## define descriptions for each predictor to be used as y axis labels
    break_at=3
    desc=predictor_description[pred]
    
    if len(desc.split(', +1'))>1: ## rectify description of predictor matrices
        description=''.join(desc.split(', +1')[:-1])
    elif len(desc.split('pseudo'))>1:
        description=''.join(desc.split(', pseudo')[:-1])
    elif len(desc.split('ln-'))>1:
        description=''.join(desc.split(', ln')[:-1])
    elif len(desc.split(',  0'))>1:
        description=''.join(desc.split(',  0')[:-1])
    else:
        description=desc
    
    break_description='\n'.join([' '.join([y for y in description.split(' ')[x:x+break_at]]) for x in range(0,len(description.split(' ')),break_at)]) ## breaks up the description into new lines to fit nicely
    
    ylabels.append(r'%s'%(break_description))
    
ax.axvline(0,ls='--',lw=1,color='k') ## add a horizontal line to main plot at coeff=0
ax.grid(axis='x',ls=':')

ax.spines['right'].set_color('none') ## make plot pretty
ax.spines['left'].set_color('none')

ax.yaxis.tick_left()

ax.set_yticks(np.arange(0,len(predictors)))
ax.set_yticklabels(ylabels,size=20)

for tick in ax.get_yticklabels():
    tick.set_size(40-np.log10(len(list(tick.get_text())))*8)

ax2.xaxis.tick_top()
ax2.set_xticks(np.linspace(0,1,3))
ax2.set_xticklabels(np.linspace(0,1,3),rotation=90)
ax2.tick_params(axis='y',size=0)
ax2.set_yticklabels([])
ax2.tick_params(axis='x',size=5,labelsize=22,direction='out',pad=10)

ax2.spines['top'].set_color('none')
ax2.spines['bottom'].set_color('none')

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax.tick_params(axis='x',labelsize=26,direction='out')
ax.tick_params(axis='y',size=0)

ax.set_ylim(-0.5,len(significant)-0.5)
ax2.set_ylim(-0.5,len(significant)-0.5)
ax.set_xlim(leftedge-0.2,rightedge+0.2)
ax2.set_xlim(0,1)

# plt.savefig(local_output+'coeff.png',dpi=100,bbox_inches='tight')
# plt.savefig(local_output+'coeff.pdf',dpi=100,bbox_inches='tight')
plt.show()


# # Analysis of jump counts
# 

loc_subset=[loc for loc in locations if location_to_country[loc] in required_countries]

empty=np.zeros((len(loc_subset),len(loc_subset))) ## empty matrix
empty.fill(np.nan)

sorted_locations=sorted(loc_subset,key=lambda x:(location_to_country[x],normalized_coords[x])) ## sort locations by country, normalized coordinate within country
analysis=pd.DataFrame(empty,index=sorted_locations,columns=sorted_locations)

for i in loc_subset:
    for j in loc_subset:
        if i!=j:
            analysis[i][j]=jump_matrix[i][j]/float(MCMClen) ## calculate posterior number of jumps

fig,ax = plt.subplots(figsize=(20,20),facecolor='w') ## start figure

masked_array = np.ma.array(np.array(analysis),mask=np.isnan(analysis)) # mask NaNs

cmap=mpl.cm.viridis ## colour map
cmap.set_bad('grey',1.)

norm=mpl.colors.LogNorm(vmin=10.0**-2,vmax=10.0**2) ## normalize within range
heatmap = ax.pcolor(masked_array,edgecolors='none', linewidths=0,cmap=cmap,alpha=1,norm=norm) ## heatmap

ax.set_yticks(np.arange(0.5,len(sorted_locations)+0.5))
ax.set_yticklabels(sorted_locations)

ax.set_xticks(np.arange(0.5,len(sorted_locations)+0.5))
ax.set_xticklabels(sorted_locations,rotation=90)

ax.set_xlabel('destination',size=30)
ax.set_ylabel('origin',size=30)
ax.tick_params(size=0,labelsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

countries_of_labels=[location_to_country[x] for x in sorted_locations]
for c in range(len(countries_of_labels)-1):
    if countries_of_labels[c]!=countries_of_labels[c+1]:
        ax.axhline(c+1,ls='--',color='w') ## delimit countries on x and y axes
        ax.axvline(c+1,ls='--',color='w')

axcb = fig.add_axes([0.93, 0.15, 0.02, 0.6], frame_on=False) ## colour bar
cb = mpl.colorbar.ColorbarBase(axcb,cmap=cmap,norm=norm,orientation='vertical',alpha=1.0,drawedges=False)

axcb.yaxis.set_label_position("left")
axcb.set_ylabel('mean of transitions across posterior',size=30)
axcb.tick_params(axis='y',which='both',direction='out',size=12,width=1,pad=10)
plt.setp(axcb.get_yticklabels(),size=32,name=typeface)

for tick in axcb.yaxis.get_ticklines():
    tick.set_markersize(10)

for tick in ax.yaxis.get_ticklabels():
    label=str(tick.get_text())
    tick.set_color(desaturate(colours[location_to_country[label]](normalized_coords[label]),0.8))

# plt.savefig(local_output+'EBOV_transition_matrix.png',dpi=200,bbox_inches='tight')
# plt.savefig(local_output+'EBOV_transition_matrix.pdf',dpi=200,bbox_inches='tight')
plt.show()


for category in ['within','between','all']: ## iterate over categories
    flux={}
    log_flux={}
    
    for i in xml_districts: ## count transitions to and from each location
        transitions_within=[x for x in popCentres.keys() if location_to_country[x]==location_to_country[i] and i!=x]
        transitions_between=[x for x in popCentres.keys() if location_to_country[x]!=location_to_country[i]]

        if category=='within':
            froms=sum([jump_matrix[i][x] for x in transitions_within])
            tos=sum([jump_matrix[x][i] for x in transitions_within if x!=i])

        elif category=='between':
            froms=sum([jump_matrix[i][x] for x in transitions_between])
            tos=sum([jump_matrix[x][i] for x in transitions_between if x!=i])

        elif category=='all':
            froms=sum(jump_matrix[i].values())
            tos=sum([jump_matrix[x][i] for x in jump_matrix.keys() if x!=i])

        flux[i]=froms/float(tos) ## divide sum of origin transitions by sum of destination transitions
        log_flux[i]=np.log10(froms/float(tos)) ## log the ratio

    effects=[path_effects.Stroke(linewidth=5, foreground='white'),path_effects.Stroke(linewidth=1, foreground='black')]
    
    for kind in [flux,log_flux]: ## iterate over regular and logged ratios
        fig,ax = plt.subplots(figsize=(20,10),facecolor='w') ## start figure
        
        sorted_flux=sorted(xml_districts,key=lambda x:-kind[x]) ## sort locations by value
        bar_colours=[desaturate(colours[location_to_country[label]](normalized_coords[label]),0.8) for label in sorted_flux]

        adjust=0.6
        plotted=adjust
        for i in range(len(sorted_flux)): ## iterate over sorted locations
            loc=sorted_flux[i]
            val=kind[loc]
            fc=bar_colours[i]
            fc=colours[location_to_country[loc]](0.5)
            ec='none'
            lw=2

            if val!=-np.inf and np.log10(val)!=-np.inf: ## if location value is valid (not 0 in log space)
                ax.bar(i+adjust,val,facecolor=fc,edgecolor=ec,width=0.9,lw=lw,align='center') ## plot bar

                label_loc='bottom'
                yloc=0.1
                if val>0.0:
                    label_loc='top'
                    yloc=-1*yloc

                ax.text(i+adjust,yloc,'%s'%(map_to_actual[loc]),rotation=90,size=22,va=label_loc,ha='center',
                        color='k',path_effects=effects) ## add location name to bar

                plotted+=1

        ax.set_xlim(0,plotted)
        ax.tick_params(axis='x',size=0)
        ax.set_xticklabels([])
        ax.tick_params(axis='y',labelsize=26,direction='out',pad=0)
        ax.yaxis.tick_left()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.text(0.05,0.95,'%s countries'%(category),size=30,transform=ax.transAxes,path_effects=effects) ## label plot

        if min(kind.values())<0.0:
            ax.set_ylabel('log$_{10}$ origin/destination transitions',size=32)
            ax.axhline(0.0,color='k',lw=1,ls='--')
#             plt.savefig(local_output+'EBOV_%s_country_logTransitions.png'%(category),dpi=300,bbox_inches='tight')
#             plt.savefig(local_output+'EBOV_%s_country_logTransitions.pdf'%(category),dpi=300,bbox_inches='tight')

        else:
            ax.set_ylabel('origin/destination transitions',size=32)
            ax.axhline(1.0,color='k',lw=1,ls='--')
#             plt.savefig(local_output+'EBOV_%s_country_transitions.png'%(category),dpi=300,bbox_inches='tight')
#             plt.savefig(local_output+'EBOV_%s_country_transitions.pdf'%(category),dpi=300,bbox_inches='tight')

        plt.show()


effects=[path_effects.Stroke(linewidth=4, foreground='white'),path_effects.Stroke(linewidth=0.5, foreground='black')]

for category in ['within','between','all']: ## iterate over categories
    
    fluxFrom={}
    log_fluxFrom={}

    fluxTo={}
    log_fluxTo={}
    
    for i in xml_districts: ## iterate over locations, separate out jumps as within or between countries
        transitions_within=[x for x in popCentres.keys() if location_to_country[x]==location_to_country[i] and i!=x]
        transitions_between=[x for x in popCentres.keys() if location_to_country[x]!=location_to_country[i]]

        if category=='within':
            froms=sum([jump_matrix[i][x] for x in transitions_within])/float(MCMClen)
            tos=sum([jump_matrix[x][i] for x in transitions_within if x!=i])/float(MCMClen)

        elif category=='between':
            froms=sum([jump_matrix[i][x] for x in transitions_between])/float(MCMClen)
            tos=sum([jump_matrix[x][i] for x in transitions_between if x!=i])/float(MCMClen)

        elif category=='all':
            froms=sum(jump_matrix[i].values())/float(MCMClen)
            tos=sum([jump_matrix[x][i] for x in jump_matrix.keys() if x!=i])/float(MCMClen)

        fluxTo[i]=float(tos)
        log_fluxTo[i]=np.log10(float(tos))

        fluxFrom[i]=float(froms)
        log_fluxFrom[i]=np.log10(float(froms))

    for kind in [[fluxFrom,fluxTo],[log_fluxFrom,log_fluxTo]]: ## iterate over normal and log space origin-destination pairs
        fig,ax = plt.subplots(figsize=(20,10),facecolor='w') ## start figure

        origin,destination=kind
#         sorted_flux=sorted(sorted_locations,key=lambda x:-destination[x]-origin[x])
#         sorted_flux=sorted(sorted_locations,key=lambda x:-origin[x])
        sorted_flux=sorted([loc for loc in xml_districts if destination[loc]>0],key=lambda x:-destination[x])
        bar_colours=[desaturate(colours[location_to_country[label]](normalized_coords[label]),0.8) for label in sorted_flux]

        adjust=0.6
        plotted=adjust
        for i in range(len(sorted_flux)):
            loc=sorted_flux[i]
            val1=origin[loc]#/float(MCMClen)
            val2=destination[loc]#/float(MCMClen)
            #print loc,val1,val2
            fc=bar_colours[i]
            fc=colours[location_to_country[loc]](0.5)
            ec='none'
            lw=2
            ax.bar(i+adjust,val1,facecolor=fc,edgecolor=ec,width=0.9,align='center',lw=lw)
            ax.bar(i+adjust,-val2,facecolor=fc,edgecolor=ec,width=0.9,align='center',lw=lw)

            yloc=0.0
            ax.text(i+adjust,yloc,'%s'%(map_to_actual[loc]),rotation=90,size=18,va='center',ha='center',
                    color='k',path_effects=effects)

            plotted+=1

        ax.set_xlim(0,plotted)
        ax.tick_params(axis='x',size=0)
        ax.set_xticklabels([])
        ax.tick_params(axis='y',labelsize=26,direction='out',pad=0)
        ax.yaxis.tick_left()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.text(0.05,0.95,'%s countries'%(category),size=30,transform=ax.transAxes,path_effects=[path_effects.Stroke(linewidth=5, foreground='white'),path_effects.Stroke(linewidth=1, foreground='black')])

        ax.axhline(0.0,color='w',lw=1,ls='--')
        
        if min(origin.values())<0.0:
            ax.set_ylabel('log$_{10}$ origin(+) and destination(-) transitions',size=32)
            #plt.savefig(local_output+'EBOV_%s_country_logTransitionRange.png'%(category),dpi=300,bbox_inches='tight')
            #plt.savefig(local_output+'EBOV_%s_country_logTransitionRange.pdf'%(category),dpi=300,bbox_inches='tight')

        else:
            ax.set_ylabel('origin(+) and destination(-) transitions',size=32)
            
#             plt.savefig(local_output+'EBOV_%s_country_transitionRange.png'%(category),dpi=300,bbox_inches='tight')
#             plt.savefig(local_output+'EBOV_%s_country_transitionRange.pdf'%(category),dpi=300,bbox_inches='tight')

        plt.show()


fig,ax = plt.subplots(figsize=(15,10),facecolor='w')
translate={'SLE':'Sierra Leone','LBR':'Liberia','GIN':'Guinea'}

for country in required_countries: ## iterate over countries
   
    tos=country_jump_to[country].values() ## fetch all introduction counts
    froms=country_jump_from[country].values() ## fetch all export counts

    mu_from=np.mean(froms)
    med_from=np.median(froms)
    hpdLo_from,hpdHi_from=hpd(froms,0.95)

    mu_to=np.mean(tos)
    med_to=np.median(tos)
    hpdLo_to,hpdHi_to=hpd(tos,0.95)

    print '\n%s export: %.2f %d %d-%d'%(country,mu_from,med_from,hpdLo_from,hpdHi_from)
    print '%s import: %.2f %d %d-%d'%(country,mu_to,med_to,hpdLo_to,hpdHi_to)

    adjust_x=required_countries.index(country)+1

    if adjust_x==1:
        ax.text(adjust_x-0.125,30,'exported',size=30,rotation=90,va='top',ha='center')
        ax.text(adjust_x+0.125,30,'imported',size=30,rotation=90,va='top',ha='center')

    ax.text(adjust_x,1,translate[country],size=34,va='center',ha='center') ## add country labels

    for x in sorted(unique(froms)): ## iterate over export numbers
        if hpdLo_from<x<=hpdHi_from: ## one colour for within HPDs
            fc=colours[country](0.4)
            ec='k'
            al=1
            z=10
        else: ## another for outside HPDs
            fc=colours[country](0.2)
            ec='w'
            al=0.8
            z=1

        ax.barh(x+1,-froms.count(x)/float(len(froms)),left=adjust_x,facecolor=fc,edgecolor=ec,
                lw=2,height=1.0,align='center',alpha=al,zorder=z) ## horizontal bars indicating posterior probability for any given count

    for x in sorted(unique(tos)): ## mirror bar for introductions
        if hpdLo_to<x<=hpdHi_to:
            fc=colours[country](0.7)
            ec='k'
            al=1
            z=10
        else:
            fc=colours[country](0.2)
            ec='w'
            al=0.8
            z=1
            
        ax.barh(x+1,tos.count(x)/float(len(tos)),left=adjust_x,facecolor=fc,ec=ec,
                    lw=2,height=1.0,align='center',alpha=al,zorder=z)

ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))

ax.tick_params(axis='y',labelsize=26,size=8,direction='out')
ax.tick_params(axis='y',which='minor',labelsize=0,size=5,direction='out')
ax.tick_params(axis='x',size=0)
ax.set_xticklabels([])

ax.set_ylabel('number of migrations',size=30)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_xlim(0.5,3.5)
ax.set_ylim(-0.1,30)
ax.grid(axis='y',which='minor',ls='--',zorder=0)

# plt.savefig(local_output+'EBOV_countryJumpCount_probs.png',dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'EBOV_countryJumpCount_probs.pdf',dpi=300,bbox_inches='tight')

plt.show()


## start figure
fig,ax = plt.subplots(figsize=(20,20),facecolor='w')

## plot all locations
for loc in popCentres.keys():
    country=location_to_country[loc]
    if country in required_countries:
        countryColour=colours[country]
        c=countryColour(0.2)

        ## plot every part of each location (islands, etc)
        for part in location_points[loc]:
            poly=plt.Polygon(part,facecolor=c,edgecolor='grey',closed=True)
            ax.add_patch(poly)

## plot the international borders
for l,local_border in enumerate(global_border):
    ax.plot(column(local_border,0),column(local_border,1),lw=2,color='k',zorder=97,label='%d_border'%(l))
    ax.plot(column(local_border,0),column(local_border,1),lw=6,color='w',zorder=96,label='%d_border'%(l))
    
jump_directions={'SLE':{'GIN':('Tonkolili','Mamou'),'LBR':('Tonkolili','Gbarpolu')},'GIN':{'SLE':('Kouroussa','Koinadugu'),'LBR':('Kouroussa','Lofa')},'LBR':{'SLE':('RiverCess','Pujehun'),'GIN':('RiverCess','Yamou')}} ## define location population centroids that will represent country-to-country jump origins and destinations

for countryA in jump_directions.keys(): ## iterate over origin countries
    for countryB in jump_directions.keys(): ## iterate over destination countries
        if countryA!=countryB:
            locA,locB=jump_directions[countryA][countryB] ## fetch locations

            fc=colours[countryA] ## get colour
            
            pointA=popCentres[locA] ## get population centroids
            beginX,beginY=pointA

            pointB=popCentres[locB]
            endX,endY=pointB

            ## calculate distance between locations
            distance=math.sqrt(math.pow(beginX-endX,2)+math.pow(beginY-endY,2))
            
            #############
            ## this controls the distance at which the Bezier line control point will be placed
            #############
            adjust_d=-1+0.1+1/float(distance)**0.15+0.5
            ## find the coordinates of a point n that is at a distance adjust_d, perpendicular to the mid-point between points A and B
            n=Bezier_control(pointA,pointB,adjust_d)

            bezier_start=0.0
            bezier_end=1.0

            ## get Bezier line points
            bezier_line=Bezier([pointA,n,pointB],bezier_start,bezier_end,num=40)

            posteriorCounts=[len(x) for x in country_jump_matrix[countryA][countryB].values()] ## get posterior counts of jumps

            med=np.median(posteriorCounts)
            hpdLo,hpdHi=hpd(posteriorCounts,0.95)
            print '%s > %s: %3d [ %3d - %3d ]'%(countryA,countryB,med,hpdLo,hpdHi)
            
            ## iterate through Bezier line segments with fading alpha and reducing width
            for q in range(len(bezier_line)-1):
                x1,y1=bezier_line[q]
                x2,y2=bezier_line[q+1]

                ## fraction along length of Bezier line
                segL=(q+1)/float(len(bezier_line))

                ## plot jumps
                ax.plot([x1,x2],[y1,y2],lw=6*hpdLo*segL,alpha=1,color=fc(1.0),zorder=100,solid_capstyle='round')
                ax.plot([x1,x2],[y1,y2],lw=6*med*segL,alpha=1,color=fc(0.5),zorder=99,solid_capstyle='round')
                ax.plot([x1,x2],[y1,y2],lw=6*hpdHi*segL,alpha=1,color=fc(0.2),zorder=98,solid_capstyle='round')
                ax.plot([x1,x2],[y1,y2],lw=3+6*hpdHi*segL,alpha=1,color='w',zorder=97,solid_capstyle='round')

import matplotlib.patheffects as path_effects

effects=[path_effects.Stroke(linewidth=4, foreground='white'),path_effects.Stroke(linewidth=0.5, foreground='black')]

display=[1,2,3,5,10,15] ## missile sizes for legend
positions=np.linspace(5,8.5,len(display))[::-1] ## positions for missile legend
for i,x in enumerate(positions): ## iterate over missile legend numbers, plot
    beginX=-12.8
    beginY=7.1

    endX=-15+len(display)*4e-10*sum(positions[:i])**5.9
    endY=x+(sum(positions)-sum(positions[:i]))*0.01

    ax.text(endX,endY,'%d'%(display[i]),size=16+display[i]*0.6,va='center',ha='center',path_effects=effects,zorder=200)

    distance=math.sqrt(math.pow(beginX-endX,2)+math.pow(beginY-endY,2))
    adjust_d=-1+0.1+1/float(distance)**0.15+0.5
    n=Bezier_control((beginX,beginY),(endX,endY),adjust_d)

    bezier_start=0.0
    bezier_end=1.0

    bezier_line=Bezier([(beginX,beginY),n,(endX,endY)],bezier_start,bezier_end,num=50)

    for q in range(len(bezier_line)-1):
        x1,y1=bezier_line[q]
        x2,y2=bezier_line[q+1]
        segL=(q+1)/float(len(bezier_line))

        ax.plot([x1,x2],[y1,y2],lw=6*display[i]*segL,alpha=1,color='k',zorder=100,solid_capstyle='round')

## make map pretty
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(size=0)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_ylim(ylimits)
ax.set_xlim(xlimits)

# plt.savefig(local_output+'EBOV_countryJump_summary.png',dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'EBOV_countryJump_summary.pdf',dpi=300,bbox_inches='tight')

plt.show()


## start figure
# fig,ax = plt.subplots(figsize=(20,20),facecolor='w')

# fig = plt.figure(figsize=(20, 20))
#gs = gridspec.GridSpec(1, 2,hspace=0.0,wspace=0.0)

cutoff=decimalDate('2014-09-01')


for b in range(2):
    a=0
#     for b in range(2):

#     ax=plt.subplot(gs[a,b])
    fig,ax = plt.subplots(figsize=(15,15),facecolor='w')
    
    def timeCondition(time):
        return True
    
    
    if b==0:
        def locCondition(locA,locB):
            ## don't share border
            if shared_border[locA][locB]==False:
                return True
            else:
                return False

        if a==0:
            ax.set_xlabel('no shared border',size=30)
    elif b==1:
        def locCondition(locA,locB):
            ## don't share border
            if shared_border[locA][locB]==True:
                return True
            else:
                return False
        if a==0:
            ax.set_xlabel('shared border',size=30)

#     if a==0:
# #             def timeCondition(time):
# #                 if time<cutoff:
# #                     return True
# #                 else:
# #                     return False

#         if b==1:
#             ax.set_ylabel('before 2014 September',size=30)
#     elif a==1:
# #         def timeCondition(time):
# #             if time>cutoff:
# #                 return True
# #             else:
# #                 return False
#         if b==1:
#             ax.set_ylabel('after 2014 September',size=30)


    ## plot all locations
    for loc in popCentres.keys():
        country=location_to_country[loc]
        if country in required_countries:
            countryColour=colours[country]
            c=countryColour(0.2)

            # plot every part of each location (islands, etc)
            for part in location_points[loc]:
                poly=plt.Polygon(part,facecolor=c,edgecolor='grey',closed=True)
                ax.add_patch(poly)

    # ## plot the international borders
    for l,local_border in enumerate(global_border):
        ax.plot(column(local_border,0),column(local_border,1),lw=1,color='k',zorder=97,label='%d_border'%(l))
        ax.plot(column(local_border,0),column(local_border,1),lw=3,color='w',zorder=96,label='%d_border'%(l))


    for locA in popCentres.keys():
        for locB in popCentres.keys():
            if locA!=locB and location_to_country[locA]!=location_to_country[locB] and locCondition(locA,locB)==True: ## locations are in different countries and satisfy the location condition defined earlier
#                 if locA!=locB and location_to_country[locA]==location_to_country[locB] and locCondition(locA,locB)==True:
    #         if locA!=locB:

    #             posteriorCounts=[len(x) for x in location_jump_matrix[locA][locB].values()]
                posteriorCounts=[len([t for t in x if timeCondition(t)==True]) for x in location_jump_matrix[locA][locB].values()]

    #             if len(posteriorCounts)/float(MCMClen)>0.1:
                if len(posteriorCounts)>1000: ## only interested in decently supported jumps
                    fc=colours[location_to_country[locA]]

                    pointA=popCentres[locA]
                    beginX,beginY=pointA

                    pointB=popCentres[locB]
                    endX,endY=pointB

                    ## calculate distance between locations
                    distance=math.sqrt(math.pow(beginX-endX,2)+math.pow(beginY-endY,2))

                    adjust_d=-1+0.1+1/float(distance)**0.15+0.5
                    ## find the coordinates of a point n that is at a distance adjust_d, perpendicular to the mid-point between points A and B
                    n=Bezier_control(pointA,pointB,adjust_d)

                    ## Bezier line starts at origin and ends at destination
                    bezier_start=0.0
                    bezier_end=1.0

                    ## get Bezier line points
                    bezier_line=Bezier([pointA,n,pointB],bezier_start,bezier_end,num=20)

                    med=np.median(posteriorCounts)
                    hpdLo,hpdHi=hpd(posteriorCounts,0.95)
                    if hpdHi>0 and location_to_country[locA]!=location_to_country[locB]:
                        print '%20s > %20s: %3d [ %3d - %3d ]'%(locA,locB,med,hpdLo,hpdHi)

                    ## iterate through Bezier line segments with fading alpha and reducing width
                    for q in range(len(bezier_line)-1):
                        x1,y1=bezier_line[q]
                        x2,y2=bezier_line[q+1]

                        ## fraction along length of Bezier line
                        segL=(q+1)/float(len(bezier_line))

                        scalar=4
                        ## plot actual jump
                        ax.plot([x1,x2],[y1,y2],lw=scalar*(hpdLo**0.8)*segL,alpha=1,color=fc(1.0),zorder=100,
                                solid_capstyle='round')
                        ax.plot([x1,x2],[y1,y2],lw=scalar*(med**0.8)*segL,alpha=1,color=fc(0.5),zorder=99,
                                solid_capstyle='round')
                        ax.plot([x1,x2],[y1,y2],lw=scalar*(hpdHi**0.8)*segL,alpha=1,color=fc(0.2),zorder=98,
                                solid_capstyle='round')
                        if hpdHi>0:
                            ax.plot([x1,x2],[y1,y2],lw=3+scalar*(hpdHi**0.8)*segL,alpha=1,color='w',zorder=97,
                                solid_capstyle='round')

    if a==0 and b==1:
        import matplotlib.patheffects as path_effects

        effects=[path_effects.Stroke(linewidth=4, foreground='white'),path_effects.Stroke(linewidth=0.5, foreground='black')]

        display=[1,2,3,4,5,6,7,8]
        positions=np.linspace(5,8.5,len(display))
        for i,x in enumerate(positions):
            beginX=-14.9
            beginY=4.5+i*(np.log(sum(positions[:i+1]))*0.1)

            endX=-15+(len(display)-i+1)*0.7
            endY=beginY

            ax.text(endX,endY,'%d'%(display[i]),size=16+display[i]*0.6,va='center',ha='center',path_effects=effects,zorder=200)

            ## calculate distance between locations
            distance=math.sqrt(math.pow(beginX-endX,2)+math.pow(beginY-endY,2))

            adjust_d=0.05
            ## find the coordinates of a point n that is at a distance adjust_d, perpendicular to the mid-point between points A and B
            n=Bezier_control((beginX,beginY),(endX,endY),adjust_d)

            bezier_start=0.0
            bezier_end=1.0

            ## get Bezier line points
            bezier_line=Bezier([(beginX,beginY),n,(endX,endY)],bezier_start,bezier_end,num=50)

            ## iterate through Bezier line segments with fading alpha and reducing width
            for q in range(len(bezier_line)-1):
                x1,y1=bezier_line[q]
                x2,y2=bezier_line[q+1]

                ## fraction along length of Bezier line
                segL=(q+1)/float(len(bezier_line))
                scalar=4
                ## plot actual jump
                ax.plot([x1,x2],[y1,y2],lw=scalar*(display[i]**0.8)*segL,alpha=1,color='k',zorder=100,
                        solid_capstyle='round')

    ## make map pretty
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(size=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylim(ylimits)
    ax.set_xlim(xlimits)
    
#     if b==0:
#         plt.savefig(local_output+'Fig4_international_distant.png',dpi=300,bbox_inches='tight')
#         plt.savefig(local_output+'Fig4_international_distant.pdf',dpi=300,bbox_inches='tight')
#     elif b==1:
#         plt.savefig(local_output+'Fig4_international_shared.png',dpi=300,bbox_inches='tight')
#         plt.savefig(local_output+'Fig4_international_shared.pdf',dpi=300,bbox_inches='tight')
    plt.show()


border_jump_path=path_to_dropbox+'Sequences/Jun2016_1610_genomes/GLM/homogeneous/56_locations/Primary/jumps/jumpsIntNatBorders.txt'


for line in open(border_jump_path,'r'):
    l=line.strip('\n').split('\t')
    if l[0]=='state':
        header=l[1:]
        border_jumps={x:[] for x in header}
    elif int(l[0])>0:
        for i,x in enumerate(l):
            if i>0:
                border_jumps[header[i-1]].append(float(x))


fig,ax1 = plt.subplots(figsize=(10,5),facecolor='w')

ax2=ax1.twinx()

kinds=sorted(border_jumps.keys(),key = lambda x:('Shared' in x,'nat' in x))
print kinds
translate={'natNoBorder':'National,\nno border', 
           'intNoBorder':'International,\nno border', 
           'natSharedBorder':'National,\nshared border', 
           'intSharedBorder':'International,\nshared border'}

kind_colours={'natNoBorder':'#5F9967', 
              'intNoBorder':'#5F7099', 
              'natSharedBorder':'#37803D', 
              'intSharedBorder':'#375A80'}

for k,kind in enumerate(kinds):
    hpdHi,hpdLo=hpd(border_jumps[kind],0.95)
    x_grid=np.linspace(hpdLo,hpdHi,101)
    
    kde=gaussian_kde(border_jumps[kind],bw_method=0.3)
    y_grid=kde.evaluate(x_grid)
    y_grid=y_grid/y_grid.max()*0.35
    
    if 'No' in kind:
        ax=ax1
        ax.set_ylim(0.01,0.4)
        ax.set_ylabel('Pairwise transition rate\n(no border)')
    else:
        ax=ax2
        ax.set_ylim(0.1,4.0)
        ax.set_ylabel('Pairwise transition rate\n(shared border)')
    ax.plot([k+y for y in y_grid],x_grid,color=kind_colours[kind],lw=2,zorder=100)
    ax.plot([k-y for y in y_grid],x_grid,color=kind_colours[kind],lw=2,zorder=100)
    
    ax.fill_betweenx(x_grid,[k-y for y in y_grid],[k+y for y in y_grid],facecolor=kind_colours[kind],edgecolor='none',alpha=0.5,zorder=100)

    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([translate[k] for k in kinds])

    ax.grid(axis='y',ls='--',color='grey',zorder=0)

    ax.xaxis.tick_bottom()

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(size=0)

ax1.axvspan(-0.5,1.5,facecolor='grey',edgecolor='none',alpha=0.08,zorder=0)

# plt.savefig(local_output+'border_rates.png',dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'border_rates.pdf',dpi=300,bbox_inches='tight')

plt.show()


# ## This cell imports most of the necessary modules, polygon outlines and creates colour maps that are used later
# 

get_ipython().magic('matplotlib inline')
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects
from IPython.display import HTML

import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr

from ebov_data import *

if locations:
    pass
else:
    status()
    setFocusCountries()
    setColourMaps()
    loadData()

## set default matplotlib font and font size
typeface='Helvetica Neue'
mpl.rcParams['font.weight']=300
mpl.rcParams['axes.labelweight']=300
mpl.rcParams['font.family']=typeface
mpl.rcParams['font.size']=22

path='<iframe style="border: 0; width: 400px; height: 372px;" src="https://bandcamp.com/EmbeddedPlayer/album=1015754526/size=large/bgcol=333333/linkcol=e99708/artwork=small/track=867144022/transparent=true/" seamless><a href="http://sefektas.bandcamp.com/album/monotonijos-anatomija">Monotonijos Anatomija by Sraiges efektas</a></iframe>'

print 'Done!'
HTML(path)


# ## This cell imports predictor matrices that will be plotted later.
# 

#########################################################################################
#### IMPORT PREDICTOR MATRICES    #######################################################
#########################################################################################
dtypes_path=path_to_dropbox+'Sequences/Aug2016_1610_genomes/Joint/HomogenousGLM/All_1610/Run1/Makona_1610_cds_ig.joint_GLM.xml'

xml_districts=[]
read_loc=False
for line in open(dtypes_path,'r'):
    if 'location.dataType' in line:
        read_loc=True
    #print line
    cerberus=re.search('<state code="([A-Za-z]+)"/>',line)
    if read_loc==True and cerberus is not None:
        xml_districts.append(cerberus.group(1))

print 'Number of datatypes: %d'%(len(xml_districts))
        
## matrix indexing
matrix_index={}
lg=len(xml_districts)
for i in range(len(xml_districts)):
    for j in range(i+1,len(xml_districts)):
        f1=int((lg*(lg-1)/2) - (lg-i)*((lg-i)-1)/2 + j - i - 1)
        f2=int((lg*(lg-1)) - (lg-i)*((lg-i)-1)/2 + j - i - 1)

        matrix_index[f1]=(xml_districts[i],xml_districts[j])
        matrix_index[f2]=(xml_districts[j],xml_districts[i])

predictor_matrices={}
predictor_description={}
counter=0
description_comment=''
store=''
for line in open(dtypes_path,'r'):
    ## find matrix
    if store!='<!--\n':
        matrix=re.search('<parameter id="([A-Za-z0-9\_]+)" value="([0-9\.\- Ee]+)" */>',line)
#         if matrix is not None:
#             print description_comment,store
        if matrix is not None and description_comment!='':
            predictor=matrix.group(1)
            
            float_matrix=map(float,matrix.group(2).strip(' ').split(' '))
            if len(float_matrix)==(len(xml_districts)**2-len(xml_districts)):
                predictor_matrices[predictor]=float_matrix
                predictor_description[predictor]=description_comment
                description_comment=''
                #print 'predictor name:',predictor,len(float_matrix)
#     else:
#         counter-=1

    ## find description of matrix
    description_comment=re.search('<!-- predictor [0-9]+: ([\(\)A-Za-z0-9, \-\.\+>]+) +-->',line)
    if description_comment is not None:
        description_comment=description_comment.group(1)
        counter+=1
        #print '\npredictor description:',description_comment,counter
    else:
        description_comment=''
    store=line
    
print '\npredictors found in file: %s\npredictor matrices: %s\npredictor descriptions: %s'%(counter,len(predictor_matrices),len(predictor_description))
# print predictor_description

for predictor in predictor_matrices.keys():
    if predictor_description[predictor]!=None:
        print predictor,predictor_description[predictor]
    else:
        predictor_matrices.pop(predictor)

#########################################################################################
#### IMPORT PREDICTOR MATRICES    #######################################################
#########################################################################################

## start figure
fig,ax = plt.subplots(figsize=(20,20),facecolor='w')

print '\nLocations in the map are normalized by %s'%(normalize_by)

for i,loc in enumerate(locations):
    country=location_to_country[loc]

    if country in required_countries:

        countryColour=colours[country]
        c=desaturate(countryColour(normalized_coords[loc]),0.8)

        ## plot population centres
        lon,lat=popCentres[loc]

        ## plot district names at centre
        ax.scatter(lon,lat,100,facecolor=c,zorder=100)

        ## plot district polygons
        ax.add_collection(PatchCollection(polygons[loc],facecolor=c,edgecolor='k',lw=0.5,zorder=0))

for local_border in global_border:
    ax.plot(column(local_border,0),column(local_border,1),color='k',lw=2,zorder=99)
    ax.plot(column(local_border,0),column(local_border,1),color='w',lw=5,zorder=98)

## make plot pretty
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(size=0)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_ylim(ylimits)
ax.set_xlim(xlimits)

ax.set_aspect(1)

plt.show()
print 'Done!'


# ## This cell plots predictors onto maps
# - "destination" or "origin" predictors are plotted by colouring the map
# - pairwise comparisons are plotted by connecting population centres with lines
# 

rows=3
cols=3

plt.figure(figsize=(cols*8,rows*8),facecolor='w')

## define subplots
gs = gridspec.GridSpec(rows, cols,hspace=0.0,wspace=0.0)

required=sorted([x for x in predictor_matrices.keys() if 'destination' in x],key=lambda d:predictor_description[d])

cmap=mpl.cm.get_cmap('viridis')

for p,predictor in enumerate(required):
    
    row=int(p/cols)
    
    if row==0:
        col=p
    else:
        col=p%cols
        
    print p,row,col,predictor
    
    ax = plt.subplot(gs[row, col])

    for i,loc in enumerate(xml_districts):
        country=location_to_country[loc]
        if country in required_countries:

            countryColour=colours[country]
            c=desaturate(countryColour(normalized_coords[loc]),0.8)

            ## transform predictor value into the range [0,1]
            omin=min(predictor_matrices[predictor])
            omax=max(predictor_matrices[predictor])
            nmin=0.0
            nmax=1.0
            newRange=nmax-nmin
            oldRange=omax-omin

            destinations={}

            med=np.median(predictor_matrices[predictor])

            for ind,value in enumerate(predictor_matrices[predictor]):
                ori,dest=matrix_index[ind]
                #oriX,oriY=popCentres[ori]
                #destX,destY=popCentres[dest]

                ## normalize entire range to be within range [0,1]
                normalizeValue=(((value-omin)*newRange)/float(oldRange)) + nmin
                destinations[dest]=normalizeValue

            ## fc is alternative
            fc=desaturate(cmap(destinations[loc]),0.8)

            ## plot district borders
            for part in location_points[loc]:
                ax.plot(column(part,0),column(part,1),lw=0.1,color='k',zorder=200)

            ## plot population centres
            lon,lat=popCentres[loc]

            ## plot district polygons
            ax.add_collection(PatchCollection(polygons[loc],facecolor=fc,edgecolor='k',lw=0.5,zorder=0))

    ## make plot pretty
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(size=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylim(ylimits)
    ax.set_xlim(xlimits)

    for local_border in global_border:
        ax.plot(column(local_border,0),column(local_border,1),color='k',lw=2,zorder=201)

    break_at=4
    break_description='\n'.join([' '.join([y for y in predictor_description[predictor].split(' ')[x:x+break_at]]) for x in range(0,len(predictor_description[predictor].split(' ')),break_at)])

    ax.text(0.01,0.01,break_description,size=18,transform=ax.transAxes)

# plt.savefig(local_output+'EBOV_destinationMap.png',dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'EBOV_destinationMap.pdf',dpi=300,bbox_inches='tight')
plt.show()


# required=['international_language_shared']
required=predictor_matrices.keys()

# cmap=mpl.cm.cubehelix_r
# cmap=mpl.cm.get_cmap('viridis_r')
cmap=mpl.cm.get_cmap('viridis')
# cmap_alt=mpl.cm.Purples
cmap_alt=mpl.cm.get_cmap('viridis')

# not_required=['origin','Assymetry','within']
not_required=['origin','Assymetry']

for predictor in required:
    if [x in predictor for x in not_required].count(False)==len(not_required):
#     if 'origin' not in predictor:
        print predictor
        ## start figure
        fig,ax = plt.subplots(figsize=(15,15),facecolor='w')

        ## transform predictor value into the range [0,1]
        omin=min(predictor_matrices[predictor])
        omax=max(predictor_matrices[predictor])
        nmin=0.0
        nmax=1.0
        newRange=nmax-nmin
        oldRange=omax-omin

        destinations={}
        
        med=np.median(predictor_matrices[predictor])
        
        for ind,value in enumerate(predictor_matrices[predictor]):
            ori,dest=matrix_index[ind]
            oriX,oriY=popCentres[ori]
            destX,destY=popCentres[dest]

            ## normalize entire range to be within range [0,1]
            normalizeValue=(((value-omin)*newRange)/float(oldRange)) + nmin
            
            alpha=normalizeValue
            zorder=int(normalizeValue*100)-1
            proceed=True
            lw=8**normalizeValue
            
            exceptions=['Distance','KL']
            mask0s=['sharedBorder','language_BinaryShared']
            
            if [exception in predictor for exception in exceptions].count(True)>=1:
                zorder=int((1-normalizeValue)*100)-1
                alpha=1-normalizeValue
                lw=8**(1-normalizeValue)
            
            if [mask0 in predictor for mask0 in mask0s].count(True)>=1:
                if value==0.0:
                    proceed=False
            
            ## if predictor is destination/origin effect or discrete states shared by countries
            # remember destination value, otherwise connect population centres
            if proceed==True:
                if 'destination' not in predictor:
                    ax.plot([oriX,destX],[oriY,destY],color=cmap(normalizeValue),lw=lw,alpha=alpha,solid_capstyle='round',zorder=zorder)
                else:
                    destinations[dest]=normalizeValue
                
        
        districtPolygons={}
        for i,loc in enumerate(xml_districts):
#             country,location=loc.split('_')
            country=location_to_country[loc]
            
            if country in required_countries:
    
                countryColour=colours[country]
                c=desaturate(countryColour(normalized_coords[loc]),0.8)

                if 'destination' in predictor:
                    ## fc is alternative
                    fc=desaturate(cmap_alt(destinations[loc]),0.8)

                ## plot population centres
                lon,lat=popCentres[loc]

                ## if predictor goes both ways districts are white with coloured centres
                if 'destination' not in predictor:
                    fc='w'
                    ## plot district names at centre
                    ax.scatter(lon,lat,100,facecolor=c,zorder=203)

                ## plot district polygons
                ax.add_collection(PatchCollection(polygons[loc],facecolor=fc,edgecolor='k',lw=0.5,zorder=0))

        ## make plot pretty
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(size=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylim(ylimits)
        ax.set_xlim(xlimits)

        ## add colorbars
        axcb = fig.add_axes([0.2, 0.1, 0.6, 0.02], frame_on=False)
        if 'destination' not in predictor:
            cb = mpl.colorbar.ColorbarBase(axcb,cmap=cmap,norm=mpl.colors.Normalize(vmin=min(predictor_matrices[predictor]),vmax=max(predictor_matrices[predictor])),orientation='horizontal',alpha=1.0,drawedges=False)
        elif 'within' in predictor or 'between' in predictor:
            pass
        else:
            cb = mpl.colorbar.ColorbarBase(axcb,cmap=cmap_alt,norm=mpl.colors.Normalize(vmin=min(predictor_matrices[predictor]),vmax=max(predictor_matrices[predictor])),orientation='horizontal',alpha=1.0,drawedges=False)

        break_at=6
        break_description='\n'.join([' '.join([y for y in predictor_description[predictor].split(' ')[x:x+break_at]]) for x in range(0,len(predictor_description[predictor].split(' ')),break_at)])

        axcb.xaxis.set_label_position("top")
        axcb.set_xlabel('%s'%(break_description),size=20)
        axcb.xaxis.labelpad=5

        axcb.tick_params(axis='y',which='both',direction='out',size=12,width=1,pad=10)
        plt.setp(axcb.get_yticklabels(),size=32,name=typeface)
        for tick in axcb.yaxis.get_ticklines():
            tick.set_markersize(10)

        for local_border in global_border:
            ax.plot(column(local_border,0),column(local_border,1),color='k',lw=2,zorder=201)
            ax.plot(column(local_border,0),column(local_border,1),color='w',lw=4,zorder=200)

#         plt.savefig(local_output+'EBOV_predictorMap_%s.png'%(predictor),dpi=300,bbox_inches='tight')
#         plt.savefig(local_output+'EBOV_predictorMap_%s.pdf'%(predictor),dpi=300,bbox_inches='tight')
        plt.show()


# ## This cell plots predictors as a matrix if they are not "destination" or "origin" predictors.
# 

# required=['destinationTmpss','destinationPrecss','withinCountry','destinationsample','destinationTemp']

country_rank={x:[popCentres[y][0] for y in location_to_country.keys() if location_to_country[y]==x][0] for x in required_countries}
required=predictor_matrices.keys()
print country_rank
if normalize_by=='PCA1':
    sorted_locations=sorted(xml_districts,key=lambda x:(-country_rank[location_to_country[x]],-normalized_coords[x]))
elif normalize_by=='PCA2':
    sorted_locations=sorted(xml_districts,key=lambda x:(location_to_country[x],normalized_coords[x]))

# cmap=mpl.cm.Spectral_r
cmap=mpl.cm.get_cmap('viridis')

## iterate through predictors
for predictor in required:
    ## only produce heatmap representation if it's not a "destination" or "origin" predictor
    if 'destination' not in predictor and 'origin' not in predictor:
        empty=np.zeros((len(xml_districts),len(xml_districts)))
        empty.fill(np.nan)
        analysis=pd.DataFrame(empty,index=sorted_locations,columns=sorted_locations)
        
        ## start figure
        fig,ax = plt.subplots(figsize=(15,15),facecolor='w')

        ## identify matrix coordinates of predictor
        ## if it's a distance predictor - invert value
        for ind,value in enumerate(predictor_matrices[predictor]):
            ori,dest=matrix_index[ind]
            oriX,oriY=popCentres[ori]
            destX,destY=popCentres[dest]

            if 'Distance' in predictor or 'KL' in predictor:
                value=1-value
                
            analysis[dest][ori]=value

        ## mask NaNs
        masked_array = np.ma.array(np.array(analysis),mask=np.isnan(analysis))

        print predictor,sum(analysis.sum())
        
        cmap.set_bad('k',1.)
        ## plot heatmap
        heatmap = ax.pcolor(masked_array,edgecolors='none', linewidths=0,cmap=cmap,alpha=1)

        ## make plot pretty
        ax.set_xticks(np.arange(0.5,len(sorted_locations)+0.5))
        ax.set_xticklabels([x for x in sorted_locations],rotation=90)
        ax.set_yticks(np.arange(0.5,len(sorted_locations)+0.5))
        ax.set_yticklabels([x for x in sorted_locations])
        ax.set_xlabel('destination',size=30)
        ax.set_ylabel('origin',size=30)
        ax.tick_params(size=0,labelsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        axcb = fig.add_axes([0.97, 0.15, 0.02, 0.6], frame_on=False)
        cb = mpl.colorbar.ColorbarBase(axcb,cmap=cmap,norm=mpl.colors.Normalize(vmin=min(predictor_matrices[predictor]),vmax=max(predictor_matrices[predictor])),orientation='vertical',alpha=1.0,drawedges=False)

        axcb.yaxis.set_label_position("left")
        axcb.tick_params(axis='y',which='both',direction='out',size=5,labelsize=26,width=1,pad=5)
        
        break_at=6
        break_description='\n'.join([' '.join([y for y in predictor_description[predictor].split(' ')[x:x+break_at]]) for x in range(0,len(predictor_description[predictor].split(' ')),break_at)])
        #ax.text(0.01,0.95,'%s'%break_description,size=26,va='bottom',ha='left',transform=ax.transAxes)
        axcb.set_ylabel('%s'%break_description,{'fontsize':28})

        for tick in ax.yaxis.get_ticklabels():
            label=str(tick.get_text())
            tick.set_color(desaturate(colours[location_to_country[label]](normalized_coords[label]),0.8))

        countries_of_labels=[location_to_country[x] for x in sorted_locations]
        for c in range(len(countries_of_labels)-1):
            if countries_of_labels[c]!=countries_of_labels[c+1]:
                ax.axhline(c+1,ls='--',color='w')
                ax.axvline(c+1,ls='--',color='w')
            
#         plt.savefig(local_output+'EBOV_predictor_%s_heatmap.png'%predictor,dpi=100,bbox_inches='tight')
#         plt.savefig(local_output+'EBOV_predictor_%s_heatmap.pdf'%predictor,dpi=100,bbox_inches='tight')
        plt.show()


# ## import language sharing matrix
lang_path=path_to_dropbox+'Data/languages/languages.txt'

raw_matrix=[]
lang_locs=[]
for line in open(lang_path,'r'):
    l=line.strip('\n').split('\t')
    if 'ISO' in l[0]:
        languages=l[2:]
    else:
        raw_matrix.append(map(float,l[2:]))
        lang_locs.append(l[1])

lang_matrix={x:{y:np.nan for y in languages} for x in lang_locs}

speakers={x:[] for x in languages}
speaks={x:[] for x in lang_locs}

## unwrap matrix
for i,x in enumerate(raw_matrix):
    for j,y in enumerate(x):
        lang_matrix[lang_locs[i]][languages[j]]=float(raw_matrix[i][j])
        
        if lang_matrix[lang_locs[i]][languages[j]]>0:
            speakers[languages[j]].append(lang_locs[i])
            speaks[lang_locs[i]].append(languages[j])
languages=[l for l in languages if len(speakers[l])>0]

lcmap=mpl.cm.Spectral

language_colours={languages[q]:desaturate(lcmap((len(languages)-q)/float(len(languages))),0.6) for q in range(len(languages))}

fig,ax = plt.subplots(figsize=(20,20),facecolor='w')

for l,loc in enumerate(lang_locs):
    country=location_to_country[loc]
    
    if country in ['SLE','LBR','GIN']:
        countryColour=colours[country]
        c=desaturate(countryColour(normalized_coords[loc]),0.8)

        ## plot population centres
        lon,lat=popCentres[loc]

        ## convert percentage speakers to radians
        lang_order=sorted(languages,key=lambda w:-lang_matrix[loc][w])
        sizes=[lang_matrix[loc][a] for a in lang_order]
        N_spoken=sum([1 if x>0.0 else 0 for x in sizes])
    #     sizes=np.rad2deg([x*np.pi*2 for x in sizes])

        ## determines the height of bars
        frac=0.4

        ax.scatter(lon,lat,s=100,facecolor='w',edgecolor='k',zorder=2000)

        ## iterate over language proportions, plot pie wedges
        for s in range(len(sizes)):
            h=''
            if languages.index(lang_order[s])%2==0:
                h='//'

            left_right,up_down=textCorrection[loc]

            if left_right==0:
                adjustX=1
            else:
                adjustX=-20*frac

            if up_down==0:
                adjustY=0.0
            else:
                adjustY=-0.5

            x=lon+0.06*adjustX+sum(sizes[:s])*frac
            y=lat-0.025*N_spoken

            lw=1
            if sizes[s]==1.0:
                lw=2
            sl=mpl.patches.Rectangle((x,y),sizes[s]*frac,0.05*N_spoken,facecolor=language_colours[lang_order[s]],
                                      edgecolor='k',lw=lw,alpha=1.0,hatch=h,zorder=1000)

            ax.add_patch(sl)

        ## plot district borders
        for part in location_points[loc]:
            ax.plot(column(part,0),column(part,1),lw=1,color='w',ls=':',zorder=10)

        h=''
        if languages.index(lang_order[0])%2==0:
            h='/'

        ## plot district polygons coloured by shared language
        fc=desaturate(language_colours[lang_order[0]],0.7)
        ax.add_collection(PatchCollection(polygons[loc],facecolor=fc,hatch=h,edgecolor='k',lw=0,zorder=1))

for ling in range(len(languages)):
    y=(len(languages)-ling)/30.0

    h=''
    if languages.index(languages[ling])%2==0:
        h='//'

    circle=mpl.patches.Circle((0.03,y+0.02),radius=0.01,facecolor=language_colours[languages[ling]],
                              hatch=h,transform=ax.transAxes)
    ax.add_patch(circle)

    ax.text(0.05,y+0.01,'%s'%(languages[ling]),size=40,color=language_colours[languages[ling]],
            transform=ax.transAxes)

## add bar to indicate distance
ycoord=np.mean([4.3,12.7])
legend_y=12.0
legend_x1=-15
legend_x2=-14.08059

ax.plot([legend_x1,legend_x2],[legend_y,legend_y],color='k',lw=6)
ax.text(np.mean([legend_x1,legend_x2]),legend_y+0.04,'%.0f km'%metricDistance((legend_x1,legend_y),(legend_x2,legend_y)),size=36,va='bottom',ha='center')
    
for local_border in global_border:
    ax.plot(column(local_border,0),column(local_border,1),lw=2,color='k',zorder=201)

## make plot pretty
ax.set_aspect(1)
ax.tick_params(size=0)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_ylim(ylimits)
ax.set_xlim(xlimits)

# plt.savefig(local_output+'EBOV_language_distribution_map_var3.png',dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'EBOV_language_distribution_map_var3.pdf',dpi=300,bbox_inches='tight')

plt.show()


unrolled_national=[]
unrolled_international=[]

#output_border=open('/Users/evogytis/Downloads/international_border.txt','w')

matrix_index={} ## predictor matrix indexing
lg=len(locations)

k=0
triangle_size = (lg*(lg-1)/2)
for i in range(len(locations)):
    for j in range(i+1,len(locations)):
        matrix_index[k] = (locations[i],locations[j])
        matrix_index[k + triangle_size] = (locations[j],locations[i])
        
        k += 1

def dist_score(L):
    return sum([metricDistance(L[x+1],L[x]) for x in range(len(L)-1)])

def findOrder(L):
    storeBest=L
    for i in range(len(L)):
        b=L[i:]+L[:i]
        score=dist_score(b)
        if score<=dist_score(storeBest):
            storeBest=b

    return storeBest   

border_output= open('/Users/evogytis/Downloads/international_border.txt','w')
for i in matrix_index.keys():
    locA,locB=matrix_index[i]
    ## get all points of a location
    countryA=location_to_country[locA]
    countryB=location_to_country[locB]
    
    pointsA=location_points['%s_%s'%(location_to_country[locA],locA)]
    pointsB=location_points['%s_%s'%(location_to_country[locB],locB)]

    ## location points can be split into parts - flatten list
    joinA=list([item for sublist in pointsA for item in sublist])
    joinB=list([item for sublist in pointsB for item in sublist])

    share_border=False
    common_points=share_points(joinA,joinB)

    p=0

    if common_points==True and countryA!=countryB:
        for polyA in pointsA:

            for polyB in pointsB:
                ol,rA,rB=overlap(map(tuple,polyA),map(tuple,polyB))
                ol+=[tuple(x) for x in rA for y in rB if metricDistance(x,y)<=0.001]

                frag=[]
                for pts in polyA:
                    if tuple(pts) in ol:
                        frag.append(tuple(pts))
                        
                frag=findOrder(frag)
                #print frag
                print>>border_output,'%s'%(frag)

border_output.close()


required=predictor_matrices.keys()

cmap=mpl.cm.Spectral_r

# start figure
fig,ax = plt.subplots(figsize=(35,35),facecolor='w')

sorted_required=sorted(required,key=lambda x:x)

empty=np.zeros((len(sorted_required),len(sorted_required)))
empty.fill(np.nan)
analysis=pd.DataFrame(empty,index=sorted_required,columns=sorted_required)

for i,predictorA in enumerate(required):
    #print '%30s%10.2f'%(predictorA,i/float(len(required)))
    for j,predictorB in enumerate(required):

        matrixA=predictor_matrices[predictorA]
        matrixB=predictor_matrices[predictorB]
        
        coeff,pvalue=spearmanr(matrixA, matrixB)
        analysis[predictorA][predictorB]=coeff

        ax.text(sorted_required.index(predictorA)+0.5,sorted_required.index(predictorB)+0.5,'%.2f'%(coeff),size=14,va='center',ha='center')

masked_array = np.ma.array(np.array(analysis),mask=np.isnan(analysis))

## plot heatmap of the predictor matrix
heatmap = ax.pcolor(masked_array,edgecolors='w', lw=1,cmap=cmap,alpha=1,norm=mpl.colors.Normalize(-1,1))

## make plot pretty
ax.set_xticks(np.arange(0.5,len(required)+0.5))
ax.set_yticks(np.arange(0.5,len(required)+0.5))
ax.set_xticklabels(sorted_required,rotation=90)
ax.set_yticklabels(sorted_required)
ax.set_xlim(0,len(required))
ax.set_ylim(0,len(required))
ax.tick_params(size=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

axcb = fig.add_axes([0.92, 0.15, 0.02, 0.6], frame_on=False)

cb = mpl.colorbar.ColorbarBase(axcb,cmap=cmap,norm=mpl.colors.Normalize(vmin=-1,vmax=1),orientation='vertical',alpha=1.0,drawedges=False)

axcb.yaxis.set_label_position("left")
axcb.tick_params(axis='y',which='both',direction='out',size=12,width=1,pad=10)
plt.setp(axcb.get_yticklabels(),size=32,name=typeface)

for tick in axcb.yaxis.get_ticklines():
    tick.set_markersize(10)

plt.show()


sorted_required=sorted([w for w in predictor_matrices.keys() if 'origin' in w],key=lambda x:x)

empty=np.zeros((len(sorted_required),len(sorted_required)))
empty.fill(np.nan)
analysis=pd.DataFrame(empty,index=sorted_required,columns=sorted_required)

cmap=mpl.cm.RdBu_r

fig,ax = plt.subplots(figsize=(25,25),facecolor='w')

gs = gridspec.GridSpec(len(sorted_required), len(sorted_required),hspace=0.03,wspace=0.03)

effects=[path_effects.Stroke(linewidth=4, foreground='white'),path_effects.Stroke(linewidth=0.5, foreground='black')]

for pA,predictorA in enumerate(sorted_required):
    for pB,predictorB in enumerate(sorted_required):
        
        if pA>pB:
            ax = plt.subplot(gs[pA, pB])
            
            xs=[]
            ys=[]
            done=[]
            for i,x in enumerate(predictor_matrices[predictorA]):
                origin,destination=matrix_index[i]
                if origin not in done:
                    xs.append(predictor_matrices[predictorB][i])
                    ys.append(predictor_matrices[predictorA][i])
                    done.append(origin)

            spear,pval=spearmanr(xs,ys)
            if pval>0.05:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
            else:
                ax.scatter(xs,ys,s=100,facecolor='k',edgecolor='none',zorder=0)
                if spear>0.0:
                    ax.text(0.97,0.03,'%.2f'%(spear),va='bottom',ha='right',transform=ax.transAxes,path_effects=effects,zorder=1000)
                else:
                    ax.text(0.03,0.03,'%.2f'%(spear),va='bottom',ha='left',transform=ax.transAxes,path_effects=effects,zorder=1000)
                    
            ax.scatter(xs,ys,s=25,facecolor=cmap((spear+1.0)/2.0),edgecolor='none',zorder=100)

            if pB==0:
                ax.set_ylabel(predictorA,rotation='horizontal',size=30)
                ax.yaxis.get_label().set_horizontalalignment("right")
            if pA==len(sorted_required)-1:
                ax.set_xlabel(predictorB,rotation=90,size=30)
                
            ax.tick_params(size=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            print '%s\t%s\t%.3f\t%.4f'%(predictorA,predictorB,spear,pval)

#plt.savefig(local_output+'EBOV_location_predictor_correlations.png',dpi=300,bbox_inches='tight')
#plt.savefig(local_output+'EBOV_location_predictor_correlations.pdf',dpi=300,bbox_inches='tight')
plt.show()


# ## Create case number diagram:
# Go through weekly WHO situation reports and patient database numbers, keep the higher number
# 

import re
import os

path_to_dropbox='/Users/evogytis/Dropbox/Ebolavirus_Phylogeography/'
cases_path=path_to_dropbox+'Case counts by district/'

report_root='2016-Feb-22_'
filenames=[report_root+'GIN.csv',report_root+'SLE.csv',report_root+'LBR.csv']
rename={report_root+'GIN.csv':'GIN',report_root+'SLE.csv':'SLE',report_root+'LBR.csv':'LBR'}

map_to_standard={'yomou':'Yamou'}
map_to_actual={}
standard_path=path_to_dropbox+'Maps/standardDistricts.tsv'
for line in open(standard_path,'r'):
    l=line.strip('\n').split('\t')
    ## map parts of a district name to its standard name
    # split by space, keep last word in lower case
    # e.g River Gee = gee, Grand Cape Mount = mount.
    map_to_standard[l[3].lower()]=l[-1]
    
    actual=l[1]
    if 'actual' not in actual:
        actual=actual.decode('utf8')
        map_to_actual[l[-1]]=actual

local_output='/Users/evogytis/Documents/EBOV_output/'
output=local_output+'EBOV_maxCases.csv'
try:
    os.remove(output)
    print 'Removed previous file'
except:
    pass

out=open(output,'w')
dates=[]

## iterate over country reports
for fname in filenames:
    print fname
    data={}
    for line in open(cases_path+fname,'r'):
        l=line.replace('"','').strip('\n').split(',')
        
        ## header
        if 'Location' in l[0]:
            ## find all epi weeks
            cerberus=re.findall('([0-9]+) ([A-Za-z]+) to [0-9]+ [A-Za-z]+ ([0-9]+)|([0-9]+) to [0-9]+ ([A-Za-z]+) ([0-9]+)|([0-9]+) ([A-Za-z]+) ([0-9]+) to [0-9]+ [A-Za-z]+ [0-9]+',','.join(l))
            
            ## if dates is empty - populate it with epi weeks
            if len(dates)==0:
                dates=['-'.join((y[0],y[1][:3],y[2])) for y in [[z for z in x[::-1] if z!=''] for x in cerberus]]
                print>>out,'country,district,standard,%s'%(','.join(dates))
                
        ## starting actual data
        elif l[0]!='':
            district=l[0]

            if data.has_key(district)==False:
                ## each district has a situation report and patient database lists
                data[district]={'situation':[0.0 for e in dates],'patient':[0.0 for e in dates]}
            
            ## combine confirmed and suspected cases for patient database
            if 'Patient database' in l[1] and 'Confirmed' in l[3]:
                data[district]['patient']=[int(x) if (x not in [' ','']) else 0 for x in l[4:]]
            elif 'Patient database' in l[1] and 'Probable' in l[3]:
                data[district]['patient']=[int(x)+y if (x not in [' ','']) else 0 for x,y in zip(l[4:],data[district]['patient'])]
                
            ## likewise for situation report
            elif 'Situation report' in l[1] and 'Confirmed' in l[3]:
                data[district]['situation']=[int(x) if (x not in [' ','']) else 0 for x in l[4:]]
            elif 'Situation report' in l[1] and 'Probable' in l[3]:
                data[district]['situation']=[int(x)+y if (x not in [' ','']) else 0 for x,y in zip(l[4:],data[district]['situation'])]
    
    ## for every district report maximum (either situation report or patient database) combined (confirmed+suspected) cases
    for q in sorted(data.keys()): 
        counts=[max(a,b) for a,b in zip(data[q]['patient'],data[q]['situation'])]
        ## output to file, including standardised district names
        print>>out,'%s,%s,%s,%s'%(rename[fname],q,map_to_standard[q.lower().replace(' ','').replace("'",'').replace('area','')],','.join(map(str,counts)))
            
## add Mandiana - didn't report any suspected cases
print>>out,'%s,%s,%s,%s'%('GIN','MANDIANA','Mandiana',','.join(['' for x in dates]))

out.close()


get_ipython().magic('matplotlib inline')
import matplotlib as mpl
#mpl.use("pgf")

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.patches as patches
#from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects
from IPython.display import HTML

from ebov_data import *

## set default matplotlib font and font size
# typeface='Arial'
typeface='Helvetica Neue'
mpl.rcParams['font.weight']=300
mpl.rcParams['axes.labelweight']=300
mpl.rcParams['font.family']=typeface
mpl.rcParams['font.size']=22
mpl.rcParams['pdf.fonttype']=42
# mpl.rcParams['ps.fonttype'] = 42


#mpl.rcParams['text.usetex'] = True
# ##############################################
# ## Define paths
# ##############################################

if locations:
    pass
else:
    status()
    setFocusCountries(['SLE','LBR','GIN'])
    setColourMaps()
    loadData()

## path to a file with sequence names
xml_path=path_to_dropbox+'Sequences/Aug2016_1610_genomes/Joint/HomogenousGLM/All_1610/Run1/Makona_1610_cds_ig.joint_GLM.xml'

##############################################
## Import sequence names
##############################################
sequences=[]
for line in open(xml_path,'r'):
    l=line.strip('\n')
    cerberus=re.search('<taxon id="(EBOV\|[A-Za-z0-9\|\-\_\.\?]+)">',l)
    
    if cerberus is not None:
        sequences.append(cerberus.group(1).split('|'))

isolation_dates=[decimalDate(x) for x in column(sequences,-1)]
extremes=['|'.join(x) for x in sorted(sequences,key=lambda z:decimalDate(z[-1])) if decimalDate(x[-1])==max(isolation_dates) or decimalDate(x[-1])==min(isolation_dates)]

print '\noldest sequence:\t%s\nyoungest sequence:\t%s'%(extremes[0],extremes[1])
##############################################
## Import sequences
##############################################

##############################################
## Define timeline
##############################################
timeline=['2013-%02d-01'%x for x in range(12,13)]
timeline+=['2014-%02d-01'%x for x in range(1,13)]
timeline+=['2015-%02d-01'%x for x in range(1,13)]
timeline+=['2016-%02d-01'%x for x in range(1,3)]
##############################################
## Define timeline
##############################################

path='<iframe style="border: 0; width: 400px; height: 405px;" src="https://bandcamp.com/EmbeddedPlayer/album=1879093915/size=large/bgcol=333333/linkcol=ffffff/artwork=small/track=748478573/transparent=true/" seamless><a href="http://obsrr.bandcamp.com/album/patogi-gelm">PATOGI GELME by OBSRR</a></iframe>'

# maxByCountry={x.split('_')[0]:max([sum(cases_byDistrict[x].values())]) for x in cases_byDistrict.keys()}
# maxByCountry={y:([totalCaseCounts[z] for z in totalCaseCounts.keys() if y in z]) for y in unique([x.split('_')[0] for x in countries])}
# maxByCountry={y:([totalCaseCounts[z] for z in totalCaseCounts.keys() if y in z]) for y in unique([x.split('_')[0] for x in countries])}
seq_byCountry={y:[x for x in sequences if x[3]==y] for y in required_countries}
seq_byMonth={date:[x for x in sequences if decimalDate(dates[y])<decimalDate(x[-1])<=decimalDate(dates[y+1])] for y,date in enumerate(dates[:-1])}
##############################################
## Import case numbers
##############################################

##############################################
## Count sequences over time in locations
##############################################
seq_byLocation={z:{y:[] for y in dates} for z in cases_byLocation.keys()}

processed=0
indict=0
ucounter={c:0 for c in required_countries}
ucounter['?']=0
ucounter['WesternArea']=0
for seq in sequences:
    processed+=1
    if seq[4]=='?' or seq[4]=='': ## location unknown
        if seq[3]!='?' or seq[3]!='': ## country known
            ucounter[seq[3]]+=1
        else: ## country unknown
            ucounter['?']+=1 
    else:
        for y in range(len(dates)-1):
            if decimalDate(dates[y])<=decimalDate(seq[-1])<decimalDate(dates[y+1]):
                if seq[4]=='WesternArea':
                    seq_byLocation['WesternUrban'][dates[y]].append(seq)
                    indict+=1
                else:
                    seq_byLocation['%s'%(seq[4])][dates[y]].append(seq)
                    indict+=1

        if decimalDate(dates[-1])<=decimalDate(seq[-1]):
            if seq[4]=='WesternArea':
                seq_byLocation['WesternUrban'][dates[-1]].append(seq)
                indict+=1
            else:
                seq_byLocation['%s'%(seq[4])][dates[-1]].append(seq)
                indict+=1

print '\nSequences with unknown locations:\n%s\nSequences with known location:\t%d\nTotal:\t%d'%('\n'.join(['%s:\t%s'%(a,ucounter[a]) for a in ucounter.keys()]),indict,len(sequences))

locations_with_sequences=len([x for x in seq_byLocation.keys() if sum([len(y) for y in seq_byLocation[x].values()])>0])
locations_with_cases=len([x for x in cases_byLocation.keys() if sum(cases_byLocation[x].values())>0])
print '\nLocations with sequences:\t%d\nLocations with cases:\t%d\nTotal number of locations:\t%d'%(locations_with_sequences,locations_with_cases,len(popCentres))
##############################################
## Count sequences over time in locations
##############################################

HTML(path)


fig,ax = plt.subplots(figsize=(20, 10),facecolor='w') 

## choose whether to log-normalize or not
# logNorm=True
logNorm=False

## plot weekly case numbers
for country in cases_byCountry.keys():
    xkeys=sorted(cases_byCountry[country].keys(),key=lambda x:decimalDate(x))
    xs=[decimalDate(x) for x in xkeys]
    ys=[cases_byCountry[country][x]+1 for x in xkeys]
    ax.plot(xs,ys,color=colours[country](0.6),lw=5,label=translate[country],zorder=int(sum(ys)))
    ax.plot(xs,ys,color='w',lw=9,zorder=int(sum(ys))-1)

total=0
for country in cases_byCountry.keys():
    xkeys=sorted(cases_byCountry[country].keys(),key=lambda x:decimalDate(x))
    ys=[cases_byCountry[country][x] for x in xkeys]
    print 'total %s cases: %d'%(country,sum(ys))
    total+=sum(ys)
print 'total cases in West Africa: %d'%(total)

ax.legend(loc=2,frameon=False,fontsize=30)
    
ax.set_ylabel('New cases per week',size=28)
ax.set_xticks([decimalDate(x)+1/24.0 for x in timeline])
ax.set_xticklabels([convertDate(x,'%Y-%m-%d','%b\n%Y') if x.split('-')[1]=='01' else convertDate(x,'%Y-%m-%d','%b') for x in timeline])
    
ax.set_xlim(decimalDate('2013-12-01'),decimalDate('2016-03-01'))

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# ax.spines['left'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.tick_params(axis='x',size=0,pad=10)
ax.tick_params(axis='y',which='major',size=8,direction='out',pad=10)
ax.tick_params(axis='y',which='minor',size=5,direction='out',pad=10)

[ax.axvspan(decimalDate(timeline[x]),decimalDate(timeline[x])+1/12.,facecolor='k',edgecolor='none',alpha=0.04) for x in range(0,len(timeline),2)]

if logNorm==True:
    ax.set_ylim(bottom=1)
    ax.set_yscale('log')
    
elif logNorm==False:
    ax.set_ylim(bottom=-1)

# plt.savefig(local_output+'EBOV_cases.png',dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'EBOV_cases.pdf',dpi=300,bbox_inches='tight')
plt.show()


## this is for setting x ticks at some monthly interval
every=3
mons=['2013-%02d-01'%(x) for x in range(11,13,every)]
mons+=['2014-%02d-01'%(x) for x in range(1,13,every)]
mons+=['2015-%02d-01'%(x) for x in range(1,13,every)]
mons+=['2016-%02d-01'%(x) for x in range(1,3,every)]

effects=[path_effects.Stroke(linewidth=4, foreground='white'),
                 path_effects.Stroke(linewidth=0.5, foreground='black')] ## black text, white outline

## iterate through countries 
for country in required_countries:

    ## collect all locations, sort by country
    locs=sorted(cases_byLocation.keys(),key=lambda x:location_to_country[x])
    
    ## then ignore locations without cases or from other countries
    ## sort it by total cases
    locs=sorted([x for x in locs if sum(cases_byLocation[x].values())>0 and location_to_country[x]==country],key=lambda a:-sum(cases_byLocation[a].values()))
    
    ## begin plot
    fig = plt.figure(figsize=(20, 20),facecolor='w') 
    
    ## define number of rows
    nrows=4

    ## plots is a grid
    gs = gridspec.GridSpec((len(locs)/nrows)+1, nrows) 

    ## define the y limit of each plot as the peak of the epidemic within the country
    extent=max([max(cases_byLocation[x].values()) for x in cases_byLocation.keys() if location_to_country[x]==country])
    print country,extent
    
    
    ## iterate through the plotting grid
    for i,g in enumerate(gs):
        ax=plt.subplot(g,zorder=len(locs)-i)
        
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
        if i<len(locs):
            loc=locs[i]

            country=location_to_country[loc]
    
#             if country in ['SLE','LBR','GIN']:
    
            xs=sorted([x for x in cases_byLocation[loc].keys()],key = lambda c:decimalDate(c))
            ys=[cases_byLocation[loc][x] for x in xs]
            c=colours[country](0.9)
            
            print country,loc,'first suspected/confirmed case: %s'%([x for x,y in zip(xs,ys) if y>0][0])

            ax.plot([decimalDate(x) for x in xs],ys,color=c,lw=2)
            ax.tick_params(size=0,labelsize=0)

            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            if seq_byLocation.has_key(loc)>0:
                sequence_dates=[decimalDate(y[-1]) for x in seq_byLocation[loc].values() for y in x if len(x)>0]
                ax.scatter(sequence_dates,[-5 for x in sequence_dates],80,lw=2,color=c,marker='|',alpha=0.6)

            ax.text(0.05,0.5,'%s'%(map_to_actual[loc]),va='top',ha='left',transform=ax.transAxes,size=20,path_effects=effects)

            ax.xaxis.tick_bottom()
            ax.yaxis.tick_left()
            ## set x ticks according to monthly intervals
            ax.set_xticks([decimalDate(x) for x in mons])
            
            ## if currently at the very first plot - add x tick labels and make y axis ticks visible
            if i==0:
                ax.set_xticklabels([convertDate(x,'%Y-%m-%d','%b\n%Y') if x.split('-')[1]=='01' else convertDate(x,'%Y-%m-%d','%b') for x in mons])
                ax.tick_params(axis='x',size=4,labelsize=20,direction='out',zorder=100)
                ax.tick_params(axis='y',size=4,labelsize=20,direction='out',pad=5)
            else:
                ax.spines['left'].set_color('none')
                ax.tick_params(axis='x',size=4,direction='out')
                ax.tick_params(axis='y',size=0,direction='out')
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        else:
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.tick_params(size=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
        ax.set_ylim(-10,extent)
        
        ax.set_xlim(decimalDate('2013-12-01'),decimalDate('2016-01-01'))
        ax.autoscale(enable=False) 
    
    ## make grid tighter
    gs.update(hspace=0.01,wspace=0.01)
#     plt.savefig(local_output+'EBOV_%s_sampling.png'%(country),dpi=300,bbox_inches='tight')
#     plt.savefig(local_output+'EBOV_%s_sampling.pdf'%(country),dpi=300,bbox_inches='tight')
    plt.show()


## colour map for coverage plots
# coverage_colours=mpl.cm.get_cmap('viridis')
coverage_colours=mpl.cm.viridis_r
coverage_alpha=0.9

## start figure
fig,ax = plt.subplots(figsize=(20,20),facecolor='w')

## find sequence coverage maximum or define it
# seq_coverage=[[q[5] for q in sequences].count(loc.split('_')[1])/float(totalCaseCounts[loc]) for loc in locations if totalCaseCounts[loc]>0]
seq_coverage=[0.5,0.5]

## cutoff for displaying >X%
seq_coverage=0.2

print '\\begin{longtable}{ | p{.1\\textwidth} | p{.25\\textwidth} | p{.15\\textwidth} | p{.15\\textwidth} | p{.2\\textwidth} | }'
print 'country & location & sequences & cases & \\%% sequences/case \\\\ '
# print maxByCountry

maxByCountry={country:0.001 for country in required_countries}
for location in cases_byLocation.keys():
    country=location_to_country[location]
    if country in required_countries:
        cases_in_location=sum(cases_byLocation[location].values())
        if maxByCountry[country]<=cases_in_location:
            maxByCountry[country]=cases_in_location

storeCountry=''
for i,loc in enumerate(sorted(cases_byLocation.keys(),key=lambda x:(location_to_country[x],-sum(cases_byLocation[x].values())))):
    country=location_to_country[loc]
    
    if country in required_countries and loc!='WesternArea':
        ## identify colour map
        countryColour=colours[country]

        ## find number of cases in the location
        cases_in_location=sum(cases_byLocation[loc].values())

        ## express cases as fraction of maximum cases seen in country
        c=countryColour(cases_in_location/float(maxByCountry[country]))

        ## add hatching for locations without EVD cases
        h=''
        if cases_in_location==0:
            h='/'

        ## mpl doesn't plot polygons unless explicitly told to plot something
        ax.plot()

        ## plot population centres
        lon,lat=popCentres[loc]

        if storeCountry!=country:
            print '\\hline'
        ## count how many times the location name turns up amongst sequences
        circleSize=[q[4] for q in sequences].count(loc)
        if cases_in_location==0:
            print '%s & %s & 0 & 0 & NA \\\\'%(country,loc)
        else:
            print '%s & %s & %d & %d & %.2f \\\\'%(country,loc,circleSize,cases_in_location,circleSize/float(cases_in_location)*100)


        if circleSize==0:
            ## if there are no sequences - plot X
            ax.scatter(lon,lat,300,marker='x',color='k',zorder=99)
        else:
            ## plot circle with radius proportional to log N sequences
            #coverage=(circleSize/float(cases_in_location))/float(seq_coverage)
            coverage=1.0
            if coverage>=1.0:
                coverage=1.0
            circle_colour=coverage_colours(coverage)
            ax.add_collection(PatchCollection([patches.Circle((lon,lat),radius=np.log10(circleSize+1)*0.1)],facecolor='none',
                                              alpha=coverage_alpha,edgecolor=circle_colour,lw=5,zorder=100))
            ax.add_collection(PatchCollection([patches.Circle((lon,lat),radius=np.log10(circleSize+1)*0.1)],facecolor='none',
                                              alpha=coverage_alpha,edgecolor='w',lw=8,zorder=99))

        ## plot location polygons
        ax.add_collection(PatchCollection(polygons[loc],facecolor=c,edgecolor='grey',lw=1,hatch=h,alpha=1.0))

        ## define available text alignments and corrections for text positions
        vas=['bottom','top']
        has=['left','right']
        corrections=[0.01,-0.01]

        ## set default text alignment (right, top)
        h=1
        v=1
        ## check if custom text positions are available
        if textCorrection.has_key(loc):
            h,v=textCorrection[loc]
        
        effect=[path_effects.Stroke(linewidth=4, foreground='white'),path_effects.Stroke(linewidth=0.5, foreground='black')]
        ## plot location names at population centres, with corrections so as not to obscure it
        ax.text(lon+corrections[h],lat+corrections[v]*1.5,r'%s'%map_to_actual[loc],size=20,va=vas[v],ha=has[h],alpha=0.8,path_effects=effect,zorder=101)
#         ax.text(lon+corrections[h],lat+corrections[v]*1.5,r'%s'%map_to_actual[loc],size=20,va=vas[v],ha=has[h],alpha=0.8,zorder=101)

        storeCountry=country
print '\\hline'
print '\\end{longtable}'
    
for local_border in global_border:
    ax.plot(column(local_border,0),column(local_border,1),color='k',lw=2,zorder=98)
    ax.plot(column(local_border,0),column(local_border,1),color='w',lw=5,zorder=97)

translate={'SLE':'Sierra Leone','LBR':'Liberia','GIN':'Guinea',
           'MLI':'Mali','SEN':'Senegal','GNB':'Guinea-Bissau','CIV':"Cote d'Ivoire"}

for label in ['GrandBassa','Pujehun','Conakry']:
    x,y=popCentres[label]
    country=location_to_country[label]
    countryName=translate[country]
    c=colours[country](0.6)
    effect=[path_effects.Stroke(linewidth=6, foreground='white'),path_effects.Stroke(linewidth=2, foreground=c)]
    
    ax.text(x-0.3,y-0.2,'%s'%(countryName),size=50,path_effects=effect,va='top',ha='right',zorder=100)
#     ax.text(x-0.3,y-0.2,'%s'%(countryName),size=50,va='top',ha='right',zorder=100)
    
# colorbarTextSize=30
# colorbarTickLabelSize=24
# colorbarWidth=0.02
# colorbarHeight=0.35
# gap=0.07
# height=0.07

# ax2 = fig.add_axes([0.1+gap, height, colorbarWidth, colorbarHeight])
# mpl.colorbar.ColorbarBase(ax2, cmap=colours['GIN'],norm=mpl.colors.LogNorm(vmin=1,vmax=maxByCountry['GIN']))
# ax2.tick_params(labelcolor='k',size=0,labelsize=colorbarTickLabelSize)
# ax2.yaxis.set_label_position('left') 
# ax2.set_ylabel('Guinea',color='k',size=colorbarTextSize)

# ax3 = fig.add_axes([0.1+gap*2, height, colorbarWidth, colorbarHeight])
# mpl.colorbar.ColorbarBase(ax3, cmap=colours['LBR'],norm=mpl.colors.LogNorm(vmin=1,vmax=maxByCountry['LBR']))
# ax3.tick_params(labelcolor='k',size=0,labelsize=colorbarTickLabelSize)
# ax3.yaxis.set_label_position('left') 
# ax3.set_ylabel('Liberia',color='k',size=colorbarTextSize)

# ax4 = fig.add_axes([0.1+gap*3, height, colorbarWidth, colorbarHeight])
# mpl.colorbar.ColorbarBase(ax4, cmap=colours['SLE'],norm=mpl.colors.LogNorm(vmin=1,vmax=maxByCountry['SLE']))
# ax4.tick_params(labelcolor='k',size=0,labelsize=colorbarTickLabelSize)
# ax4.yaxis.set_label_position('left') 
# ax4.set_ylabel('Sierra Leone',color='k',size=colorbarTextSize)

## define which circle sizes to display as legend
# exampleSizes=[1,2,5,10,15,20,30,50,100,150,200]
# for i,j in enumerate(exampleSizes):
#     x=-12.80+np.log10(j+1)*0.1+sum(np.log10([q+5 for q in exampleSizes[:i]]))*0.22
#     y=4.7
    
#     ## add circle with specified radius
# #     circle_colour=coverage_colours((i)/float(len(exampleSizes)-1))
#     circle_colour=coverage_colours(1.0)
#     ax.add_collection(PatchCollection([patches.Circle((x,y),radius=np.log10(j+1)*0.1)],facecolor='none',edgecolor=circle_colour,lw=5,zorder=99,alpha=coverage_alpha))
    
#     ax.text(x,y+np.log10(j+1)*0.11,j,size=24,ha='center',va='bottom',zorder=100)
# #     if (i+1)<len(exampleSizes):
# #         ax.text(x,y-np.log10(j+2)*0.13,'%.2f'%(((i)/float(len(exampleSizes)-1))*float(seq_coverage)),size=24,rotation=90,ha='center',va='top',zorder=100)
# #     else:
# #         ax.text(x,y-np.log10(j+2)*0.13,'>%.2f'%(((i)/float(len(exampleSizes)-1))*float(seq_coverage)),size=24,rotation=90,ha='center',va='top',zorder=100)
    
## plot labelling of example circles
# ax.text(-12.80,4.7+np.log10(exampleSizes[int(len(exampleSizes)/2)])*0.45,'number of sequences',va='top',ha='left',size=24)

## plot labelling of example circles
#ax.text(-12.75,4.7-np.log10(exampleSizes[int(len(exampleSizes)/2)])*0.45,'sequences/case',va='top',ha='left',size=24)

## make plot pretty
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(size=0)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_ylim(ylimits)
ax.set_xlim(xlimits)

# plt.savefig(local_output+'sampling.png',dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'sampling.pdf',dpi=300,bbox_inches='tight')
plt.show()


## choose how many epi weeks to lump together
partitioning=4

## sort locations by country and normalized coordinate
sorted_locations=sorted(cases_byLocation.keys(),key=lambda x:(location_to_country[x],normalized_coords[x]))
## only keep those with reported cases
sorted_locations=[x for x in sorted_locations if sum(cases_byLocation[x].values())>0]

## find total number of cases each location had during the entire epidemic
total_cases=sum([sum([cases_byLocation[q][y] for y in dates]) for q in sorted_locations])
total_seqs=len(sequences)

print 'cases: %d\nsequences: %d'%(total_cases,total_seqs)


print [sum([cases_byLocation[q][y] for q in sorted_locations]) for y in dates]

xlabels=[]

dateRange=range(0,len(dates),partitioning)

matrix=np.zeros((len(sorted_locations),len(dateRange)))
matrix.fill(np.nan)

case_matrix=np.zeros((len(sorted_locations),len(dateRange)))
case_matrix.fill(np.nan)

seqs_matrix=np.zeros((len(sorted_locations),len(dateRange)))
seqs_matrix.fill(np.nan)


seen=[]

## normalize observed-expected counts by assuming a fixed sequencing capacity within each time bin
# assumeSequencingCapacity=True
assumeSequencingCapacity=False

## normalize observed-expected counts by accounting for the number of cases within each time bin
# assumeCaseNumbers=True
assumeCaseNumbers=False


accumulateExpected=0

## iterate through timeslices
for z,j in enumerate(range(0,len(dates),partitioning)):
    if j+partitioning>=len(dates):
        start=dates[j]
        end=dates[-1]
    else:
        start=dates[j]
        end=dates[j+partitioning]

    ## setup x tick labels
    xlabels.append('%s - %s'%(convertDate(start,'%Y-%m-%d','%Y-%b-%d'),convertDate(end,'%Y-%m-%d','%Y-%b-%d')))

    ## determines which epi weeks are included in a slice
    epiweeks=dates[j:j+partitioning]
    
    ## find cases the location had across lumped dates
    cases=[sum([cases_byLocation[q][y] for y in epiweeks]) for q in sorted_locations]
    for i,x in enumerate(cases):
        if x==0:
            case_matrix[i][z]=np.nan
        else:
            case_matrix[i][z]=x

    seqs=[sum([len(seq_byLocation[q][y]) for y in epiweeks]) for q in sorted_locations]
    for i,x in enumerate(seqs):
        if x==0:
            seqs_matrix[i][z]=np.nan
        else:
            seqs_matrix[i][z]=x
        
    ## total cases across locations at time slice
    cases_in_locations={loc:sum([cases_byLocation[loc][w] for w in epiweeks]) for loc in sorted_locations}
    
    ## cases in slice is the sum across a column
    cases_in_slice=sum(cases_in_locations.values())
    
    if cases_in_slice==0.0:
        cases_in_slice=np.nan

    ## fraction of total cases in timeslice at location - only used when assuming case numbers
    case_fraction={loc:cases_in_locations[loc]/float(cases_in_slice) for loc in sorted_locations}
    
    ## fraction of total cases across entire epidemic in location
    # i.e. normalizes the number of cases a location has at a time point by the total number of cases the location had across the entire epidemic
    total_case_fraction={loc:cases_in_locations[loc]/float(total_cases) for loc in sorted_locations}
    
    ## observed numbers of sequences in each location
    sequences_in_locations={loc:sum([len(seq_byLocation[loc][w]) for w in epiweeks]) for loc in sorted_locations}
    observed_seqs=sequences_in_locations
    
    ## total number of sequences in timeslice
    sequences_in_slice=sum(sequences_in_locations.values())
    
    ## redistribute all cases and all sequences according to situation at the time
    if assumeSequencingCapacity==True and assumeCaseNumbers==True:
        ## number of sequences in slice * fraction of total cases a location has in timeslice
        # i.e. given the sequencing capacity at the time, how many sequences should we have?
        expected_seqs={loc:sequences_in_slice*case_fraction[loc] for loc in sorted_locations}
    
    ## redistribute sequences in a slice according to proportion of total cases at location
    if assumeSequencingCapacity==True and assumeCaseNumbers==False:
        ## if we had to redistribute available sequences at the time according to total number of cases
        expected_seqs={loc:sequences_in_slice*total_case_fraction[loc] for loc in sorted_locations}

    ## redistribute all cases and all sequences homogenously through the epidemic
    if assumeSequencingCapacity==False and assumeCaseNumbers==False:    
        ## total number of sequences * fraction of all cases a location had during the epidemic
        #i.e. assuming a homogenous epidemic, how many sequences should we have?
        expected_seqs={loc:total_seqs*total_case_fraction[loc] for loc in sorted_locations}

    ## redistribute all sequences according to proportion of total cases at location
    if assumeSequencingCapacity==False and assumeCaseNumbers==True:
        # if we had to redistribute all sequences according to case numbers at each time point
        expected_seqs={loc:total_seqs/float(len(dates2))/float(partitioning)*case_fraction[loc] for loc in sorted_locations}

        
    ## observed-expected sequences
    OE=[observed_seqs[loc]-expected_seqs[loc] for loc in sorted_locations]
    
    #print sum(expected_seqs),sum(observed_seqs),start,end
    if np.isnan(sum(expected_seqs.values()))==False:
        accumulateExpected+=sum(expected_seqs.values())
    
    for i,x in enumerate(OE):
        ## mask out matrix if location had no cases at the time
        if case_fraction[sorted_locations[i]]==0:
            matrix[i][z]=np.nan
        else:
            matrix[i][z]=x

print accumulateExpected
## output sampling matrices
# output=open('/Users/admin/Downloads/EBOV_expectedSequences.csv','w')
# print>>output,',%s'%(','.join(xlabels))
# print>>output,'\n'.join('%s,%s'%(x,','.join(map(str,matrix[i]))) for i,x in enumerate(sorted_locations))
# output.close()

# output=open('/Users/admin/Downloads/EBOV_observedSequences.csv','w')
# print>>output,',%s'%(','.join(xlabels))
# print>>output,'\n'.join('%s,%s'%(x,','.join(map(str,seqs_matrix[i]))) for i,x in enumerate(sorted_locations))
# output.close()

# output=open('/Users/admin/Downloads/EBOV_observedCases.csv','w')
# print>>output,',%s'%(','.join(xlabels))
# print>>output,'\n'.join('%s,%s'%(x,','.join(map(str,case_matrix[i]))) for i,x in enumerate(sorted_locations))
# output.close()

## start figure
fig,ax = plt.subplots(figsize=(len(matrix[0])/2,20),facecolor='w')

## mask out NaNs
masked_array = np.ma.array(np.array(matrix),mask=np.isnan(matrix))

## redistribute all cases and all sequences according to situation at the time
if assumeSequencingCapacity==True and assumeCaseNumbers==True:
    print 'Calculating expected sequences by redistributing sequences available within a time window according to case numbers within a time window.'

if assumeSequencingCapacity==True and assumeCaseNumbers==False:
    print 'Calculating expected sequences by redistributing sequences available within a time window according to case numbers over the entire epidemic.'

if assumeSequencingCapacity==False and assumeCaseNumbers==True:
    print 'Calculating expected sequences by redistributing all available sequences according to case numbers within a time window.'

if assumeSequencingCapacity==False and assumeCaseNumbers==False:
    print 'Calculating expected sequences by redistributing all available sequences according to case numbers over the entire epidemic.'

    
## set ticks and tick labels
ax.set_xticks(np.arange(0.0,len(xlabels)+0.0))
ax.set_xticklabels(xlabels,rotation=90)
ax.set_yticks(np.arange(0.0,len(sorted_locations)+0.0))
ax.set_yticklabels([x for x in sorted_locations])

## set colourmap
cmap=mpl.cm.get_cmap('viridis')
cmap.set_bad('grey')

## plot Observed-Expected sequences
heatmap = ax.imshow(masked_array,interpolation='nearest',cmap=cmap,alpha=1)

## add colourbar
cbar = plt.colorbar(heatmap,ax=ax,shrink=0.5,aspect=30)
cbar.ax.yaxis.set_label_position('left') 
cbar.ax.set_ylabel('observed-expected sequences',size=24)
cbar.ax.tick_params(labelsize=24)

## make plot pretty
ax.tick_params(size=0,labelsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

for tick in ax.yaxis.get_ticklabels():
    label=str(tick.get_text())
    location=label
    country=location_to_country[location]
    tick.set_text(location)
    tick.set_color(desaturate(colours[country](normalized_coords[location]),0.7))

ax.set_yticklabels(sorted_locations)

# plt.savefig(local_output+'EBOV_sampling_heatmap_combine%02d.png'%(partitioning),dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'EBOV_sampling_heatmap_combine%02d.pdf'%(partitioning),dpi=300,bbox_inches='tight')


plt.show()

## start figure
fig,ax = plt.subplots(figsize=(len(matrix[0])/2,20),facecolor='w')

## mask out NaNs
masked_array = np.ma.array(np.array(case_matrix),mask=np.isnan(case_matrix))

## set up ticks and tick labels for the plot
ax.set_xticks(np.arange(0.0,len(xlabels)+0.0))
ax.set_xticklabels(xlabels,rotation=90)
ax.set_yticks(np.arange(0.0,len(sorted_locations)+0.0))
ax.set_yticklabels([x for x in sorted_locations])

## set colourmap
cmap=mpl.cm.get_cmap('viridis')
cmap.set_bad('grey')
## plot cases
heatmap = ax.imshow(masked_array,interpolation='nearest',cmap=cmap,alpha=1,norm=mpl.colors.LogNorm())

## add colourbar
cbar = plt.colorbar(heatmap,ax=ax,shrink=0.5,aspect=30)
cbar.ax.yaxis.set_label_position('left') 
cbar.ax.set_ylabel('cases',size=24)
cbar.ax.tick_params(labelsize=24)

## make plot pretty
ax.tick_params(size=0,labelsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

for tick in ax.yaxis.get_ticklabels():
    label=str(tick.get_text())
    location=label
    country=location_to_country[location]
    tick.set_text(location)
    tick.set_color(desaturate(colours[country](normalized_coords[location]),0.7))

ax.set_yticklabels(sorted_locations)

# plt.savefig(local_output+'EBOV_cases_heatmap_combine%02d.png'%(partitioning),dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'EBOV_cases_heatmap_combine%02d.pdf'%(partitioning),dpi=300,bbox_inches='tight')

plt.show()


## start figure
fig,ax = plt.subplots(figsize=(len(matrix[0])/2,20),facecolor='w')

## mask out NaNs
masked_array = np.ma.array(np.array(seqs_matrix),mask=np.isnan(seqs_matrix))

## set up ticks and tick labels for the plot
ax.set_xticks(np.arange(0.0,len(xlabels)+0.0))
ax.set_xticklabels(xlabels,rotation=90)
ax.set_yticks(np.arange(0.0,len(sorted_locations)+0.0))
ax.set_yticklabels([x for x in sorted_locations])

## set colourmap
cmap=mpl.cm.get_cmap('viridis')
cmap.set_bad('grey')
## plot cases
heatmap = ax.imshow(masked_array,interpolation='nearest',cmap=cmap,alpha=1,norm=mpl.colors.Normalize())

## add colourbar
cbar = plt.colorbar(heatmap,ax=ax,shrink=0.5,aspect=30)
cbar.ax.yaxis.set_label_position('left') 
cbar.ax.set_ylabel('sequences',size=24)
cbar.ax.tick_params(labelsize=24)

## make plot pretty
ax.tick_params(size=0,labelsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

for tick in ax.yaxis.get_ticklabels():
    label=str(tick.get_text())
    location=label
    country=location_to_country[location]
    tick.set_text(location)
    tick.set_color(desaturate(colours[country](normalized_coords[location]),0.7))

ax.set_yticklabels(sorted_locations)
    
# plt.savefig(local_output+'EBOV_sequences_heatmap_combine%02d.png'%(partitioning),dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'EBOV_sequences_heatmap_combine%02d.pdf'%(partitioning),dpi=300,bbox_inches='tight')

plt.show()


from scipy.stats import spearmanr

# start figure
fig,ax = plt.subplots(figsize=(15,15),facecolor='w')

c={x:sum(cases_byLocation[x].values())+1 for x in cases_byLocation}
s={x:sum([len(y) for y in seq_byLocation[x].values()])+1 for x in seq_byLocation}

vas=['bottom','top']
has=['left','right']
corrections=[0.01,-0.01]

for loc in cases_byLocation:
    country=location_to_country[loc]
    countryColour=colours[country]

    ## set default text alignment (right, top)
    h=1
    v=1
    ## check if custom text positions are available
    if textCorrection.has_key(loc):
        h,v=textCorrection[loc]
    
#     print loc,c[loc],s[loc]
    if s[loc]>0 and c[loc]>0:
        k=countryColour(normalized_coords[loc])
#         k=countryColour(0.5)
        ax.scatter(c[loc],s[loc],s=200,facecolor=k,edgecolor='none',zorder=100)
        ax.scatter(c[loc],s[loc],s=300,facecolor='k',edgecolor='none',zorder=99)
        effects=[path_effects.Stroke(linewidth=4, foreground='white'),path_effects.Stroke(linewidth=0.5, foreground=countryColour(normalized_coords[loc]))]
#         ax.text(c[loc],s[loc],'%s'%(map_to_actual[location]),size=14,va=vas[v],ha=has[h],path_effects=effects,alpha=1)
        #ax.text(c[loc],s[loc],'%s'%(map_to_actual[loc]),size=14,va='top',ha='right',path_effects=effects,alpha=1)
    
xs=[c[x] for x in cases_byLocation]
ys=[s[x] for x in cases_byLocation]

r,pval=spearmanr(xs,ys)

print 'Spearman correlation coefficient: %.2f\np-value: %.4f'%(r,pval)

ax.set_xlim(0.9,3600)
ax.set_ylim(0.9,190)

ax.set_yscale('log')
ax.set_xscale('log')

for label in ax.get_yticklabels() :
    label.set_fontproperties(typeface)

ax.tick_params(size=8,which='major',labelsize=28)
ax.tick_params(size=5,which='minor',labelsize=28)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

ax.set_xlabel('Number of cases (+1)',size=32)
ax.set_ylabel('Number of sequences (+1)',size=32)
ax.grid(ls='--',which='major')

ax.set_aspect(1)

# plt.savefig(local_output+'samplingCorrelations.png',dpi=300,bbox_inches='tight')
# plt.savefig(local_output+'samplingCorrelations.pdf',dpi=300,bbox_inches='tight')

plt.show()


