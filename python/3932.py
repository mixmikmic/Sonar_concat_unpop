# This notebook will largely just pull code from the `map_exploration_days0.ipynb` located in this folder. I've recently made the decision to label forest-fires as any observations that fall within 500m of a forest-fire perimter on their date, or up to 3 days in the future. Now, it's time to take a little bit more in-depth look at how that affected the geographical distribution of forest-fires. I've already done this once with another dataset (see `map_exploration_days0.ipynb`), which is why most of the code is pulled from there. See that notebook for a more detailed explanation of what the code is doing. 
# 
# I'm largely looking here to make sure that everything makes sense, and nothing jumps out as incorrect or illogical. I'll be looking at plots of overall fires across states and years, as well as taking a more detailed look at some counties and individual days (following the same structure as in `map_exploration_days0.ipynb`). 
# 

import pandas as pd
from dsfuncs.geo_plotting import USMapBuilder
import matplotlib.pyplot as plt
import fiona
import subprocess
import datetime
get_ipython().magic('matplotlib inline')


# The first step will be to create functions to read in the data, and then only grab those rows of the data set that correspond to a given location (state & county) as well as a given month (or months). 
# 

def read_df(year, modis=True): 
    """This function will read in a year of data, and add a month column. 
    
    Args: 
        year: str
        modis: bool
            Whether to use the modis or viirs data for plotting. 
        
    Return:
        Pandas DataFrame
    """
    if modis: 
        output_df = pd.read_csv('../../../data/csvs/day3_500m/detected_fires_MODIS_' + str(year) + '.csv', 
                                parse_dates=['date'], true_values=['t'], false_values=['f'])
    else: 
         output_df = pd.read_csv('../../../data/csvs/day3_500m/detected_fires_VIIRS_' + str(year) + '.csv', 
                                parse_dates=['date'], true_values=['t'], false_values=['f'])
    output_df['month'] = output_df.date.apply(lambda dt: dt.strftime('%B'))
    output_df.dropna(subset=['region_name'], inplace=True) # These will be obs. in Canada. 
    return output_df
    
def grab_by_location(df, state_names, county_names=None): 
    """Grab the data for a specified inputted state and county. 
    
    Args: 
        df: Pandas DataFrame
        state: set (or iterable of strings)
            State to grab for plotting.
        county: set (or iterable of strings) (optional)
            County names to grab for plotting. If None, simply grab the 
            entire state to plot. 
            
    Return: 
        Pandas DataFrame
    """
    if county_names: 
        output_df = df.query('state_name in @state_names and county_name in @county_names')
    else: 
        output_df = df.query('state_name in @state_names')
    return output_df

def grab_by_date(df, months=None, dt=None): 
    """Grab the data for a set of specified months.
    
    Args: 
        df: Pandas DataFrame
        months: set (or iterable of strings)
    
    Return: 
        Pandas DataFrame
    """
    if months is not None: 
        output_df = df.query("month in @months")
    else: 
        split_dt = dt.split('-')
        year, month, dt = int(split_dt[0]), int(split_dt[1]), int(split_dt[2])
        match_dt = datetime.datetime(year, month, dt, 0, 0, 0)
        output_df = df.query('date == @match_dt')
    return output_df


# The next step is to just parse the data to get it into a format to plot. The format that the `USMapBuilder` class will expect the data to be in is an iterable of three items: 
# 
# 1. Longitude of the point. 
# 2. Latitude of the point. 
# 3. Color to plot the point in. 
# 
# I'll create a function that will take in the previously parsed location/date DataFrame and get the data set up to be in that format. 
# 

def format_df(df): 
    """Format the data to plot it on maps. 
    
    This function will grab the latitude and longitude 
    columns of the DataFrame, and return those, along 
    with a third column that will be newly generated. This 
    new column will hold what color we want to use to plot 
    the lat/long coordinate - I'll use red for fire and 
    green for non-fire. 
    
    Args: 
        df: Pandas DataFrame
    
    Return: 
        numpy.ndarray
    """
    
    keep_cols = ['long', 'lat', 'fire_bool']
    intermediate_df = df[keep_cols]
    output_df = parse_fire_bool(intermediate_df)
    output_array = output_df.values
    return output_array

def parse_fire_bool(df): 
    """Parse the fire boolean to a color for plotting. 
    
    Args: 
        df: Pandas DataFrame
        
    Return: 
        Pandas DataFrame
    """
    
    # Plot actual fires red and non-fires green. 
    output_df = df.drop('fire_bool', axis=1)
    output_df['plotting_mark'] = df['fire_bool'].apply(lambda f_bool: 'ro' if f_bool == True else 'go')
    return output_df


# Now let's put all of this into a master function. 
# 

def read_n_parse(year, state_names, county_names=None, months=None, plotting=False, dt=None): 
    """Read and parse the data for plotting.
    
    Args: 
        year: str
        state_names: set (or other iterable) of state names
            State names to grab for plotting. 
        county_names: set (or other iterable) of county names (optional)
            County names to grab for plotting. 
        months: months (or other iterable) of months (optional)
            Month names to grab for plotting. 
        plotting: bool 
            Whether or not to format the data for plotting. 
            
    Return: 
        Pandas DataFrame
    """
    
    fires_df = read_df(year)
    if state_names: 
        fires_df = grab_by_location(fires_df, state_names, county_names)
    
    if months or dt: 
        fires_df = grab_by_date(fires_df, months, dt)
    
    if plotting: 
        fires_df = format_df(fires_df)
    return fires_df

def plot_states(year, state_names, months=None, plotting=True): 
    """Plot a state map and the given fires data points for that state. 
    
    Args: 
        year: str
        state_names: set (or other iterable) of state names
            State names to grab for plotting. 
        months: set (or other iterable) of month names 
            Month names to grab for plotting. 
        plotting: bool 
            Whether or not to format the data for plotting. 
    
    Return: Plotted Basemap
    """
    ax = plt.subplot(1, 2, 1)
    state_map = USMapBuilder('data/state_shapefiles_2014/cb_2014_us_state_500k2.shp', geo_level='State', 
                        state_names=state_names, ax=ax, border_padding=1)
    fires_data = read_n_parse(year, state_names, months=months, plotting=plotting)
    fires_data_trues = fires_data[fires_data[:,2] == 'ro']
    fires_data_falses = fires_data[fires_data[:,2] == 'go']
    print fires_data_trues.shape, fires_data_falses.shape
    state_map.plot_points(fires_data_trues)
    ax = plt.subplot(1, 2, 2)
    state_map = USMapBuilder('data/state_shapefiles_2014/cb_2014_us_state_500k2.shp', geo_level='State',  
                        state_names=state_names, ax=ax, border_padding=1)
    state_map.plot_points(fires_data_falses)
    plt.show()


years = ['2012', '2013', '2014', '2015']
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 
         'November', 'December']


state_names = ['California']
for month in months: 
    for year in years: 
        print 'Year: {}, Month: {}'.format(year, month)
        plot_states(year, state_names, months=[month])


state_names = ['Colorado']
for month in months: 
    for year in years: 
        print 'Year: {}, Month: {}'.format(year, month)
        plot_states(year, state_names, months=[month])


state_names = ['Montana']
for month in months: 
    for year in years: 
        print 'Year: {}, Month: {}'.format(year, month)
        plot_states(year, state_names, months=[month])


state_names = ['Washington']
for month in months: 
    for year in years: 
        print 'Year: {}, Month: {}'.format(year, month)
        plot_states(year, state_names, months=[month])


state_names = ['Texas']
for month in months: 
    for year in years: 
        print 'Year: {}, Month: {}'.format(year, month)
        plot_states(year, state_names, months=[month])


state_names = ['California']
for year in years: 
    print 'Year: {}'.format(year)
    plot_states(year, state_names)


state_names = ['Colorado']
for year in years: 
    print 'Year: {}'.format(year)
    plot_states(year, state_names)


state_names = ['Montana']
for year in years: 
    print 'Year: {}'.format(year)
    plot_states(year, state_names)


state_names = ['Washington']
for year in years: 
    print 'Year: {}'.format(year)
    plot_states(year, state_names)


state_names = ['Texas']
for year in years: 
    print 'Year: {}'.format(year)
    plot_states(year, state_names)


# In looking at the maps for each year and comparing them to the ground truth as labeling obs that are forest-fires only if they fall within a fire-perimeter boundary on that day, there are some clear differences. At this point, I'm going to just explore five counties from each of the states above, for some subset of dates (making sure to try to look at obs. that are both labeled forest-fires and non forest-fires). I'm doing this to try to further explore the data and make sure nothing looks off or amiss. 
# 
# First, I'll take the code from the other notebook. 
# 

def grab_fire_perimeters(dt, st_name, st_abbrev, st_fips, county=False): 
    """Grab the fire perimter boundaries for a given state and year.
    
    Args: 
        dt: str
        st_name: str
            State name of boundaries to grab. 
        st_abbrev: str
            State abbreviation used for bash script. 
        st_fips: str
            State fips used for bash script. 
            
    Return: Shapefile features. 
    """
            
    # Run bash script to pull the right boundaries from Postgres. 
    subprocess.call("./grab_perim_boundary.sh {} {} {}".format(st_abbrev, st_fips, dt), shell=True)
    
    filepath = 'data/{}/{}_{}_2D'.format(st_abbrev, st_abbrev, dt)
    return filepath

def plot_st_fires_boundaries(dt, st_name, st_abbrev, st_fips): 
    """Plot the fire boundaries for a year and given state.
    
    Args
    ----
        dt: str
            Contains date of boundaries to grab. 
        state_name: str
            Holds state to plot. 
        st_abbrev: str
            Holds the states two-letter abbreviation. 
        st_fips: str
            Holds the state's fips number. 
        
    Return: Plotted Basemap
    """
    
    boundaries_filepath = grab_fire_perimeters(dt, st_name, st_abbrev, st_fips)
    st_map = USMapBuilder('data/state_shapefiles_2014/cb_2014_us_state_500k2.shp', geo_level='State', 
                          state_names=[st_name], border_padding=1)
    st_map.plot_boundary(boundaries_filepath)
    plt.show()
    
def plot_counties_fires_boundaries(year, state_name, st_abbrev, st_fips, county_name, 
                                   months=None, plotting=True, markersize=4): 
    """Plot a county map in a given state, including any fire perimeter boundaries and potentially
    detected fires in those counties. 
    
    Args: 
        year: str
        state_name: str
            State name to grab for plotting. 
        st_abbrev: str
            Holds the states two-letter abbrevation
        st_fips: str
            Holds the state's fips number. 
        county_name: str or iterable of strings 
            County names to grab for plotting. 
        months: set (or other iterable) of strings (optional)
            Month names to grab for plotting
        plotting: bool
            Whether or not to format the data for plotting. 
    """
    
    county_name = county_name if isinstance(county_name, list) else [county_name]
    county_map = USMapBuilder('data/county_shapefiles_2014/cb_2014.shp', geo_level='County', 
                state_names=[state_name], county_names=county_name, figsize=(40, 20), border_padding=0.1)
    boundaries_filepath = grab_fire_perimeters(year, state_name, st_abbrev, st_fips)
    county_map.plot_boundary(boundaries_filepath)
    if plotting: 
        fires_data = read_n_parse(year, state_name, months=months, plotting=plotting)
        county_map.plot_points(fires_data, markersize)
    plt.show()


def plot_county_dt(st_name, st_abbrev, st_fips, county_name, fires_dt=None, perims_dt=None, 
                  markersize=4): 
    """Plot all obs., along with the fire-perimeter boundaries, for a given county and date. 
    
    Read in the data for the inputted year and parse it to the given state/county and date. 
    Read in the fire perimeter boundaries for the given state/county and date, and parse 
    those. Plot it all. 
    
    Args: 
        st_name: str
            State name to grab for plotting. 
        st_abbrev: str
            State abbrevation used for the bash script. 
        st_fips: str
            State fips used for the base script. 
        county_name: str
            County name to grab for plotting. 
        dt: str
            Date to grab for plotting. 
        markersize (optional): int
            Used to control the size of marker to use for plotting fire points. 
    """
    
    year = fires_dt.split('-')[0]
    perims_dt = fires_dt if not perims_dt else perims_dt
    county_names = [county_name] if not isinstance(county_name, list) else county_name
    fires_data = read_n_parse(year, state_names=st_name, county_names=county_name, dt=fires_dt, plotting=True)
    county_map = USMapBuilder('data/county_shapefiles_2014/cb_2014.shp', geo_level='County', 
                state_names=[st_name], county_names=county_names, figsize=(40, 20), border_padding=0.1)
    fire_boundaries = grab_fire_perimeters(perims_dt, st_name, st_abbrev, st_fips)
    try: 
        county_map.plot_boundary(fire_boundaries)
    except Exception as e:
        print e
    print len(fires_data)
    county_map.plot_points(fires_data, markersize=markersize)
    plt.show()


for dt in xrange(1, 31): 
    dt = '2015-06-0' + str(dt) if dt < 10 else '2015-06-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Colorado', 'co', '08', 'Moffat', dt)


for dt in xrange(1, 31): 
    dt = '2012-06-0' + str(dt) if dt < 10 else '2012-06-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Colorado', 'co', '08', 'Mesa', dt)


for dt in xrange(1, 31): 
    dt = '2013-07-0' + str(dt) if dt < 10 else '2013-07-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Colorado', 'co', '08', ['Hinsdale', 'Mineral'], dt)


for dt in xrange(1, 31): 
    dt = '2015-07-0' + str(dt) if dt < 10 else '2015-07-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Colorado', 'co', '08', 'Logan', dt)


for dt in xrange(1, 31): 
    dt = '2014-08-0' + str(dt) if dt < 10 else '2014-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Colorado', 'co', '08', 'Las Animas', dt)


for dt in xrange(1, 31): 
    dt = '2014-01-0' + str(dt) if dt < 10 else '2014-01-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', 'Los Angeles', dt)


for dt in xrange(1, 29): 
    dt = '2014-02-0' + str(dt) if dt < 10 else '2014-02-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', ['Glenn'], dt)


for dt in xrange(1, 31): 
    dt = '2013-05-0' + str(dt) if dt < 10 else '2013-05-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', ['Ventura'], dt, markersize=2)


for dt in xrange(1, 31): 
    dt = '2012-07-0' + str(dt) if dt < 10 else '2012-07-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', ['Fresno'], dt, markersize=4)


for dt in xrange(1, 31): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', ['Mono'], dt, markersize=4)


for dt in xrange(1, 31): 
    dt = '2012-07-0' + str(dt) if dt < 10 else '2012-07-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Montana', 'mt', '30', ['Big Horn'], dt, markersize=4)


for dt in xrange(1, 31): 
    dt = '2013-08-0' + str(dt) if dt < 10 else '2013-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Montana', 'mt', '30', ['Ravalli'], dt, markersize=4)


for dt in xrange(1, 31): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Montana', 'mt', '30', ['Lewis and Clark'], dt, markersize=4)


for dt in xrange(1, 31): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Montana', 'mt', '30', ['Pondera'], dt, markersize=4)


for dt in xrange(1, 31): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Montana', 'mt', '30', ['Lake', 'Missoula', 'Ravalli', 'Mineral', 'Sanders'], dt, markersize=4)


for dt in xrange(1, 31): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Washington', 'wa', '53', ['Columbia', 'Garfield'], dt, markersize=4)


for dt in xrange(1, 31): 
    dt = '2012-08-0' + str(dt) if dt < 10 else '2012-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Washington', 'wa', '53', ['Asotin'], dt, markersize=4)


for dt in xrange(1, 31): 
    dt = '2014-03-0' + str(dt) if dt < 10 else '2014-03-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Washington', 'wa', '53', ['Walla Walla', 'Franklin', 'Benton'], dt, markersize=4)


for dt in xrange(1, 31): 
    dt = '2013-07-0' + str(dt) if dt < 10 else '2013-07-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Washington', 'wa', '53', ['Klickitat'], dt, markersize=4)


for dt in xrange(1, 31): 
    dt = '2014-08-0' + str(dt) if dt < 10 else '2014-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Washington', 'wa', '53', ['Mason'], dt, markersize=4)


for dt in xrange(1, 31): 
    dt = '2012-01-0' + str(dt) if dt < 10 else '2012-01-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Texas', 'tx', '48', ['Cameron', 'Hidalgo', 'Willacy'], dt, markersize=4)


for dt in xrange(1, 31): 
    dt = '2014-04-0' + str(dt) if dt < 10 else '2014-04-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Texas', 'tx', '48', ['Presidio', 'Brewster', 'Jeff Davis'], dt, markersize=4)





# It has become fairly clear that I should at least look into labeling the ground truth a different way. With other maps, I can see that some observations aren't labeled as forest-fires that probably should be, or at least from a business/real-world perspective. I can see that some observations aren't labeled as forest-fires because they don't fall within a fire-perimeter boundary on that given day, but they do fall within a fire-perimeter boundary days later. In terms of training a model, I think it's reasonable that we should train it to identify these observations as forest fires. Given that they do fall within a fire-perimter boundary within a couple of days, it's highly likely that they were forest-fire observations, and either fire-perimeter boundaries weren't submitted for that day for that area, or the fire wasn't seen (meaning no perimeter boundary drawn around it). In either case, I think it's reasonable to label these as forest-fires and see. 
# 
# Another case where it's reasonable to label observations as forest-fires is for those that fall near a fire perimeter boundary (e.g. within 100m, 250m, etc.). The documentation for the fire perimeter boundaries states that they may not be the most accurate, and from exploring the data it looks fairly clearly like some observations that are just outside boundaries should be labeled as forest-fires. 
# 
# In this notebook, I'll be exploring what I label as forest-fires/non-forest-fires if I go out 0, 1, 3, 5, and 7 days ahead to look for fire perimeter boundaries that an ob. falls in, and if I extend the fire-perimeter boundaries by 100m, 250m, and 500m. 
# 

import pandas as pd
import matplotlib.pyplot as plt
import fiona
import subprocess
import datetime
from dsfuncs.geo_plotting import USMapBuilder
from dsfuncs.dist_plotting import plot_var_dist, plot_binary_response
get_ipython().magic('matplotlib inline')


def read_df(year, days_ahead): 
    """This function will read in a year of data, and add a month column. 
    
    Args: 
        year: str
        days_ahead: int
            Used to determine what data to read in.  
        
    Return:
        Pandas DataFrame
    """
    output_df = pd.read_csv('../../../data/csvs/day{}/detected_fires_MODIS_'.format(days_ahead) + str(year) + '.csv', 
            parse_dates=['date'], true_values=['t'], false_values=['f'])
    output_df['month'] = output_df.date.apply(lambda dt: dt.strftime('%B'))
    output_df.dropna(subset=['region_name'], inplace=True) # These will be obs. in Canada. 
    return output_df
    
def grab_by_location(df, state_names, county_names=None): 
    """Grab the data for a specified inputted state and county. 
    
    Args: 
        df: Pandas DataFrame
        state: set (or iterable of strings)
            State to grab for plotting.
        county: set (or iterable of strings) (optional)
            County names to grab for plotting. If None, simply grab the 
            entire state to plot. 
            
    Return: 
        Pandas DataFrame
    """
    if county_names: 
        output_df = df.query('state_name in @state_names and county_name in @county_names')
    else: 
        output_df = df.query('state_name in @state_names')
    return output_df

def grab_by_date(df, months=None, dt=None): 
    """Grab the data for a set of specified months.
    
    Args: 
        df: Pandas DataFrame
        months: set (or iterable of strings)
    
    Return: 
        Pandas DataFrame
    """
    if months is not None: 
        output_df = df.query("month in @months")
    else: 
        split_dt = dt.split('-')
        year, month, dt = int(split_dt[0]), int(split_dt[1]), int(split_dt[2])
        match_dt = datetime.datetime(year, month, dt, 0, 0, 0)
        output_df = df.query('date == @match_dt')
    return output_df

def format_df(df): 
    """Format the data to plot it on maps. 
    
    This function will grab the latitude and longitude 
    columns of the DataFrame, and return those, along 
    with a third column that will be newly generated. This 
    new column will hold what color we want to use to plot 
    the lat/long coordinate - I'll use red for fire and 
    green for non-fire. 
    
    Args: 
        df: Pandas DataFrame
    
    Return: 
        numpy.ndarray
    """
    
    keep_cols = ['long', 'lat', 'fire_bool']
    intermediate_df = df[keep_cols]
    output_df = parse_fire_bool(intermediate_df)
    output_array = output_df.values
    return output_array

def parse_fire_bool(df): 
    """Parse the fire boolean to a color for plotting. 
    
    Args: 
        df: Pandas DataFrame
        
    Return: 
        Pandas DataFrame
    """
    
    # Plot actual fires red and non-fires green. 
    output_df = df.drop('fire_bool', axis=1)
    output_df['plotting_mark'] = df['fire_bool'].apply(lambda f_bool: 'ro' if f_bool == True else 'go')
    return output_df

def read_n_parse(year, state_names, county_names=None, months=None, plotting=False, dt=None, days_ahead=0): 
    """Read and parse the data for plotting.
    
    Args: 
        year: str
        state_names: set (or other iterable) of state names
            State names to grab for plotting. 
        county_names: set (or other iterable) of county names (optional)
            County names to grab for plotting. 
        months: months (or other iterable) of months (optional)
            Month names to grab for plotting. 
        plotting: bool 
            Whether or not to format the data for plotting. 
            
    Return: 
        Pandas DataFrame
    """
    
    fires_df = read_df(year, days_ahead)
    if state_names: 
        fires_df = grab_by_location(fires_df, state_names, county_names)
    
    if months or dt: 
        fires_df = grab_by_date(fires_df, months, dt)
    
    if plotting: 
        fires_df = format_df(fires_df)
    return fires_df

def grab_fire_perimeters(dt, st_name, st_abbrev, st_fips, county=False): 
    """Grab the fire perimter boundaries for a given state and year.
    
    Args: 
        dt: str
        st_name: str
            State name of boundaries to grab. 
        st_abbrev: str
            State abbreviation used for bash script. 
        st_fips: str
            State fips used for bash script. 
            
    Return: Shapefile features. 
    """
            
    # Run bash script to pull the right boundaries from Postgres. 
    subprocess.call("./grab_perim_boundary.sh {} {} {}".format(st_abbrev, st_fips, dt), shell=True)
    
    filepath = 'data/{}/{}_{}_2D'.format(st_abbrev, st_abbrev, dt)
    return filepath

def plot_county_dt(st_name, st_abbrev, st_fips, county_name, fires_dt=None, perims_dt=None, 
                  days_ahead=0): 
    """Plot all obs., along with the fire-perimeter boundaries, for a given county and date. 
    
    Read in the data for the inputted year and parse it to the given state/county and date. 
    Read in the fire perimeter boundaries for the given state/county and date, and parse 
    those. Plot it all. 
    
    Args: 
        st_name: str
            State name to grab for plotting. 
        st_abbrev: str
            State abbrevation used for the bash script. 
        st_fips: str
            State fips used for the base script. 
        county_name: str
            County name to grab for plotting. 
        dt: str
            Date to grab for plotting. 
    """
    
    year = fires_dt.split('-')[0]
    perims_dt = fires_dt if not perims_dt else perims_dt
    county_names = [county_name] if not isinstance(county_name, list) else county_name
    fires_data = read_n_parse(year, state_names=st_name, county_names=county_name, dt=fires_dt, plotting=True,
                             days_ahead=days_ahead)
    county_map = USMapBuilder('data/county_shapefiles_2014/cb_2014.shp', geo_level='County', 
                state_names=[st_name], county_names=county_names, figsize=(40, 20), border_padding=0.1)
    fire_boundaries = grab_fire_perimeters(perims_dt, st_name, st_abbrev, st_fips)
    try: 
        county_map.plot_boundary(fire_boundaries)
    except Exception as e:
        print e
    county_map.plot_points(fires_data, 4)
    plt.show()


# Let's start off by plotting the original data. There were quite a few fires in Trinity county, so we'll look at that again. After that, we'll plot the fires for a fixed day (the 1st) and the boundaries for future days. This should give us a reasonable idea of how 
# obs. label's change. 
# 

for dt in xrange(1, 9): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', 'Trinity', dt, days_ahead=0)


for days_ahead in (0, 1, 3, 5, 7): 
    dt = '2015-08-0' + str(days_ahead + 1) if days_ahead < 10 else '2015-08-' + str(days_ahead + 1)
    print 'Days Ahead: {}'.format(days_ahead)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', 'Trinity', '2015-08-01', dt, days_ahead=days_ahead)
    plt.clf()


# From the above, we can see that the merge looks to be labeling some observations as forest-fires based off later boundaries, but that some of the merging isn't happening correctly. 
# 
# After looking into this, and trying a number of different merges in PostgreSQL (~10) using their PostGIS extension (each of which gave me the exact same results), I have no good explanation for this. Some forums online have suggested that the PostGIS extension can be a little finicky. Given that I'll also be extending the boundaries by some distance, I'm going to look into that and see what effect that has. Before doing so, I'd still like to see how these above merging changes the number of observations labeled as forest-fires, and the distributions of some of the variables. 
# 

for year in xrange(2012, 2016): 
    print 'Year: {}'.format(year)
    print '-' * 50
    for days_forward in (0, 1, 3, 5, 7): 
        print 'Days Forward: {}'.format(days_forward)
        print '=' * 50
        df = read_df(year, days_forward)
        df.info()
        print '=' * 50


for year in xrange(2012, 2016): 
    print 'Year: {}'.format(year)
    print '-' * 50
    for days_forward in (0, 1, 3, 5, 7): 
        print 'Days Forward: {}'.format(days_forward)
        df = read_df(year, days_forward)
        print df.fire_bool.mean(), df.fire_bool.sum()
    print '\n' * 2


continous_vars = ('lat', 'long', 'gmt', 'temp', 'spix', 'tpix', 'conf', 'frp', 'county_aland', 'county_awater')
categorical_vars = ('urban_areas_bool', 'src', 'sat_src')


def check_dists(year, continous_vars, categorical_vars): 
    """Plot the distributions of varaibles for the inputted year. 
    
    Read in the data for the inputted year. Then, take the inputted 
    variable names in the continous_vars and categorical_vars parameters, 
    and plot their distributions. Do this separately for observations 
    labeled as forest-fires and those labeled as non forest-fires. 
    
    Args: 
        year: int
            Holds the year of data to use for plotting. 
        continous_vars: tuple (or other iterable) of strings
            Holds the names of the continuous variables to use for plotting. 
        categorical_vars: tuple (or other iterable) of strings. 
            Holds the names of the categorical variables to use for plotting. 
    """
    dfs = []
    for days_forward in (0, 1, 3, 5, 7): 
        dfs.append(read_df(year, days_forward))
    
    fires = []
    non_fires = []
    for df in dfs: 
        fires.append(df.query('fire_bool == 1'))
        non_fires.append(df.query('fire_bool == 0'))
        
    print 'Continuous Vars'
    print '-' * 50
    for var in continous_vars: 
        print 'Variable: {} : Fires, then non-fires'.format(var)
        print '-' * 50
        for idx, df in enumerate(dfs): 
            f, axes = plt.subplots(1, 8, figsize=(20, 5))
            fire_var = fires[idx][var]
            non_fire_var = non_fires[idx][var]
            print fire_var.mean(), non_fire_var.mean()
            plot_var_dist(fire_var, categorical=False, ax=axes[0:4], show=False)
            plot_var_dist(non_fire_var, categorical=False, ax=axes[4:], show=False)
            plt.show()
    print 'Categorical Vars'
    print '-' * 50
    for var in categorical_vars: 
        print 'Variable: {} : Fires, then non-fires'.format(var)
        print '-' * 50
        for idx, df in enumerate(dfs):
            f, axes = plt.subplots(1, 2)
            plot_var_dist(fires[idx][var], categorical=True, ax=axes[0], show=False)
            plot_var_dist(non_fires[idx][var], categorical=True, ax=axes[1], show=False)
            plt.show()


check_dists(2012, continous_vars, categorical_vars)


check_dists(2013, continous_vars, categorical_vars)


check_dists(2014, continous_vars, categorical_vars)


check_dists(2015, continous_vars, categorical_vars)


# In the end, none of the distributions across days forward look that different (they aren't really noticeably different). Now, on to looking at what happens if we expand out the fire perimeter boundaries. 
# 

def read_df(year, meters_nearby=None): 
    """This function will read in a year of data, and add a month column. 
    
    Args: 
        year: str
        meters_nearby (optional): int
            How far we go out from the original boundaries to label
            forest-fires. Used to determine what data to read in. If 
            not passed in, use the original, raw data (stored in the days0 
            folder). 
        
    Return:
        Pandas DataFrame
    """
    if not meters_nearby: 
        output_df = pd.read_csv('../../../data/csvs/day0/detected_fires_MODIS_' + str(year) + '.csv', 
                parse_dates=['date'], true_values=['t'], false_values=['f'])
    else: 
        output_df = pd.read_csv('../../../data/csvs/fires_{}m/detected_fires_MODIS_'.format(meters_nearby) 
                        + str(year) + '.csv', parse_dates=['date'], true_values=['t'], 
                        false_values=['f'])
    output_df['month'] = output_df.date.apply(lambda dt: dt.strftime('%B'))
    output_df.dropna(subset=['region_name'], inplace=True) # These will be obs. in Canada. 
    return output_df

def read_n_parse(year, state_names, county_names=None, months=None, plotting=False, 
                 dt=None, meters_nearby=0): 
    """Read and parse the data for plotting.
    
    Args: 
        year: str
        state_names: set (or other iterable) of state names
            State names to grab for plotting. 
        county_names: set (or other iterable) of county names (optional)
            County names to grab for plotting. 
        months: months (or other iterable) of months (optional)
            Month names to grab for plotting. 
        plotting: bool 
            Whether or not to format the data for plotting. 
        meters_nearby: int
            How many meters to go out when labeleing fires. Used to figure out
            what data to read in. 
            
    Return: 
        Pandas DataFrame
    """
    
    fires_df = read_df(year, meters_nearby)
    if state_names: 
        fires_df = grab_by_location(fires_df, state_names, county_names)
    
    if months or dt: 
        fires_df = grab_by_date(fires_df, months, dt)
    
    if plotting: 
        fires_df = format_df(fires_df)
    return fires_df

def plot_county_dt(st_name, st_abbrev, st_fips, county_name, fires_dt=None, perims_dt=None, 
                  meters_nearby=0): 
    """Plot all obs., along with the fire-perimeter boundaries, for a given county and date. 
    
    Read in the data for the inputted year and parse it to the given state/county and date. 
    Read in the fire perimeter boundaries for the given state/county and date, and parse 
    those. Plot it all. 
    
    Args: 
        st_name: str
            State name to grab for plotting. 
        st_abbrev: str
            State abbrevation used for the bash script. 
        st_fips: str
            State fips used for the base script. 
        county_name: str
            County name to grab for plotting. 
        dt: str
            Date to grab for plotting. 
        meters_nearby: int
            Holds the number of meters to go out to look for fires. Used to 
            figure out what data to read in. 
    """
    
    year = fires_dt.split('-')[0]
    perims_dt = fires_dt if not perims_dt else perims_dt
    county_names = [county_name] if not isinstance(county_name, list) else county_name
    fires_data = read_n_parse(year, state_names=st_name, county_names=county_name, dt=fires_dt, plotting=True,
                             meters_nearby=meters_nearby)
    county_map = USMapBuilder('data/county_shapefiles_2014/cb_2014.shp', geo_level='County', 
                state_names=[st_name], county_names=county_names, figsize=(40, 20), border_padding=0.1)
    fire_boundaries = grab_fire_perimeters(perims_dt, st_name, st_abbrev, st_fips)
    try: 
        county_map.plot_boundary(fire_boundaries)
    except Exception as e:
        print e
    county_map.plot_points(fires_data, 4)
    plt.show()


# Back to the original for a second. 
for dt in xrange(1, 9): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', 'Trinity', dt)


# Let's look at 100m out. 
for dt in xrange(1, 9): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', 'Trinity', dt, meters_nearby=100)


# Let's look at 250m out. 
for dt in xrange(1, 9): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', 'Trinity', dt, meters_nearby=250)


# Let's look at 500m out. 
for dt in xrange(1, 9): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', 'Trinity', dt, meters_nearby=500)


for year in xrange(2012, 2016): 
    print 'Year: {}'.format(year)
    print '-' * 50
    for meters_nearby in (0, 100, 250, 500): 
        print 'Meters Nearby: {}'.format(meters_nearby)
        print '=' * 50
        df = read_df(year, meters_nearby)
        df.info()
        print '=' * 50


def check_dists(year, continous_vars, categorical_vars): 
    """Plot the distributions of varaibles for the inputted year. 
    
    Read in the data for the inputted year. Then, take the inputted 
    variable names in the continous_vars and categorical_vars parameters, 
    and plot their distributions. Do this separately for observations 
    labeled as forest-fires and those labeled as non forest-fires. 
    
    Args: 
        year: int
            Holds the year of data to use for plotting. 
        continous_vars: tuple (or other iterable) of strings
            Holds the names of the continuous variables to use for plotting. 
        categorical_vars: tuple (or other iterable) of strings. 
            Holds the names of the categorical variables to use for plotting. 
    """
    dfs = []
    for meters_ahead in (0, 100, 250, 500): 
        dfs.append(read_df(year, meters_ahead))
    
    fires = []
    non_fires = []
    for df in dfs: 
        fires.append(df.query('fire_bool == 1'))
        non_fires.append(df.query('fire_bool == 0'))
        
    print 'Continuous Vars'
    print '-' * 50
    for var in continous_vars: 
        print 'Variable: {} : Fires, then non-fires'.format(var)
        print '-' * 50
        for idx, df in enumerate(dfs): 
            f, axes = plt.subplots(1, 8, figsize=(20, 5))
            fire_var = fires[idx][var]
            non_fire_var = non_fires[idx][var]
            print fire_var.mean(), non_fire_var.mean()
            plot_var_dist(fire_var, categorical=False, ax=axes[0:4], show=False)
            plot_var_dist(non_fire_var, categorical=False, ax=axes[4:], show=False)
            plt.show()
    print 'Categorical Vars'
    print '-' * 50
    for var in categorical_vars: 
        print 'Variable: {} : Fires, then non-fires'.format(var)
        print '-' * 50
        for idx, df in enumerate(dfs):
            f, axes = plt.subplots(1, 2)
            plot_var_dist(fires[idx][var], categorical=True, ax=axes[0], show=False)
            plot_var_dist(non_fires[idx][var], categorical=True, ax=axes[1], show=False)
            plt.show()


check_dists(2012, continous_vars, categorical_vars)


check_dists(2013, continous_vars, categorical_vars)


check_dists(2014, continous_vars, categorical_vars)


check_dists(2015, continous_vars, categorical_vars)


# In the end, none of these distributions look that different either (which I actually find a little surprising). However, I think that the distributions of how many nearby fires there are will change quite a bit now. I'm going to move forward with calling fires within 3 days of the actual date (forward), and being in a boundary or within 500m of one. From the maps, it's clear that 250m doesn't quite cut it, and at that point we still miss out on obs. that clearly look like forest-fires. 
# 
# As a last note, there will be more exploration into this in another notebook, once the dataset with forest-fires labeled as above is created. 
# 




import pickle
import os
import pandas as pd


os.chdir('/Users/sallamander/galvanize/forest-fires/data/pickled_data/MODIS/')


with open('df_2001.pkl') as f: 
    df_2001 = pickle.load(f)


df_2001.sort(['year', 'month', 'day', 'LAT', 'LONG'], inplace=True)


lat_long_df = pd.DataFrame(df_2001.groupby(['LAT', 'LONG']).count()['AREA']).reset_index().rename(columns={'AREA': 'COUNT'})


# The lat_long_df now holds a count of the number of times a lat, long pair ends up in our dataframe. I need to figure out what this means - is it a fire that occurs over several days, are there just mistakenly fires that are in the dataframe multiple times, or what?

# Check shape before merging to make sure it remains the same. 
df_2001.shape


lat_long_2001_df = df_2001.merge(lat_long_df, on=['LAT', 'LONG'])


lat_long_2001_df.shape


lat_long_2001_df.head(5)


lat_long_2001_df.query('COUNT >=4')


# From the above, it looks like it might be hard to follow fires across time, at least if we assume that the fires are 
# occuring at the same latitude/longitude throughout their lifespan(which is probably an incredibly simplistic and 
# incorrect assumption). Let's try seeing if we restrict fires to +/- one degree what happens. I'll focus on the lat/long coordinates from the fires above, since these combos of lat/long coordinates show up 4 times a year. 
# 

lat_long_df['LAT'] = lat_long_df['LAT'].astype(float)
lat_long_df['LONG'] = lat_long_df['LONG'].astype(float)


type(lat_long_df['LAT'][0])


lat_long_2001_df.query('LAT > 42.18 & LAT < 43.18 & LONG < -111.094 & LONG > -112.094').sort(['year', 'month', 'day'])


lat_long_2001_df.query('LAT > 42.679 & LAT < 42.681 & LONG < -111.593 & LONG > -111.595').sort(['year', 'month', 'day'])


# So it looks like the basis of this project will be some kind of grouping algorithm. If we go out a half degree in the longitude/latitude direction, it looks like we have some groupings of fires. Let's checkout 2015 though and see what that looks like. 
# 

os.chdir('/Users/sallamander/galvanize/forest-fires/data/pickled_data/MODIS/')


with open('df_2015.pkl') as f: 
    df_2015 = pickle.load(f)


lat_long_df = pd.DataFrame(df_2015.groupby(['LAT', 'LONG']).count()['AREA']).reset_index().rename(columns={'AREA': 'COUNT'})


lat_long_2015_df = df_2015.merge(lat_long_df, on=['LAT', 'LONG'])


lat_long_2015_df['COUNT'].max()


lat_long_2015_df.query('COUNT >=10').sort(['LAT', 'LONG', 'year', 'month', 'day'])


lat_long_2015_df.query('COUNT >=3 & COUNT <= 5').sort(['LAT', 'LONG', 'year', 'month', 'day'])


lat_long_2015_df['AREA'].describe()





# In another notebook (`map_exploration_of_ground_truth`), I noted that some of the merging for individual geometrical points didn't occur to be correct when merging/comparing points with boundaries that were ahead in time. They would line up and appear to work correctly with the boundaries that occured on the same day as the point, but appeared to not line up correctly with boundaries that occured a couple days in the future. 
# 
# In this notebook, I'm going to attempt to do the merging with `shapely`/`fiona`, instead of doing it with the `PostGIS` extension in `Postgres`. There were some posts online suggesting that `PostGIS` mergings can be a little finickey, so I'm seeing here if a different methodology gives the same results. If so, I think it's just a visualization thing, where the viz. alters things a little bit (it is lat/long projects, after all).  
# 

import fiona 
from shapely.geometry import Point, asShape
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import datetime
from itertools import izip
from dsfuncs.geo_plotting import USMapBuilder
get_ipython().magic('matplotlib inline')


# First, I'll pull a bunch of functions from another notebook that I can use to visualize the `PostGIS` merging results. 
# 

def read_df(year, modis=True): 
    """This function will read in a year of data, and add a month column. 
    
    Args: 
        year: str
        modis: bool
            Whether to use the modis or viirs data for plotting. 
        
    Return:
        Pandas DataFrame
    """
    if modis: 
        output_df = pd.read_csv('../../../data/csvs/day3_500m/detected_fires_MODIS_' + str(year) + '.csv', 
                                parse_dates=['date'], true_values=['t'], false_values=['f'])
    else: 
         output_df = pd.read_csv('../../../data/csvs/day3_500m/detected_fires_VIIRS_' + str(year) + '.csv', 
                                parse_dates=['date'], true_values=['t'], false_values=['f'])
    output_df['month'] = output_df.date.apply(lambda dt: dt.strftime('%B'))
    output_df.dropna(subset=['region_name'], inplace=True) # These will be obs. in Canada. 
    return output_df
    
def grab_by_location(df, state_names, county_names=None): 
    """Grab the data for a specified inputted state and county. 
    
    Args: 
        df: Pandas DataFrame
        state: set (or iterable of strings)
            State to grab for plotting.
        county: set (or iterable of strings) (optional)
            County names to grab for plotting. If None, simply grab the 
            entire state to plot. 
            
    Return: 
        Pandas DataFrame
    """
    if county_names: 
        output_df = df.query('state_name in @state_names and county_name in @county_names')
    else: 
        output_df = df.query('state_name in @state_names')
    return output_df

def grab_by_date(df, months=None, dt=None): 
    """Grab the data for a set of specified months.
    
    Args: 
        df: Pandas DataFrame
        months: set (or iterable of strings)
    
    Return: 
        Pandas DataFrame
    """
    if months is not None: 
        output_df = df.query("month in @months")
    else: 
        split_dt = dt.split('-')
        year, month, dt = int(split_dt[0]), int(split_dt[1]), int(split_dt[2])
        match_dt = datetime.datetime(year, month, dt, 0, 0, 0)
        output_df = df.query('date == @match_dt')
    return output_df

def format_df(df): 
    """Format the data to plot it on maps. 
    
    This function will grab the latitude and longitude 
    columns of the DataFrame, and return those, along 
    with a third column that will be newly generated. This 
    new column will hold what color we want to use to plot 
    the lat/long coordinate - I'll use red for fire and 
    green for non-fire. 
    
    Args: 
        df: Pandas DataFrame
    
    Return: 
        numpy.ndarray
    """
    
    keep_cols = ['long', 'lat', 'fire_bool']
    intermediate_df = df[keep_cols]
    output_df = parse_fire_bool(intermediate_df)
    output_array = output_df.values
    return output_array

def parse_fire_bool(df): 
    """Parse the fire boolean to a color for plotting. 
    
    Args: 
        df: Pandas DataFrame
        
    Return: 
        Pandas DataFrame
    """
    
    # Plot actual fires red and non-fires green. 
    output_df = df.drop('fire_bool', axis=1)
    output_df['plotting_mark'] = df['fire_bool'].apply(lambda f_bool: 'ro' if f_bool == True else 'go')
    return output_df

def read_n_parse(year, state_names, county_names=None, months=None, plotting=False, dt=None): 
    """Read and parse the data for plotting.
    
    Args: 
        year: str
        state_names: set (or other iterable) of state names
            State names to grab for plotting. 
        county_names: set (or other iterable) of county names (optional)
            County names to grab for plotting. 
        months: months (or other iterable) of months (optional)
            Month names to grab for plotting. 
        plotting: bool 
            Whether or not to format the data for plotting. 
            
    Return: 
        Pandas DataFrame
    """
    
    fires_df = read_df(year)
    if state_names: 
        fires_df = grab_by_location(fires_df, state_names, county_names)
    
    if months or dt: 
        fires_df = grab_by_date(fires_df, months, dt)
    
    if plotting: 
        fires_df = format_df(fires_df)
    return fires_df

def plot_states(year, state_names, months=None, plotting=True): 
    """Plot a state map and the given fires data points for that state. 
    
    Args: 
        year: str
        state_names: set (or other iterable) of state names
            State names to grab for plotting. 
        months: set (or other iterable) of month names 
            Month names to grab for plotting. 
        plotting: bool 
            Whether or not to format the data for plotting. 
    
    Return: Plotted Basemap
    """
    ax = plt.subplot(1, 2, 1)
    state_map = USMapBuilder('data/state_shapefiles_2014/cb_2014_us_state_500k2.shp', geo_level='State', 
                        state_names=state_names, ax=ax, border_padding=1)
    fires_data = read_n_parse(year, state_names, months=months, plotting=plotting)
    fires_data_trues = fires_data[fires_data[:,2] == 'ro']
    fires_data_falses = fires_data[fires_data[:,2] == 'go']
    print fires_data_trues.shape, fires_data_falses.shape
    state_map.plot_points(fires_data_trues)
    ax = plt.subplot(1, 2, 2)
    state_map = USMapBuilder('data/state_shapefiles_2014/cb_2014_us_state_500k2.shp', geo_level='State',  
                        state_names=state_names, ax=ax, border_padding=1)
    state_map.plot_points(fires_data_falses)
    plt.show()
    
def grab_fire_perimeters(dt, st_name, st_abbrev, st_fips, county=False): 
    """Grab the fire perimter boundaries for a given state and year.
    
    Args: 
        dt: str
        st_name: str
            State name of boundaries to grab. 
        st_abbrev: str
            State abbreviation used for bash script. 
        st_fips: str
            State fips used for bash script. 
            
    Return: Shapefile features. 
    """
            
    # Run bash script to pull the right boundaries from Postgres. 
    subprocess.call("./grab_perim_boundary.sh {} {} {}".format(st_abbrev, st_fips, dt), shell=True)
    
    filepath = 'data/{}/{}_{}_2D'.format(st_abbrev, st_abbrev, dt)
    return filepath

def plot_st_fires_boundaries(dt, st_name, st_abbrev, st_fips): 
    """Plot the fire boundaries for a year and given state.
    
    Args
    ----
        dt: str
            Contains date of boundaries to grab. 
        state_name: str
            Holds state to plot. 
        st_abbrev: str
            Holds the states two-letter abbreviation. 
        st_fips: str
            Holds the state's fips number. 
        
    Return: Plotted Basemap
    """
    
    boundaries_filepath = grab_fire_perimeters(dt, st_name, st_abbrev, st_fips)
    st_map = USMapBuilder('data/state_shapefiles_2014/cb_2014_us_state_500k2.shp', geo_level='State', 
                          state_names=[st_name], border_padding=1)
    st_map.plot_boundary(boundaries_filepath)
    plt.show()
    
def plot_counties_fires_boundaries(year, state_name, st_abbrev, st_fips, county_name, 
                                   months=None, plotting=True, markersize=4): 
    """Plot a county map in a given state, including any fire perimeter boundaries and potentially
    detected fires in those counties. 
    
    Args: 
        year: str
        state_name: str
            State name to grab for plotting. 
        st_abbrev: str
            Holds the states two-letter abbrevation
        st_fips: str
            Holds the state's fips number. 
        county_name: str or iterable of strings 
            County names to grab for plotting. 
        months: set (or other iterable) of strings (optional)
            Month names to grab for plotting
        plotting: bool
            Whether or not to format the data for plotting. 
    """
    
    county_name = county_name if isinstance(county_name, list) else [county_name]
    county_map = USMapBuilder('data/county_shapefiles_2014/cb_2014.shp', geo_level='County', 
                state_names=[state_name], county_names=county_name, figsize=(40, 20), border_padding=0.1)
    boundaries_filepath = grab_fire_perimeters(year, state_name, st_abbrev, st_fips)
    county_map.plot_boundary(boundaries_filepath)
    if plotting: 
        fires_data = read_n_parse(year, state_name, months=months, plotting=plotting)
        county_map.plot_points(fires_data, markersize)
    plt.show()

def plot_county_dt(st_name, st_abbrev, st_fips, county_name, fires_dt=None, perims_dt=None, 
                  markersize=2): 
    """Plot all obs., along with the fire-perimeter boundaries, for a given county and date. 
    
    Read in the data for the inputted year and parse it to the given state/county and date. 
    Read in the fire perimeter boundaries for the given state/county and date, and parse 
    those. Plot it all. 
    
    Args: 
        st_name: str
            State name to grab for plotting. 
        st_abbrev: str
            State abbrevation used for the bash script. 
        st_fips: str
            State fips used for the base script. 
        county_name: str
            County name to grab for plotting. 
        dt: str
            Date to grab for plotting. 
        markersize (optional): int
            Used to control the size of marker to use for plotting fire points. 
    """
    
    year = fires_dt.split('-')[0]
    perims_dt = fires_dt if not perims_dt else perims_dt
    county_names = [county_name] if not isinstance(county_name, list) else county_name
    fires_data = read_n_parse(year, state_names=st_name, county_names=county_name, dt=fires_dt, plotting=True)
    print fires_data.shape
    print county_name, st_name
    county_map = USMapBuilder('data/county_shapefiles_2014/cb_2014.shp', geo_level='County', 
                state_names=[st_name], county_names=county_names, figsize=(40, 20), border_padding=0.1)
    fire_boundaries = grab_fire_perimeters(perims_dt, st_name, st_abbrev, st_fips)
    try: 
        county_map.plot_boundary(fire_boundaries)
    except Exception as e:
        print e
    print len(fires_data)
    county_map.plot_points(fires_data, markersize=6)
    plt.show()


# Now, I'll build some new functions to work on conducting the merging using `fiona`/`shapely`. 
# 

def plot_shapely(state_name, st_abbrev, st_fips, cnty_name, fires_dt, 
                perims_dt): 
    """Plot observations as fires/non-fires from shapely calculations. 
    
    Args: 
    ----
        state_name: str
        st_abbrev: str
        cnty_name: list of strs or str
        fires_dt: str
            Date to grab fire observations from. 
        perims_dt: str
            Date to grab fire perimeters from. 
    """
    cnty_name = [cnty_name] if not isinstance(cnty_name, list) else cnty_name

    date_parts = fires_dt.split('-')
    fires_data = read_n_parse(date_parts[0], state_names=state_name, county_names=cnty_name, dt=fires_dt, plotting=False)
    lat_longs = fires_data[['lat', 'long']]
    lat_longs.reset_index(drop=True, inplace=True)
    lat_longs['fire_bool'] = False
    
    points = []
    for lat, lng in izip(lat_longs['lat'], lat_longs['long']): 
        points.append(Point(lng, lat))
    
    filepath = grab_fire_perimeters(perims_dt, state_name, st_abbrev, st_fips)
    perimeters = fiona_collection = fiona.open(filepath + '.shp')
    for idx1, perim in enumerate(perimeters): 
        lat_longs[idx1] = False
        shape = asShape(perim['geometry'])
        for idx2, point in enumerate(points):
            lat_longs.ix[idx2, idx1] = shape.distance(point) < 0.05
        lat_longs['fire_bool'] = (lat_longs['fire_bool'] | 
                                 lat_longs[idx1])
    county_map = USMapBuilder('data/county_shapefiles_2014/cb_2014.shp', geo_level='County', 
                state_names=[state_name], county_names=cnty_name, figsize=(40, 20), border_padding=0.1)
    county_map.plot_boundary(filepath)
    fires_df = format_df(lat_longs)
    county_map.plot_points(fires_df, markersize=6)
    
    plt.show()


plot_shapely('Colorado', 'co', '08', 'Mesa', fires_dt='2012-06-28', 
                    perims_dt='2012-06-29')


plot_county_dt(st_name='Colorado', st_abbrev='co', st_fips='08', 
               county_name='Mesa', fires_dt='2012-06-28', 
               perims_dt='2012-06-29')


plot_shapely('Colorado', 'co', '08', ['Hinsdale', 'Mineral'], fires_dt='2013-07-03', 
                    perims_dt='2013-07-04')


plot_county_dt(st_name='Colorado', st_abbrev='co', st_fips='08', 
               county_name=['Hinsdale', 'Mineral'], fires_dt='2013-07-03', 
               perims_dt='2013-07-04')


plot_shapely('California', 'ca', '06', 'Mono', fires_dt='2015-08-17', 
                    perims_dt='2015-08-18')


plot_county_dt(st_name='California', st_abbrev='ca', st_fips='06', 
               county_name='Mono', fires_dt='2015-08-17', 
               perims_dt='2015-08-18')


plot_shapely('Montana', 'mt', '30', 'Pondera', fires_dt='2015-08-27', 
                    perims_dt='2015-08-29')


plot_county_dt('Montana', 'mt', '30', 'Pondera', fires_dt='2015-08-27', 
              perims_dt='2015-08-29')


plot_shapely('Montana', 'mt', '30', 'Pondera', fires_dt='2015-08-28', 
                    perims_dt='2015-08-29')


plot_county_dt('Montana', 'mt', '30', 'Pondera', fires_dt='2015-08-28', 
              perims_dt='2015-08-29')


# It's clear from the above merging that the postgreSQL is not quite merging correctly, and the shapely/fiona stuff is doing what I want. The next step is to code up the ground truth labeling process so that it actually uses this shapely/fiona to label the ground truth for all observations. 
# 




# This notebook is aimed at looking at some of the features that were engineered to run through the model, namely the columns containin the number of nearby observations and fires across time and space. Specifically, columns were created that hold the counts of observations and fires (denoted by the column `fire_bool == True`) that are within 0.1 km of a given observation, and within 1-7 days, as well as within 365, 730, 1095 days. These columns all have the base name `all_nearby_count` and `all_nearby_fires`, and then add on the number of days back in time that were used to look for nearby observations. For example, `all_nearby_count1` holds the number of observations that were within 0.1 km of a given observation, up to 1 day prior, whereas `all_nearby_fires2` holds the number of positively labeled observations (e.g. `fire_bool == True`) that were within 0.1 im of a given observation, up to 2 days prior. 
# 
# This notebook will be used to examine the distributions of these engineered columns. 
# 

from dsfuncs.processing import remove_outliers
from dsfuncs.dist_plotting import plot_var_dist
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# I'll only be looking at certain columns, so let's only read in the ones that we'll actually be looking at. 
keep_cols = ['fire_bool']
for days_back in (0, 1, 2, 3, 4, 5, 6, 7, 365, 730, 1095): 
    keep_cols.append('all_nearby_count' + str(days_back))
    keep_cols.append('all_nearby_fires' + str(days_back))


engineered_df = pd.read_csv('../../modeling/model_input/geo_time_done.csv', usecols=keep_cols)
engineered_df.columns


keep_cols.remove('fire_bool') # We don't want this in there when we cycle through each of the columns to plot.
non_fires = engineered_df.query('fire_bool == 0')
fires = engineered_df.query('fire_bool == 1')


for col in keep_cols: 
    print 'Variable: {} : Non-fires, then fires'.format(col)
    print '-' * 50
    f, axes = plt.subplots(1, 8, figsize=(20, 5))
    plot_var_dist(non_fires[col], categorical=False, ax=axes[0:4], show=False)
    plot_var_dist(fires[col], categorical=False, ax=axes[4:], show=False)
    plt.show()


for col in keep_cols: 
    print 'Variable: {} : Non-fires, then fires'.format(col)
    print '-' * 50
    print 
    print non_fires[col].describe()
    print 
    print fires[col].describe()
    print '\n' * 3


# Given the above, it looks like the majority of these columns could be helpful in identifying fires. The observations that are fires clearly have different distributions than observations that are not labeled as fires, which was to be expected. The observations that are fires have a larger number of nearby observations, regardless of whether or not we restrict those nearby observations to be fires or not. 
# 
# The one column that won't be helpful (and it was designed this way, so I really shouldn't have created it), is the `all_nearby_fires0`. So, I'll have to throw that out when modeling, and the rest I'll have to test out to see if they add predictive power. Obviously, they are all highly correlated, and as a result it might be the case that a handful of them can capture the most of the predictive power that they collectively offer. 
# 




# It turns out that in some of the perimeter boundary file for fires, there are duplicate entries if I group by fire name and date. I need to figure out what is going with those - if I merge on the detected fires to the perimeter boundary files by date and gemoetry, some detected fires end up merging to two different boundaries, which creates what appear to be duplicate entries. Before dropping what appear to be duplicate perimeter boundaries for the same fire, I need to figure out why there are two different rows in the data for these fires. Is it because two different sources reported the same fire boundary, or what?

import psycopg2
import numpy as np


conn = psycopg2.connect(dbname='forest_fires')
c = conn.cursor()


# Grab the fire names for 2013 that have the highest number of obs per fire_name and date.
c.execute('''SELECT COUNT(fire_name) as total, fire_name, date_
            FROM daily_fire_shapefiles_2013 
            GROUP BY fire_name, date_
            ORDER BY total DESC
            LIMIT 20; ''')
c.fetchall()


# Now let's look at a a couple of these and see whats different. SELECT * 
# won't work below because there is a field that returns all blanks and 
# causes an error. 
columns = ['acres', 'agency', 'time_', 'comments', 'year_', 'active', 
          'unit_id', 'fire_num', 'fire', 'load_date', 'inciweb_id', 
          'st_area_sh', 'st_length_', 'st_area__1', 'st_length1', 
          'st_area__2', 'st_lengt_1']
for column in columns: 
    c.execute('''SELECT ''' + column + '''
                FROM daily_fire_shapefiles_2013 
                WHERE fire_name = 'Douglas Complex' and date_ = '2013-8-4'; ''')
    print column + ':', np.unique(c.fetchall())


# Upon first glance what it looks like is that there are multiple entries per fire name because there are different parts of the fire. If we look at 'fire' variable above we see a number of different names - ['Brimstone' 'Dads Creek' 'Farmers' 'Malone' 'Malone Creek' 'McNab' 'Milo'
#  'Rabbit Mountain' 'Tom East']. With some googling we can tell that these are different parts/areas of the same fire. Let's look at 2014 and check one ob. there to (a). Just check it out, and (b.) Kind of confirm that what I think is true is true in another year. 
# 

# Grab the fire names for 2014 that have the highest number of obs per fire_name and date.
c.execute('''SELECT COUNT(fire_name) as total, fire_name, date_
            FROM daily_fire_shapefiles_2014 
            GROUP BY fire_name, date_
            ORDER BY total DESC
            LIMIT 20; ''')
c.fetchall()


# Now let's look at a a couple of these and see whats different. SELECT * 
# won't work below because there is a field that returns all blanks and 
# causes an error. The columns also aren't the same in 2014 as they are in 2013. 
columns = ['acres', 'agency', 'time_', 'comments', 'year_', 'active', 
          'unit_id', 'fire_num', 'fire', 'load_date', 'inciweb_id', 
          'st_area_sh', 'st_length_']
for column in columns: 
    c.execute('''SELECT ''' + column + '''
                FROM daily_fire_shapefiles_2014 
                WHERE fire_name = 'July Complex' and date_ = '2014-8-27'; ''')
    print column + ':', np.unique(c.fetchall())


# Cool - the 2014 data seems to tell the same story. 
# 




import pickle
import numpy as np
import pandas as pd
from pandas.tseries.offsets import *


# There are three things that I want to check out here. The previous eda files have more or less been playing around 
# with less clear purposes. 
# 
# 1.) How does this data look - after contacting the forest services, I've been told that each row should be a fire
# detected, and there should be four pictures for every long/lat. pair, corresponding to 4 pictures in that given day. 
# I need to verify that, and if it is true see how far back that works. 
# 
# 2.) How does the confidence level range across fires in general, and then how about across the 4 pictures per day? Is there a good range of fire confidence level from 10-90%?
# 
# 3.) How am I going to group these fires from day to day - do I go out 0.01 degrees in each lat/long direction, 0.10 
# degrees, etc.? To look into this what I'm going to do is just cycle through a bunch of different values and see how many fires pop up within that many degrees. 
# 

def row_examination(df):
    counts = df.groupby(['LAT', 'LONG', 'year', 'month', 'day']).count()
    print 'Max. number of rows per lat/long coordinate: ', counts.max()[0]
    print 'Min. number of rows per lat/long coordinate: ', counts.min()[0]
    print 'Mean number of rows per lat/long coordinate: ', counts.mean()[0]


def conf_levels_examination(df): 
    print 'Confidence level info: ', df['CONF'].describe()


for year in xrange(2015, 2002, -1): 
    with open('../../../data/pickled_data/MODIS/df_' + str(year) + '.pkl') as f: 
        df = pickle.load(f)
        print 'Year: ', str(year)
        print '-' * 50, '\n'
        row_examination(df)
        conf_levels_examination(df)


# From the above, I can see that starting in 2009 it looks like the 4 pictures per lat/long coordinates might be true, but before that it doesn't look to be true. So now I need to just pull out some rows and see exactly what is going on. 
# 

def examine_index(df, index): 
    print df.query('LAT == @index[0] & LONG == @index[1] & year == @index[2] & month == @index[3] & day == @index[4]')


def examine_lown_rows(df, count_num, output = False): 
    fires_counts = df.groupby(['LAT', 'LONG', 'year', 'month', 'day']).count()['AREA'] == count_num
    if output: 
        for index in fires_counts[fires_counts == True].index[0:10]:
            print '-' * 50
            examine_index(df, index)
    else:  
        return fires_counts.index[0:10]


for year in xrange(2015, 2014, -1): 
    with open('../../../data/pickled_data/MODIS/df_' + str(year) + '.pkl') as f: 
        df = pickle.load(f)
        examine_lown_rows(df, 2, True)     


# My hunch is that for those obs. where there are only 1 or 2 obs. for a given lat/long/date combination, it's because 
# the fire moved quickly and so the lat/long coordinates changed quickly. The way to check this would be to see if there are larger number of fires (rows) within a given lat/long distance from that current long/lat distance. 
# 

for year in xrange(2015, 2014, -1): 
    with open('../../../data/pickled_data/MODIS/df_' + str(year) + '.pkl') as f: 
        df = pickle.load(f)
        for row_number in [2, 4, 10]: 
            print 'Row Number', str(row_number)
            print '-' * 50
            rows_less = examine_lown_rows(df, row_number, False)
            df['LAT'] = df['LAT'].astype(float)
            df['LONG'] = df['LONG'].astype(float)
            for dist_out in [0.001, 0.01, 0.05, 0.1]:
                print 'dist', str(dist_out)
                print '-' * 50
                for index in rows_less: 
                    lat_1, lat_2 = float(index[0]) - dist_out, float(index[0]) + dist_out
                    long_1, long_2 = float(index[1]) - dist_out, float(index[1]) + dist_out
                    result = df.query('LAT > @lat_1 & LAT < @lat_2 & LONG > @long_1 & LONG < @long_2')
                    print result.shape[0], index[0], index[1]


# The above doesn't seem to support my hypothesis (but I did only look at 10 obs). Maybe it's just that those fires aren't actually present at those locations later in the day, and so there aren't pictures for them. Ultimately, the way to represent a square kilometer is by +/- 0.01 degrees, so I need to stick with that. What I need to do now is to find some large fire in a given year and try to track that, and see how likely it is that I can track a particular fire (i.e. are there 4 pics. a day for that fire, are there 4 pics. a day for multiple days for that fire, etc.). What does this data look like for a location where I know there was a large fire that burned for a very long period of time.
# 

# If I go to the following website (https://www.nifc.gov/fireInfo/fireInfo_stats_lgFires.html), I can pick some large fires that are from 2009+ (since earlier I decided to only look past this data). I'll start off by looking at some 2012 fires, and figure out if I go out a certain distance from that fire's origin, how many rows there are in the data (i.e. how many detected fires that far out from the large fires origin). I'll look at the Long Draw fire (42.392, -117.894), the Holloway fire (41.973, -118.366), the Mustang Complex (45.425, -114.59), Rush (40.621   , -120.152), and Ash Creek MT (45.669, -106.469). We'll check these out first.  
# 

fires = [('Long Draw', 42.392, -117.894), ('Holloway', 41.973, -118.366), ('Mustang Complex', 45.425, -114.59), 
        ('Rush', 40.621, -120.152), ('Ash Creek MT', 45.669, -106.469)]


with open('../../../data/pickled_data/MODIS/df_' + str(2012) + '.pkl') as f: 
    df = pickle.load(f)
    df['LAT'] = df['LAT'].astype(float)
    df['LONG'] = df['LONG'].astype(float)
    for fire in fires: 
        print fire[0]
        print '-' * 50
        lat_orig, long_orig = fire[1], fire[2]
        for dist_out in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]: 
            lat_1, lat_2 = lat_orig - dist_out, lat_orig + dist_out 
            long_1, long_2 = long_orig - dist_out, long_orig + dist_out
            result = df.query('LAT > @lat_1 & LAT < @lat_2 & LONG > @long_1 & LONG < @long_2')
            print 'Dist_out: %s' %str(dist_out), result.shape[0], '\n'


# This looks fairly legit, but what I want to do is take 100 unique random LAT/LONG pairs from the 2012 database, and 
# look at the average number of obs. we get at dist_out of the above values (0.001, 0.01, 0.1, 0.2, 0.3, 0.4, and 0.5). If the number of observations that far out is on average the same as above, that would suggest that we weren't actually able to locate these large fires. 
# 

with open('../../../data/pickled_data/MODIS/df_' + str(2012) + '.pkl') as f: 
    df = pickle.load(f)
    df['LAT'] = df['LAT'].astype(float)
    df['LONG'] = df['LONG'].astype(float)
    df = df.set_index(['LAT', 'LONG'])
    indices = df.index
    unique_indices = np.unique(indices)
    num_indices = len(unique_indices)
    obs_array = []
    rand_indices = np.random.randint(low=0, high=num_indices, size=100)
    for index in rand_indices: 
        lat_orig, long_orig = unique_indices[index]
        num_obs = []
        for dist_out in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]: 
            lat_1, lat_2 = lat_orig - dist_out, lat_orig + dist_out 
            long_1, long_2 = long_orig - dist_out, long_orig + dist_out
            result = df.query('LAT > @lat_1 & LAT < @lat_2 & LONG > @long_1 & LONG < @long_2')
            num_obs.append(result.shape[0])
        obs_array.append(num_obs)


np.array(obs_array).mean(axis = 0)


# The above seems to suggest that our results from attempting to look at all the 2012 fires are fairly successful, at least if we look out far enough. Up to 0.1 degrees, the numbers we were seeing for the fires seem to be roughly the same on average, whereas if we go 0.3 degrees +, we end up with a much larger number of obs. for most of the fires. The last thing I want to check is if we resrict by date. If we take the lat/longs for those fires we looked at above, how many obs for a given dist_out are within a week, two, and a month of when that fire started (later by that time period, since I'm using the start dates)?  For Long Draw, we have a start date of 07/09, for Holloway 08/08, for Mustang Complex 08/17, for Rush 8/12, and for Ash Creek MT 6/27. 
# 

# Let's just add a datetime instance to this for our purposes, and then we can reuse the code. If this weren't eda and 
# I wasn't in an ipython notebook, I would have written this all into a function. 
fires = [('Long Draw', 42.392, -117.894, '2012-07-09'), ('Holloway', 41.973, -118.366, '2012-08-08'), 
         ('Mustang Complex', 45.425, -114.59, '2012-08-17'), ('Rush', 40.621, -120.152, '2012-08-12'), 
         ('Ash Creek MT', 45.669, -106.469, '2012-06-27')]


with open('../../../data/pickled_data/MODIS/df_' + str(2012) + '.pkl') as f: 
    df = pickle.load(f)
    year, month, day = df['year'], df['month'], df['day']
    df['datetime'] = pd.Series([pd.to_datetime(str(year) + '-' + str(month) + '-' + str(day)) for year, month, day in zip(year, month, day)])
    df['datetime'] = pd.Series([datetime.date() for datetime in df['datetime']])
    df['LAT'] = df['LAT'].astype(float)
    df['LONG'] = df['LONG'].astype(float)
    for fire in fires: 
        print fire[0]
        print '-' * 50
        lat_orig, long_orig = fire[1], fire[2]
        date = fire[3]
        for dist_out in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]: 
            for date_out in xrange(0, 6): 
                date_beg, date_end = pd.to_datetime(date).date(), (pd.to_datetime(date) + DateOffset(weeks=date_out)).date()
                lat_1, lat_2 = lat_orig - dist_out, lat_orig + dist_out 
                long_1, long_2 = long_orig - dist_out, long_orig + dist_out
                result = df.query('LAT > @lat_1 & LAT < @lat_2 & LONG > @long_1 & LONG < @long_2') 
                result2 = result[(result['datetime'] >= date_beg) & (result['datetime'] <= date_end)] 
                if date_out == 0: 
                    print 'Dist_out: %s' %str(dist_out), 'Weeks Out: %s' %str(date_out), result.shape[0], '\n', date_end
                else: 
                    print 'Dist_out: %s' %str(dist_out), 'Weeks Out: %s' %str(date_out), result2.shape[0], '\n', date_end
            print '-' * 50


# The above seems to indicate fairly well that the majority of fires around that given location where there were large fires happen within 5 weeks of the fire. Honestly, I still expected to see a larger percentage. So I'm going to go a week back and see if that gives me a larger percentage. Maybe I just had the wrong start dates for some of these. 
# 

with open('../../../data/pickled_data/MODIS/df_' + str(2012) + '.pkl') as f: 
    df = pickle.load(f)
    year, month, day = df['year'], df['month'], df['day']
    df['datetime'] = pd.Series([pd.to_datetime(str(year) + '-' + str(month) + '-' + str(day)) for year, month, day in zip(year, month, day)])
    df['datetime'] = pd.Series([datetime.date() for datetime in df['datetime']])
    df['LAT'] = df['LAT'].astype(float)
    df['LONG'] = df['LONG'].astype(float)
    for fire in fires: 
        print fire[0]
        print '-' * 50
        lat_orig, long_orig = fire[1], fire[2]
        date = fire[3]
        for dist_out in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]: 
            for date_out in xrange(0, 6): 
                date_beg, date_end = (pd.to_datetime(date)- DateOffset(weeks=1)).date(), (pd.to_datetime(date) + DateOffset(weeks=date_out)).date()
                lat_1, lat_2 = lat_orig - dist_out, lat_orig + dist_out 
                long_1, long_2 = long_orig - dist_out, long_orig + dist_out
                result = df.query('LAT > @lat_1 & LAT < @lat_2 & LONG > @long_1 & LONG < @long_2') 
                result2 = result[(result['datetime'] >= date_beg) & (result['datetime'] <= date_end)] 
                if date_out == 0: 
                    print 'Dist_out: %s' %str(dist_out), 'Weeks Out: %s' %str(date_out), result.shape[0], '\n', date_end
                else: 
                    print 'Dist_out: %s' %str(dist_out), 'Weeks Out: %s' %str(date_out), result2.shape[0], '\n', date_end
            print '-' * 50








# When I merge the detected fire centroids into the fire perimeter boundary shapefiles, I end up with more observations than I started with (even doing a left join on the centroids), which suggests that there are some fire centroids that are matching on to multiple fire perimeters. I need to figure out why so that I can then decide what to do about it. 
# 

import psycopg2
import numpy as np


# First we need to merge on the perimeter boundary files, and then figure out which detected
# fire centroids have merged on in mutiple places. In the detected fire data, a unique 
# index is (lat, long, date, gmt, src). We'll start with 2013. 
conn = psycopg2.connect('dbname=forest_fires')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE merged_2013 AS
             (SELECT points.*, polys.fire_name, polys.fire, polys.agency, polys.unit_id
             FROM detected_fires_2013 as points
                    LEFT JOIN daily_fire_shapefiles_2013 as polys
             ON points.date = polys.date_ 
                AND ST_WITHIN(points.wkb_geometry, polys.wkb_geometry));
            ''')
conn.commit()


# Just to display what I'm talking about: 
cursor.execute('''SELECT COUNT(*)
                FROM detected_fires_2013;''')
print 'Detected_fires_2013 obs: ', cursor.fetchall()[0][0]

cursor.execute('''SELECT COUNT(*)
                FROM merged_2013;''')
print 'Merged_2013 obs: ', cursor.fetchall()[0][0]


# Let's check if any obs. now have more than one row per (lat, long, date, gmt, and src), which
# prior to this merge was unique. 
cursor.execute('''SELECT COUNT(*) as totals
                FROM merged_2013
                GROUP BY lat, long, date, gmt, src
                ORDER BY totals DESC
                LIMIT 10;''')
cursor.fetchall()


# Okay, cool this is exactly what I thought. Let's get the (lat, long, date, gmt, src) of some 
# of these obs and checkout what is going on. 
# 

cursor.execute('''WITH totals_table AS 
                (SELECT COUNT(*) as totals, lat, long, date, gmt, src
                FROM merged_2013
                GROUP BY lat, long, date, gmt, src)
                
                SELECT lat, long, date, gmt, src 
                FROM totals_table 
                WHERE totals > 1;''')
duplicates_list = cursor.fetchall()


# Let's just go down the above list and figure out what is going on. 
duplicates_info = []
for duplicate in duplicates_list[:20]: 
    lat_coord, long_coord, date1, gmt, src = duplicate
    
    cursor.execute('''SELECT fire_name, fire, unit_id, agency
                    FROM merged_2013
                    WHERE lat = {} and long = {}
                        and gmt = {} and date = '{}'
                        and src = '{}'; '''.format(lat_coord, long_coord, gmt, date1, src))
    duplicates_info.append([cursor.fetchall(), duplicate])


for duplicate in duplicates_info[:10]: 
    print '-' * 50 
    print duplicate[0]
    print '\n' * 2
    print duplicate[1]


# From the above, it doesn't look like I'll be able to tell too much, with the exception of how many perimeter boundaries that given (lat, long, date, gmt, src) merged onto. I think to solve this definitively, I need to graph some of the above boundaries and see how/if they overlap (I assume they must overlap, or else I wouldn't be having this issue). 
# 




# The plan with this notebook is to explore and take a look at the distributions of the original variables that I'll be working with in this project. I'll plan on looking at the majority of the variables both with and without outliers.
# 

from dsfuncs.processing import remove_outliers
from dsfuncs.dist_plotting import plot_var_dist, plot_binary_response
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


def read_df(year): 
    """Read in a year's worth of data. 
    
    Args: 
        year: int
            Holds the year of data to read in. 
    """
    
    df = pd.read_csv('../../../data/csvs/detected_fires_MODIS_' + str(year) + '.csv', true_values =['t', 'True'], false_values=['f', 'False'])
    df.dropna(subset=['region_name'], inplace=True) # These will be obs. in Canada. 
    return df


# Test out my function and see what columns I actually want to look at the distributions of. 
fires_df_2012 = read_df(2012)
fires_df_2012.columns


fires_df_2012.info()


# I'm going to look at the following set of continous and categorical variables. 
continous_vars = ('lat', 'long', 'gmt', 'temp', 'spix', 'tpix', 'conf', 'frp', 'county_aland', 'county_awater')
categorical_vars = ('urban_areas_bool', 'src', 'sat_src')


# Test out the outliers function to make sure it runs. 
print fires_df_2012['lat'].values.shape
print remove_outliers(fires_df_2012['lat']).shape


# Testing out the plot_var_dist function for a categorical variable. 
plot_var_dist(fires_df_2012['urban_areas_bool'], categorical=True)


# Testing out the plot_var_dist function for a continuous variable. 
f, axes = plt.subplots(1, 4)
plot_var_dist(fires_df_2012['lat'], categorical=False, ax=axes[0:])


def check_dists(year, continous_vars, categorical_vars): 
    """Plot the distributions of varaibles for the inputted year. 
    
    Read in the data for the inputted year. Then, take the inputted 
    variable names in the continous_vars and categorical_vars parameters, 
    and plot their distributions. Do this separately for observations 
    labeled as forest-fires and those labeled as non forest-fires. 
    
    Args: 
        year: int
            Holds the year of data to use for plotting. 
        continous_vars: tuple (or other iterable) of strings
            Holds the names of the continuous variables to use for plotting. 
        categorical_vars: tuple (or other iterable) of strings. 
            Holds the names of the categorical variables to use for plotting. 
    """
    
    df = read_df(year)
    fires = df.query('fire_bool == 0')
    non_fires = df.query('fire_bool == 1')
    print 'Continuous Vars'
    print '-' * 50
    for var in continous_vars: 
        print 'Variable: {} : Non-fires, then fires'.format(var)
        f, axes = plt.subplots(1, 8, figsize=(20, 5))
        plot_var_dist(fires[var], categorical=False, ax=axes[0:4], show=False)
        plot_var_dist(non_fires[var], categorical=False, ax=axes[4:], show=False)
        plt.show()
    print 'Categorical Vars'
    print '-' * 50
    for var in categorical_vars: 
        print 'Variable: {} : Non-fires, then fires'.format(var)
        f, axes = plt.subplots(1, 2)
        plot_var_dist(fires[var], categorical=True, ax=axes[0], show=False)
        plot_var_dist(non_fires[var], categorical=True, ax=axes[1], show=False)
        plt.show()


check_dists(2012, continous_vars, categorical_vars)


check_dists(2013, continous_vars, categorical_vars)


check_dists(2014, continous_vars, categorical_vars)


check_dists(2015, continous_vars, categorical_vars)


# #### Notes on distributions 
# 
# There are a couple of things worth nothing about these distributions above (These notes don't change too much if we change what we're calling outliers using 3 standard deviations instead of 2 - and I realize this use of outliers may not be the best. But, it just helps to get a sense of how the distributions look originally and after throwing out "outliers"). 
# 
# The latitude and longitude appear to be pretty informative - those observations that are actually fires are on average in locations we would expect more fires. What's interesting is that the distributions of latitude and longitude differ for fires as much as they do when we look at with and without "outliers". 
# 
# The gmt distributions are fairly interesting. If we look at those from non-fire observations, we see that the distribution is centered around times later in the day, around 1800. For fires, the times are pretty spread out, occuring at all times of the day. It seems logical, and at least somewhat supported by the data, that these non-fires that get labeled as fires are simply really hot surfaces, which peak in afternoon times. 
# 
# The average temp of fire observations is higher than non-fires, which isn't too surprising. The same holds true of the conf. variable, which also isn't surprising. The frp variable seems like it might on average be higher for fires than non-fires, which makes sense. It's a little hard to tell from the distributions above, though, how different frp is across fires and non-fires. The tpix and spix variables don't show any real differences between the fire and non-fire observations. The county area of land and county area of water only seem to show that larger counties on average have more fires. This is pretty intuitive. 
# 
# The urban areas bool suggests what we might expect - fires occur in non-urban areas only. The src variable shows something that looks interesting; those observations from the gsfc_drl make up a decent proportion of the non-fire observations, but almost none of the fire observations. It looks like having that as a dummy (and maybe one or two of the other observations) could potentially be helpful. The sat_src variable doesn't look like it'll be too helpful. 
# 

def add_land_water_ratio(df):
    """Add a land_water_ratio column to the inputted DataFrame. 
    
    Add a new variable to the inputted DataFrame that represents 
    the ratio of a counties land area to its total area (land plus water). 
    
    Args: 
        df: Pandas DataFrame
    
    Return: Pandas DataFrame
    """
    
    df.eval('land_water_ratio = county_aland / (county_aland + county_awater)')
    return df

def plot_land_water_ratio(year): 
    """Plot the land_water_ratio distribution for the inputted year. 
    
    For the inputted year, read in the data, add a land_water_ratio 
    variable to the DataFrame, and then plot it's distribution. 
    
    Args: 
        year: int
            Holds the year of data to plot. 
    """
    
    
    df = read_df(year)
    df = add_land_water_ratio(df)
    var = 'land_water_ratio'
    
    fires = df.query('fire_bool == 0')[var]
    non_fires = df.query('fire_bool == 1')[var]
    
    print 'Fires mean: {}'.format(fires.mean())
    print 'Non Fires mean: {}'.format(non_fires.mean())
    
    f, axes = plt.subplots(1, 8, figsize=(20, 5))
    plot_var_dist(fires, categorical=False, ax=axes[0:4], show=False)
    plot_var_dist(non_fires, categorical=False, ax=axes[4:], show=False)
    plt.show()


for year in xrange(2012, 2016): 
    print 'Year: {}'.format(str(year))
    print '-' * 50
    plot_land_water_ratio(2012)


# #### Notes on land_water_ratio
# 
# The distributions for the ratio of land to total area (land plus water) are fairly interesting. I would have expected that the fires were in areas with a higher ratio of land to total area (i.e. less water), but the plots above don't seem to support this idea. They seem to suggest that this variable probably won't be too helpful. 
# 

for year in xrange(2012, 2016): 
    df = read_df(year)
    print 'Year {}'.format(str(year))
    print '-' * 50
    for category in categorical_vars: 
        print category
        plot_binary_response(df, category, 'fire_bool')


# #### Notes on the distribution of the response across categoricals 
# 
# I thought it would be useful to see what the percentage of fires was in each category (for those categorical variables), along with what percentage of the data set that category made up (which is shown in the text above the bars). I think there are two takeaways here: 
# 
# 1. Almost none of the observations are from urban areas, **but** for those that are, almost none are fires. 
# 2. A small proportion of the observations have the `gsfc_drl` as the `src` variable. For those that do, though, a disproportionately small number are actually fires (relative to the other `src` categories). 
# 

