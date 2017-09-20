import pandas as pd
import numpy as np
import time

import shapefile
import shapely
from shapely.geometry import Polygon
from shapely.geometry import Point
from rtree import index

from pandas.io.json import json_normalize
import json

##Get the data
#Public Data (some columns removed, loaded into msgpacks):
nyc_bldgs = pd.read_msgpack('nyc_bldgs')
nyc_crashes = pd.read_msgpack('nyc_crashes')
nyc_crime = pd.read_msgpack('nyc_crime')
phl_bldgs = pd.read_msgpack('phl_bldgs')
phl_crashes = pd.read_msgpack('phl_crashes')
phl_crime = pd.read_msgpack('phl_crime')

#foursquare data (CLIENT_ID and CLIENT_SECRET omitted, get from foursquare)
city_ll_box = {'New York':(-74.3, 40.45, -73.65, 40.95), 'Philadelphia':(-75.45,39.85,-74.85,40.15)}
def interp_grid(box, div):
    h_step = (box[2] - box[0])/div
    v_step = (box[3] - box[1])/div
    xs = np.arange(box[0], box[2], h_step)
    ys = np.arange(box[1], box[3], v_step)
    return np.meshgrid(xs, ys)

nyc_grid = interp_grid(city_ll_box['New York'], 100)
phl_grid = interp_grid(city_ll_box['Philadelphia'], 100)

nyc_grid = [k for k in zip(list(nyc_grid[0].flatten()),list(nyc_grid[1].flatten()))]
phl_grid = [k for k in zip(list(phl_grid[0].flatten()),list(phl_grid[1].flatten()))]

def geog_join(grid, shapefile_name):
    shapefile_instance = shapefile.Reader(shapefile_name+".shp")
    shapes = shapefile_instance.shapes()
    records = shapefile_instance.records()
    shape_points = [k.points for k in shapes]
    polygons = [Polygon(k) for k in shape_points]
    points = [Point(k) for k in grid]

    idx = index.Index()
    for k in range(len(shapes)):
        idx.insert(k, shapes[k].bbox)
    
    def get_name(n_idx):
        if n_idx == None: return None
        else: return records[n_idx][3]    
    
    pt_nbhd = dict()
    for i in range(len(points)):
        temp = None
        for j in idx.intersection(grid[i]):
            if points[i].within(polygons[j]):
                temp=j
                break
        if temp != None: pt_nbhd[grid[i]] = get_name(temp)        
    return pt_nbhd

nyc_nbhd_grid = geog_join(nyc_grid, 'ZillowNeighborhoods-NY')
phl_nbhd_grid = geog_join(phl_grid, 'ZillowNeighborhoods-PA')

CLIENT_ID = ""
CLIENT_SECRET = ""

def get_data(grid):
    relevant = []
    count = 0
    for pt in grid.keys():
        if count%10 == 0: 
            time.sleep(5)
        url = "https://api.foursquare.com/v2/venues/search?"\
                "ll={},{}&"\
                "intent=browse&"\
                "radius=500&"\
                "limit=100&"\
                "client_id={}&"\
                "client_secret={}&"\
                "v=20170906".format(pt[1], pt[0], CLIENT_ID, CLIENT_SECRET)
        data = requests.get(url).json()['response']['venues']
        relevant.extend([(grid[pt],
                     k['name'],
                     k['location']['lat'],
                     k['location']['lng'],
                     k['categories'][0]['name']) for k in data if len(k['categories']) == 1])
        count += 1
    df = pd.DataFrame(relevant)
    df.columns = ['nbhd', 'venue', 'venue_lat', 'venue_lon', 'venue_cat']
    return(df)

phl_venues = get_data(phl_nbhd_grid)
nyc_venues = get_data(nyc_nbhd_grid)

##Clean up public data
#normalize labels
nyc_bldgs = nyc_bldgs.rename(columns = {'lon':'lng'})
nyc_crashes = nyc_crashes.rename(columns={'LATITUDE':'lat', 'LONGITUDE':'lng'})
nyc_crime = nyc_crime.rename(columns = {'Longitude':'lng', 'Latitude':'lat'})
phl_crashes = phl_crashes.rename(columns = {'LATITUDE':'lat', 'LONGITUDE':'lng'})

#dropna
nyc_bldgs = nyc_bldgs.dropna()
nyc_crashes = nyc_crashes.dropna()
nyc_crime = nyc_crime.dropna()
phl_bldgs = phl_bldgs.dropna()
phl_crashes = phl_crashes.dropna()
phl_crime = phl_crime.dropna()

#Label records with their neighborhood
def geog_join(df, shapefile_name):
    """
    df should be a pandas dataframe with columns labelled 'lat' and 'lng' for the latitude and longitude, 
    shapefile_name is the name of the shapefile containing regions without extension.
    """
    point_coords=[(df['lng'][k], df['lat'][k]) for k in df.index]
    shapefile_instance = shapefile.Reader(shapefile_name+".shp")
    shapes = shapefile_instance.shapes()
    records = shapefile_instance.records()
    shape_points = [k.points for k in shapes]
    polygons = [Polygon(k) for k in shape_points]
    points = [Point(k) for k in point_coords]

    idx = index.Index()
    for k in range(len(shapes)):
        idx.insert(k, shapes[k].bbox)

    nbhd_idx = []
    for i in range(len(points)):
        temp = None
        for j in idx.intersection(point_coords[i]):
            if points[i].within(polygons[j]):
                temp=j
                break
        nbhd_idx.append(temp)
    
    def get_name(n_idx):
        if n_idx == None: return None
        else: return records[n_idx][3]
        
    nbhds = [get_name(k) for k in nbhd_idx]
    return nbhds

nyc_bldgs['nbhd'] = geog_join(nyc_bldgs, 'ZillowNeighborhoods-NY')
nyc_crashes['nbhd'] = geog_join(nyc_crashes, 'ZillowNeighborhoods-NY')
nyc_crime['nbhd'] = geog_join(nyc_crime, 'ZillowNeighborhoods-NY')
phl_bldgs['nbhd'] = geog_join(phl_bldgs, 'ZillowNeighborhoods-PA')
phl_crashes['nbhd'] = geog_join(phl_crashes, 'ZillowNeighborhoods-PA')
phl_crime['nbhd'] = geog_join(phl_crime, 'ZillowNeighborhoods-PA')

#standardize columns
nyc_bldgs['other_building'] = nyc_bldgs['numbldgs'].apply(lambda x: x>1)
nyc_bldgs = nyc_bldgs.drop('numbldgs', axis = 1)
nyc_bldgs = nyc_bldgs.rename(columns={'numfloors':'number_stories', 'yearbuilt':'year_built'})
nyc_bldgs = nyc_bldgs.drop('income15', axis=1)
nyc_bldgs = nyc_bldgs.dropna()

phl_bldgs['other_building'] = phl_bldgs['other_building'].apply(lambda x: x == 'Y')
phl_bldgs = phl_bldgs.dropna()

phl_bldgs = phl_bldgs[phl_bldgs['year_built'] != '196Y']
phl_bldgs['year_built'] = phl_bldgs['year_built'].astype(int)

nyc_crashes = nyc_crashes.rename(columns={'NUMBER OF PERSONS INJURED':'injured', 'NUMBER OF PERSONS KILLED':'killed'})

phl_crashes = phl_crashes.rename(columns={'FATAL_COUNT':'killed', 'INJURY_COUNT':'injured'})

nyc_crime = nyc_crime.rename(columns = {'OFNS_DESC':'description'})
nyc_crime_dict = {'CRIMINAL MISCHIEF & RELATED OF':'mischief', 'HARRASSMENT 2':'misc',
       'DANGEROUS DRUGS':'drugs', 'ROBBERY':'theft', 'UNAUTHORIZED USE OF A VEHICLE':'auto_theft',
       'OFFENSES AGAINST PUBLIC ADMINI':'misc', 'FELONY ASSAULT':'assault', 'FORGERY':'fraud',
       'ASSAULT 3 & RELATED OFFENSES':'assault', 'GRAND LARCENY':'theft',
       'OFFENSES AGAINST THE PERSON':'assault', 'GAMBLING':'gambling', 'PETIT LARCENY':'theft',
       'DANGEROUS WEAPONS':'weapons', 'INTOXICATED & IMPAIRED DRIVING':'dui',
       'VEHICLE AND TRAFFIC LAWS':'misc', 'OFF. AGNST PUB ORD SENSBLTY &':'mischief',
       'POSSESSION OF STOLEN PROPERTY':'possess', 'OTHER OFFENSES RELATED TO THEF':'theft',
       "BURGLAR'S TOOLS":'burglary', 'MISCELLANEOUS PENAL LAW':'misc', 'CRIMINAL TRESPASS':'misc',
       'OFFENSES INVOLVING FRAUD':'fraud', 'BURGLARY':'burglary',
       'MURDER & NON-NEGL. MANSLAUGHTER':'murder', 'THEFT-FRAUD':'fraud',
       'GRAND LARCENY OF MOTOR VEHICLE':'auto_theft', 'OTHER STATE LAWS (NON PENAL LA':'misc',
       'ADMINISTRATIVE CODE':'misc', 'FRAUDS':'fraud', 'SEX CRIMES':'rape', 'ARSON':'arson',
       'NYS LAWS-UNCLASSIFIED FELONY':'misc', 'OFFENSES AGAINST PUBLIC SAFETY':'misc',
       'FRAUDULENT ACCOSTING':'misc', 'KIDNAPPING & RELATED OFFENSES':'misc',
       'ALCOHOLIC BEVERAGE CONTROL LAW':'alcohol',
       'OTHER STATE LAWS (NON PENAL LAW)':'misc', 'JOSTLING':'misc', 'THEFT OF SERVICES':'theft',
       'PROSTITUTION & RELATED OFFENSES':'prostitution',
       'AGRICULTURE & MRKTS LAW-UNCLASSIFIED':'misc', 'ENDAN WELFARE INCOMP':'misc',
       'DISORDERLY CONDUCT':'mischief', 'NYS LAWS-UNCLASSIFIED VIOLATION':'misc', 'ESCAPE 3':'misc',
       'HOMICIDE-NEGLIGENT,UNCLASSIFIE':'manslaughter', 'PETIT LARCENY OF MOTOR VEHICLE':'auto_theft',
       'CHILD ABANDONMENT/NON SUPPORT':'child', 'OFFENSES RELATED TO CHILDREN':'child',
       'NEW YORK CITY HEALTH CODE':'misc', 'LOITERING/GAMBLING (CARDS, DIC':'gambling',
       'OTHER STATE LAWS':'misc', 'HOMICIDE-NEGLIGENT-VEHICLE':'manslaughter',
       'INTOXICATED/IMPAIRED DRIVING':'dui', 'KIDNAPPING':'misc',
       'ANTICIPATORY OFFENSES':'misc', 'DISRUPTION OF A RELIGIOUS SERV':'mischief'}
nyc_crime['description'] = nyc_crime['description'].apply(lambda x: nyc_crime_dict[x])

phl_crime = phl_crime.rename(columns = {'text_general_code':'description'})
phl_crime_dict = {'Other Assaults':'assault', 'Thefts':'theft', 'All Other Offenses':'misc', 'Fraud':'fraud',
       'Vandalism/Criminal Mischief':'mischief', 'DRIVING UNDER THE INFLUENCE':'dui',
       'Burglary Non-Residential':'burglary', 'Narcotic / Drug Law Violations':'drugs',
       'Burglary Residential':'burglary', 'Aggravated Assault No Firearm':'assault', 'Rape':'rape',
       'Robbery No Firearm':'theft', 'Weapon Violations':'weapons', 'Robbery Firearm':'theft',
       'Theft from Vehicle':'theft', 'Other Sex Offenses (Not Commercialized)':'rape',
       'Embezzlement':'fraud', 'Recovered Stolen Motor Vehicle':'possess',
       'Motor Vehicle Theft':'auto_theft', 'Disorderly Conduct':'mischief',
       'Aggravated Assault Firearm':'assault',
       'Prostitution and Commercialized Vice':'prostitution', 'Homicide - Criminal':'murder',
       'Forgery and Counterfeiting':'fraud', 'Homicide - Criminal ':'murder',
       'Public Drunkenness':'alcohol', 'Liquor Law Violations':'alcohol', 'Arson':'arson',
       'Vagrancy/Loitering':'misc', 'Receiving Stolen Property':'possess',
       'Offenses Against Family and Children':'child', 'Gambling Violations':'gambling',
       'Homicide - Gross Negligence':'manslaughter'}
phl_crime['description'] = phl_crime['description'].apply(lambda x: phl_crime_dict[x])

#cleaning
phl_crime = phl_crime[phl_crime['date-time']<pd.to_datetime('2017-07')]

phl_crime = phl_crime.dropna()
nyc_crime = nyc_crime.dropna()

#to counts
def to_counts(phl_df, nyc_df, col_name):
    phl_df = phl_df.groupby(['nbhd', col_name]).apply(len).unstack()
    nyc_df = nyc_df.groupby(['nbhd', col_name]).apply(len).unstack()
    out = pd.concat([phl_df, nyc_df], keys = ['Philadelphia', 'New York']).fillna(0)
    return out.groupby(out.index).sum()

story_ct = to_counts(phl_bldgs, nyc_bldgs, 'number_stories')
story_ct = pd.concat([story_ct.loc[:,k:k+0.9].sum(axis = 1) for k in range(91)], axis=1)
venue_ct = to_counts(phl_venues, nyc_venues, 'venue_cat')
age_ct = to_counts(phl_bldgs, nyc_bldgs, 'year_built').drop([0, 195, 1191], axis = 1)
kill_ct = to_counts(phl_crashes, nyc_crashes, 'killed')
inj_ct = to_counts(phl_crashes, nyc_crashes, 'injured')
crime_ct = to_counts(phl_crime, nyc_crime, 'offense_class')

nbhd_ct_table = pd.concat([story_ct, age_ct, kill_ct, inj_ct, crime_ct, venue_ct], 
    keys = ['stories', 'ages', 'deaths', 'injuries', 'crimes', 'venues'], axis=1).fillna(0)

nbhd_ct_table.to_msgpack('nbhd_ct_table')
nyc_bldgs.to_msgpack('nyc_bldgs_clean')
nyc_crime.to_msgpack('nyc_crime_clean')
nyc_crashes.to_msgpack('nyc_crashes_clean')
nyc_venues.to_msgpack('nyc_venues_clean')
phl_bldgs.to_msgpack('phl_bldgs_clean')
phl_crime.to_msgpack('phl_crime_clean')
phl_crashes.to_msgpack('phl_crashes_clean')
phl_venues.to_msgpack('phl_venues_clean')