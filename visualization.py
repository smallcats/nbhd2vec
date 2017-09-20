import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

import shapefile as shp
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from colour import Color
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch

import shapefile as shpf
import shapely as shpy

from shapely.geometry import Polygon as sPolygon
from shapely.geometry import Point

import pickle

U = np.load('Ucity.npy')
s = np.load('s.npy')
V = np.load('V.npy')
ndims = np.load('ndims.npy')
with open('nbhd_labels.p', 'rb') as file:
	nbhd_labels = pickle.load(file)

S = np.diag(s)

def get_nbhd_reps(city):
    city_state = {'New York':'NY', 'Philadelphia':'PA'}
    nbhd_map = shpf.Reader("ZillowNeighborhoods-{}.shp".format(city_state[city]))
    shapes = nbhd_map.shapes()
    records = nbhd_map.records()
    shape_points = [k.points for k in shapes]
    polygons = [sPolygon(k) for k in shape_points]
    out = []
    exceptout = []
    for poly, record in zip(polygons, records):
        try:
            if record[2] == city:
                out.append(((city,record[3]),(poly.representative_point().xy[0][0],poly.representative_point().xy[1][0])))
        except ValueError:
            out.append(((city, record[3]), None))
            exceptout.append(((city,record[3]),None))
            print(city, record[3])
    return dict(out), dict(exceptout)

phl_nbhd_coords, pnce = get_nbhd_reps('Philadelphia')
nyc_nbhd_coords, nnce = get_nbhd_reps('New York')

errcoords = [(-73.772019804103536, 40.60988094307978),
	(-73.820285688669443, 40.617884230338689),
	(-74.013411144335223, 40.668181787191422),
	(-73.930422742411011, 40.602532925213083),
	(-73.818876999999929, 40.865456702256424),
	(-73.958674954437299, 40.787768830896816),
	(-73.878991942331766, 40.80171357336571),
	(-73.804484955815624, 40.813496009429848),
	(-74.179093468023197, 40.58062198359071),
	(-73.968782728364289, 40.654624248925515),
	(-73.77981288839139, 40.855034848163783),
	(-73.841246241898446, 40.629670408908652),
	(-74.095030942790004, 40.627669752074247),
	(-73.796964158708818, 40.624806665346149),
	(-73.833621060386562, 40.725575678000432)]
errnbhds = [('New York', 'Far Rockaway'),
	('New York', 'Broad Channel'),
	('New York', 'Red Hook'),
	('New York', 'Marine Park'),
	('New York', 'Co-op City'),
	('New York', 'Central Park'),
	('New York', 'Hunts Point'),
	('New York', 'Throggs Neck'),
	('New York', 'Fresh Kills Park'),
	('New York', 'Prospect Park'),
	('New York', 'Pelham Bay Park'),
	('New York', 'West Jamaica Bay Islands'),
	('New York', 'Silver Lake'),
	('New York', 'East Jamaica Bay Islands'),
	('New York', 'Flushing Meadows Corona Park')]

for nbhd, coord in zip(errnbhds, errcoords):
	nnce[nbhd] = coord
#Could improve placement of nbhd labels here...
for place in nnce.keys():
    nyc_nbhd_coords[place] = nnce[place]

def get_similar(nbhd, Utrunc):
    sim_score = [cosine(Utrunc[nbhd_labels.index(nbhd),:], Utrunc[k,:]) for k in range(len(nbhd_labels))]
    return [k if k>0 else 0 for k in sim_score]

colors = list(Color('Red').range_to(Color('White'),10))

def top5(nbhd_labels, sim, city):
    comb = zip(nbhd_labels, sim)
    topall = sorted(comb, key= lambda x:x[1])
    return [k for k in topall if k[0][0] == city][:5]

def plot_map(nbhd, U, S, ndims, colors, nbhd_labels, nbhd_coords, to_city):
    city_state = {'New York':'NY', 'Philadelphia':'PA'}
    city_ll_box = {'New York':(-74.3, 40.45, -73.65, 40.95), 'Philadelphia':(-75.45,39.85,-74.85,40.15)}
    fig = plt.figure(figsize=(40,20))
    ax = fig.add_subplot(111)

    to_map = Basemap(llcrnrlon=city_ll_box[to_city][0],llcrnrlat=city_ll_box[to_city][1],
                     urcrnrlon=city_ll_box[to_city][2],urcrnrlat=city_ll_box[to_city][3],
                     resolution='i', projection='tmerc', lat_0 = 40, lon_0 = -75)
    to_map.fillcontinents(color='#ffffff',lake_color='aqua')
    to_map.readshapefile('ZillowNeighborhoods-'+city_state[to_city], 'to', drawbounds=False)

    patches = dict(zip(range(11), [[] for k in range(11)]))
    sim = get_similar(nbhd, np.dot(U,S)[:, :ndims])
    
    multiplier = 9.999/max(sim)
    sim_col = [int(np.floor(k*multiplier)) for k in sim]
    
    top5nbhd = dict(top5(nbhd_labels, sim, to_city))
    
    for info, shape in zip(to_map.to_info, to_map.to):
        city_nbhd = (info['City'], info['Name'])
        if info['City'] == to_city:
            if not city_nbhd in nbhd_labels:
                patches[10].append(Polygon(np.array(shape),True))
            else:
                idx = nbhd_labels.index(city_nbhd)
                if city_nbhd in top5nbhd.keys():
                    x,y = to_map(nbhd_coords[city_nbhd][0], nbhd_coords[city_nbhd][1])
                    plt.plot(x,y, 'ko')
                    plt.text(x,y,info['Name'],fontsize=20,ha='right',va='bottom',color='k')
                patches[sim_col[idx]].append(Polygon(np.array(shape),True))

    for k in range(10):
        ax.add_collection(PatchCollection(patches[k], facecolor=str(colors[k]), edgecolor='k', linewidths=1., zorder=2))

    ax.add_collection(PatchCollection(patches[10], facecolor='#2d2d2d', edgecolor='k', linewidths=1., zorder=2))    
    return fig

fig = plot_map(('Philadelphia', 'University City'), U_minus_city, S, ndims, colors, nbhd_labels, nyc_nbhd_coords, 'New York')
fig.savefig('University_City.png')

pickle.dump(phl_nbhd_coords, open("phl_nbhd_coords.p", "wb"))
pickle.dump(nyc_nbhd_coords, open("nyc_nbhd_coords.p", "wb"))