import matplotlib
matplotlib.use('Agg')

from flask import render_template, session, request, redirect, url_for
from webapp import app
import io
import base64

# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
# import psycopg2

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from colour import Color
import shapefile as shp
from mpl_toolkits.basemap import Basemap

import pickle

import numpy as np
from scipy.spatial.distance import cosine

colors = list(Color('Red').range_to(Color('White'),10))

nbhd_labels = pickle.load(open('nbhd_labels.p', 'rb'))
phl_nbhd_list = [(k[1], k[1].replace(' ', '_')) for k in nbhd_labels if k[0]=='Philadelphia']
nyc_nbhd_list = [(k[1], k[1].replace(' ', '_')) for k in nbhd_labels if k[0]=='New York']
phl_nbhd_coords = pickle.load(open('phl_nbhd_coords.p', 'rb'))
nyc_nbhd_coords = pickle.load(open('nyc_nbhd_coords.p', 'rb'))
U = np.load('Ucity.npy')
s = np.load('s.npy')
S = np.diag(s)

ndims = 92

def get_similar(nbhd, Utrunc):
    sim_score = [cosine(Utrunc[nbhd_labels.index(nbhd),:], Utrunc[k,:]) for k in range(len(nbhd_labels))]
    return [k if k>0 else 0 for k in sim_score]

def top5(nbhd_labels, sim, city):
    comb = zip(nbhd_labels, sim)
    topall = sorted(comb, key= lambda x:x[1])
    return [k for k in topall if k[0][0] == city][:5]

@app.route('/', methods = ['GET', 'POST'])
def show_map():
	# if 'phlNbhd' in session: session.pop('phlNbhd', None)
	# if 'nycNbhd' in session: session.pop('nycNbhd', None)

	if request.method == 'POST':
		return redirect(url_for('input'))

	if 'phlNbhd' in session or 'nycNbhd' in session:
		if 'phlNbhd' in session:
			curr_nbhd = session['phlNbhd'].replace('_', ' ')
			curr_city = 'Philadelphia'
			nbhd_coords = nyc_nbhd_coords
			to_city = 'New York'
			session.pop('phlNbhd', None)
		else:
			curr_nbhd = session['nycNbhd'].replace('_', ' ')
			curr_city = 'New York'
			nbhd_coords = phl_nbhd_coords
			to_city = 'Philadelphia'
			session.pop('nycNbhd', None)

		def plot_map(nbhd, U, S, ndims, colors, nbhd_labels, to_city):
		    city_state = {'New York':'NY', 'Philadelphia':'PA'}
		    city_ll_box = {'New York':(-74.3, 40.45, -73.65, 40.95), 'Philadelphia':(-75.45,39.85,-74.85,40.15)}
		    fig = plt.figure(figsize=(10,10))
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
		                    plt.text(x,y,info['Name'],fontsize=8,ha='right',va='bottom',color='k',weight='bold')
		                patches[sim_col[idx]].append(Polygon(np.array(shape),True))

		    for k in range(10):
		        ax.add_collection(PatchCollection(patches[k], facecolor=str(colors[k]), edgecolor='k', linewidths=1., zorder=2))

		    ax.add_collection(PatchCollection(patches[10], facecolor='#2d2d2d', edgecolor='k', linewidths=1., zorder=2))
		    return None

		img = io.BytesIO()

		plot_map((curr_city,curr_nbhd), U, S, ndims, colors, nbhd_labels, to_city)

		plt.savefig(img, format='png')
		img.seek(0)

		plot_url = base64.b64encode(img.getvalue()).decode()
		img_tag = '<img src="data:image/png;base64,{}">'.format(plot_url)

		# if 'phlNbhd' in session: session.pop('phlNbhd', None)
		# else: session.pop('nycNbhd', None)

		return render_template('output.html', curr_nbhd=curr_nbhd, curr_city=curr_city, plot_url=plot_url)

		# return """<center>
		# 		<head>NYC Neighborhoods Similar to {}</head>
		# 			<img src="data:image/png;base64,{}">
		# 				<form method='POST'>
		# 					<button type="submit">Try Again</button>
		# 				</form>
		# 	</center>
		# """.format(curr_nbhd, plot_url)

	else: return redirect(url_for('input'))

@app.route('/input', methods = ['GET', 'POST'])
def input():
	if request.method == 'POST':
		if request.form['submit'] == 'phlsubmit': session['phlNbhd'] = request.form['phlNbhd']
		elif request.form['submit'] == 'nycsubmit': session['nycNbhd'] = request.form['nycNbhd']
		return redirect(url_for('show_map'))
	else:
		return render_template('input.html', phl_nbhd_list=phl_nbhd_list, nyc_nbhd_list=nyc_nbhd_list)

app.secret_key = 'dfaklj39j3$%#mqd32q$femal'