#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from ast import literal_eval
from matplotlib import rc
rc('text', usetex=True)
import scipy.stats as sts
import seaborn as sns; sns.set()
import networkx as nx
import numpy as np
from humanfriendly import format_timespan
import random as rn
import pandas as pd
import geopandas
import mplleaflet
import json

# https://movement.uber.com/explore/london/

# https://medium.com/uber-movement/examining-the-impact-of-the-london-tower-bridge-closure-5b7626e44915


# # Processing data
# ## Greater London

# In[2]:


# Load the data of the city's region

with open('london_lsoa.json') as coors: coors = json.load(coors) 
data = geopandas.read_file('london_lsoa.json')


# In[3]:


data['centroid'] = [[data.centroid[i].xy[0][0], data.centroid[i].xy[1][0]] for i in range(len(data))]


# In[4]:


data.plot(column='la_code', figsize=(17, 7), );


# In[5]:


'''
Check for adjecency based on coordinates to create edges between region.
'''

edges = []
for i in range(len(coors['features'])):
    if i % 100 == 0: print('processing {}%'.format(i/10))
    for j in range(len(coors['features'])):
        if i != j:
            coor1 = np.array(coors['features'][i]['geometry']['coordinates'][0][0])
            coor2 = np.array(coors['features'][j]['geometry']['coordinates'][0][0])
            shared_coor = len(np.array([x for x in set(tuple(x)                                           for x in coor1) & set(tuple(x) for x in coor2)]))
            
            if shared_coor > 0 and (i, j) not in edges and (j, i) not in edges:
                edges.append((i, j))


# In[6]:


len(edges) #{>0}#2611 #{>1}2231 #{>2}#1998 {>5}#1542 {>8}#1826 {>10}#1132


# In[7]:


G =nx.Graph()
G.add_nodes_from(map(int, data.MOVEMENT_ID))
G.add_edges_from(edges)

print(nx.info(G))


# In[8]:


plt.figure(figsize=(25, 15))
nx.draw(G, pos = data['centroid'], edgelist=edges,
        node_size=80, node_color = 'blue', with_labels=False)
plt.show()


# In[9]:


nx.is_connected(G)


# ## Central London

# In[433]:


selected = np.array([841, 212, 210, 363, 365, 922, 981, 982, 362, 364,
                    980, 359, 573, 21, 572, 755, 106, 567, 322, 940,
                    935, 930, 157, 768, 939, 331, 330, 231, 861, 230,
                    854, 898, 899, 900, 901, 903, 285, 907, 287, 232,
                    867, 906, 909, 288, 233, 358, 905, 501, 492, 902,
                    869, 868, 517, 524, 551, 933, 936, 155, 156, 646,
                    921, 926, 897, 289, 873, 925, 904, 908, 753, 754, 931])

data_small = data[data.MOVEMENT_ID.isin(map(str, selected))]

edges_small = [i for i in edges if i[0] in               map(int, list(data_small.MOVEMENT_ID))               and i[1] in map(int, data_small.MOVEMENT_ID)]


# In[434]:


G_small = nx.Graph()
G_small.add_nodes_from(map(int, data_small.MOVEMENT_ID))
G_small.add_edges_from(edges_small)

print(nx.info(G_small))


# In[435]:


data_small.plot(column='MOVEMENT_ID', edgecolor='k', figsize=(15, 10))
mplleaflet.display()


# ### The bridges

# In[436]:


'''
Assigning the edges to the bridges they represent
'''
bridges = {
    'tower_bridge': [(322, 899)],
    'london_bridge': [(21, 899), (21, 898)],
    'southwark_bridge': [(21, 898)],
    'blackfriars_bridge': [(21, 898)],
    'waterloo_bridge': [(362, 854)],
    'westminster_bridge': [(364, 854)],
    'lambeth_bridge': [(364, 854),(364, 230),(980, 230)],
    'vauxhaul_bridge': [(980, 861),(982, 231),(982, 861),(980, 231)]}

sum(list(bridges.values()), [])


# In[437]:


'''
- The final would be much more interesting if we would take advantage
of the open source dataset of Uber Movement. The idea is that the repartition
of cities is already established which can serve as a delimiter for regions.

- The average of travelling time between the any two regions will serve as a way 
to associate weights with each of the edges of the graph.

- Finding a realistic geolocalizator of hospitals and ambulance bases in the UK,
then building a simulation to assess how reliable the positioning is and ways we 
could improve it.

- The change on infrastructure will base a new challenge for the project as the 
simulator would estimate the effect of the closure of the London Tower Bridge

- The distribution of the waiting time [exponential]

- Using Kepler for the visualization.
'''
print('')


# ### Crime rate data

# In[438]:


crime_rate = {
    'Camden':{'codes': [106, 492, 501, 517, 524, 551, 567, 572, 573], 'crime_rate': 141.2},
    'City of London':{'codes': [21], 'crime_rate': 86.4},
    'Hackney':{'codes': [155, 156, 157, 646], 'crime_rate': 111.2},
    'Islington':{'codes': [753, 754, 755, 768], 'crime_rate': 121.2},
    'Kensington and Chelsea':{'codes': [210, 212, 841], 'crime_rate': 139.2},
    'Lambeth':{'codes': [230, 231, 232, 233, 854, 861, 867, 868, 869], 'crime_rate': 105.0},
    'Lewisham':{'codes': [873], 'crime_rate': 84.1},
    'Southwark':{'codes': [285, 287, 288, 289, 897, 898, 899, 900, 
                           901, 902, 903, 904, 905, 906, 907, 908, 909], 'crime_rate': 111.1},
    'Tower Hamlets':{'codes': [322, 921, 922, 925, 926, 930, 931, 933, 935, 936, 939, 940], 'crime_rate': 106.5},
    'Wandsworth':{'codes': [330, 331], 'crime_rate': 80.1},
    'Westminster':{'codes': [358, 359, 362, 363, 364, 365, 980, 981, 982], 'crime_rate': 150.2}}

for node in G_small.nodes:
    G_small.nodes[node]['area_km2'] = data_small.area_km2[node]
    G_small.nodes[node]['radius'] = (data_small.area_km2[node] / np.pi)**(1/2)
    for region in crime_rate:
        if node in crime_rate[region]['codes']:
            G_small.nodes[node]['crime_rate'] = crime_rate[region]['crime_rate']


# ### Hopspital data

# In[439]:


import googlemaps
gmaps = googlemaps.Client(key='AIzaSyBTz-a4viHh6g9xYwWNvfMWxFBmOPk2dm4')


# In[440]:


hospital = gmaps.places('Hospital in London')
hospital = pd.DataFrame.from_dict(hospital['results'])
hospital['location'] = [list(hospital.geometry[row]['location'].values())[::-1] for row in range(len(hospital))]
hospital = hospital[hospital.columns[hospital.columns.isin(['name','location','rating','user_ratings_total'])]]

hospital


# In[441]:


hospital_names=["St Thomas' Hospital", 'The Royal London Hospital', 
                'London Bridge Hospital', "King Edward VII's Hospital",
                'University College Hospital', 
                'Royal London Hospital for Integrated Medicine']

hospital = hospital[hospital.name.isin(hospital_names)]


# In[442]:


'''
hospital.to_csv('london_hospitals.csv', index=False)
hospital = pd.read_csv('london_hospitals.csv')
hospital.location  = [literal_eval(hospital.location[i]) for i in range(len(hospital))]
'''
hospital


# In[443]:


G_hospital = nx.Graph()
G_hospital.add_nodes_from(map(int, hospital.index))

for node in G_hospital.nodes:
    G_hospital.nodes[node]['rating'] = hospital['rating'][node]

print(nx.info(G_hospital))


# In[444]:


# Include hospitals in their respective areas

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
hospital_nodes = []
for i in range(len(hospital)):
    point = Point(hospital.iloc[i]['location'])
    for j in G_small.nodes:
        polygon = data_small.geometry[j]
        if polygon.contains(point):
            hospital_nodes.append([j, hospital.iloc[i]['name']])
            G_small.nodes[j]['hospitals'] = {'name': hospital.iloc[i]['name'],
                                             'rating': hospital.iloc[i]['rating']}
hospital_nodes


# ### Travel time data

# In[445]:


def extract_trvl_time(graph, file):
    trvl_time = pd.read_csv(file)
    trvl_time['segment'] = list(zip(trvl_time.sourceid, trvl_time.dstid))
    trvl_time_small = trvl_time[trvl_time.segment.isin(edges_small)]
    trvl_time_small = trvl_time_small.groupby('segment').mean().reset_index()
    
    for idx,edge in enumerate(G_small.edges):
        graph.edges[edge]['m_travel_time'] = trvl_time_small.mean_travel_time[idx]
        graph.edges[edge]['sd_travel_time'] = trvl_time_small.standard_deviation_travel_time[idx]
    trvl_time_small.head()


# In[446]:


'''
We use two datasets:
- The third quarter of 2016 before the closure of the London Tower bridge:
#london-lsoa-2016-3-All-MonthlyAggregate.csv

- The fouth quarter of 2016 during the closure of the London Tower bridge:
#london-lsoa-2016-4-All-MonthlyAggregate.csv
'''

extract_trvl_time(G_small, 'london-lsoa-2016-3-All-MonthlyAggregate.csv')


# ### The final network

# In[447]:


plt.figure(figsize=(15, 7))


nx.draw(G_hospital, pos = hospital['location'], edgelist = [],
        node_size = 200, cmap='Greens', with_labels=False, vmax=10, vmin=-2,
        node_color = [G_hospital.nodes[i]['rating'] for i in G_hospital.nodes])

nx.draw(G_small, pos = data_small['centroid'], edgelist = edges_small, 
        # Size of the node represent the area of the region
        node_size = 500*np.array([G_small.nodes[i]['area_km2'] for i in G_small.nodes]), 
        with_labels=True, edge_cmap=plt.cm.binary,
        # The color of the edge represent the average travelling time
        edge_color=[G_small.edges[i, j]['m_travel_time'] for i, j in G_small.edges],
        # The color of the node represents the crime rate in the region
        node_color=[G_small.nodes[i]['crime_rate'] for i in G_small.nodes], vmin=60, vmax=160,
        cmap='coolwarm')

plt.show()


# # Simulation

# In[372]:


class simulation:
    def __init__(self):
        # The list of nodes that include hospitals
        self.hospital_nodes = [854, 931, 899, 358, 567, 573]
        
        
    def initialize(self, graph, n_requests, v_ambulance):
        
        self.graph = graph # Graph of the city
        self.v_ambulance = v_ambulance # measured in Km/h
        self.n_requests = n_requests # The number of requests
        
        self.visits = [] # Record the hospital from which ambulance dispatched
        self.requests = [] # Record the regions that generated ambulance calls
        
    def generate_requests(self):
        '''
        The function generates emergency calls in different regions of the city
        using a random sampler weighted by the normalized criminality rate.
        (i.e., a region with relatively high criminality rate would have more
        frequent ambulance calls)
        '''
        # Put all crime rates of regions into a list
        crime_rates = list(nx.get_node_attributes(self.graph,'crime_rate').values())
        # Normalize all the crime rate to serve as a probability distribution
        normalized_crime_rates = [float(i)/sum(crime_rates) for i in crime_rates]
        # Create n_requests for ambulance in different regions of the city
        for i in range(self.n_requests):
            self.requests.append(np.random.choice(list(self.graph.nodes), p=normalized_crime_rates))
            
    def estimated_travel_time(self, edge_list):
        '''
        The function estimates the average travel time thru the shortest path between 
        the hospital and the region from which the emergency call was generated.
        
        - Methodology:
        For each edge on the shortest path, a sample is taken from a normal distribution with
        mean = average travel time of the edge
        sd = standard deviation of travel time of the edge
        Note: both estimations of mean and sd are extracted from Uber Movement dataset
        
        - Returns: the sum of edge samples.
        '''
        trv_cost = 0 # Counts the travel time on the shortest path
        # Turn the nodes of the shortest path into a list of edges
        edge_list = list(map(tuple, zip(edge_list, edge_list[1:])))
        for i in edge_list:
            # Add the edge samples based on the parameters (means and standard deviations)
            trv_cost+=np.random.normal(self.graph.edges[i]['m_travel_time'], self.graph.edges[i]['sd_travel_time'])
        # Return the total travel time thru the shortest path
        return trv_cost
        
    def shortest_path(self, source):
        '''
        The function computes the shortest path between hospital nodes and the node
        from which the emergency call was generated.
        
        Break-down:
        - Edge: Using Dijkstra method, the mean travel time of edges are used as weights 
          to identify the fastest paths rather than the shorted paths to the emergency node.
          The function evaluates all the hospitals to determine the best option.
          
        - Dispatching: based on the radius of the region from which the ambulace is dispatched
          the estimated time of dispatching is calculated based on the ambulace average speed.
          mean = time for the ambulance to travel the radius of the region
          standard deviation = 5 seconds
        
        - Pick_up: Also based on the radius of the region from which the emergency call was generated
          mean = time for the ambulance to travel the radius of the region
          standard deviation = 10 seconds
          Note: sd is higher because the ambulance is less certain about the emergency call source
          the ambulace average speed is in Km\h hence the conversion to seconds
        '''
        # Identify the hospital node that is fastest to dispatch the ambulance from
        edge = nx.multi_source_dijkstra(self.graph, sources=self.hospital_nodes, target=source, weight='m_travel_time')
        
        # Estimate the time for the ambulance to be dispatched
        dispatching = np.random.normal((self.graph.nodes[edge[1][-1]]['radius']/(self.v_ambulance))*3600, 5)
        
        # Estimate the time it takes for the ambulance to pick up the patient
        pick_up = np.random.normal((self.graph.nodes[source]['radius']/(self.v_ambulance))*3600, 10)
        
        # Record the hospital from which the ambulance was dispateched
        self.visits.append(edge[1][0])
        
        # Estimate the travel time on the shortest path
        trip = self.estimated_travel_time(edge[1])
        
        # Return the sum of the whole trip
        return dispatching + trip + pick_up
            
    def run(self):
        '''
        The function runs the simulation for the designated number of requests
        by summing up the shortest path timer for each ambulance dispatching.
        '''
        emergency_response_timer = 0
        # Generate the requests of emergency service
        self.generate_requests()
        for idx,i in enumerate(self.requests):
            # Add up the shortest path time for the ambulance
            response_time = self.shortest_path(i)
            self.requests[idx] = [i, response_time]
            emergency_response_timer += response_time
        # Returns the average responce time for all requests
        return emergency_response_timer / self.n_requests
    
    def hospital_responces(self):
        '''
        The function returns the count of emergency responses for each hospital
        '''
        record = {x:[self.graph.nodes[x]['hospitals']['name'], 
                     self.visits.count(x)] for x in self.visits}
        return pd.DataFrame.from_dict(record, 
               orient='index', columns=['Name', 'Count']).sort_values(by=['Count'], ascending=False)
    
    def region_requests(self):
        '''
        The function returns the count of emergency calls for each region
        as well as the overall response time and average response time.
        '''
        # Create a data frame based on the randomly generated requests
        df = pd.DataFrame(self.requests, columns=['Region', 'Average response time']).sort_values(by=['Region'])
        # Count the occurance of emergencies in each region
        rt = list(df.groupby('Region').count()['Average response time'])
        df = df.groupby('Region').mean(); df['Count'] = rt
        return df


# In[373]:


# EXAMPLE RUN FOR TESTING
sim_example = simulation()
sim_example.initialize(G_small, 1000, 20)
sim_example.run()


# In[374]:


sim_example.hospital_responces()


# In[375]:


sim_example.region_requests().head()


# ### Response time distribution

# In[376]:


'''
Simulating the distribution of the response time of ambulances
- Number of simulations = 1000
- Average ambulance speed = 20 Km/h
- Number of requests = 1000
'''
history = []
for i in range(500):
    if (i+1)%125==0: print('Progress {}%'.format((i+1)/5))
    sim = simulation()
    sim.initialize(G_small, 2000, 20)
    history.append(sim.run())


# In[377]:


print('Mean of response time: ', format_timespan(np.mean(history)),
      '\n\nConfidence interval:\n', 
      format_timespan(np.percentile(history, 5)), ' | ',
      format_timespan(np.percentile(history, 95)))

plt.figure(figsize=(10, 5))
plt.hist(history, density=True, bins=20)
plt.axvline(np.mean(history), color='red', Linewidth=.5)
plt.title('Distribution of ambulance response-time', fontsize=22)
plt.xlabel('Response time (seconds)', fontsize=15)
plt.show()


# In[378]:


sim.hospital_responces()


# In[379]:


sim.region_requests().head()


# ### The effect of average speed on response-time

# In[380]:


'''
Simulating the distribution of the response time of ambulances
- Number of simulations = 100
- Number of requests = 2000
'''
m_history_v = []
speeds = [10, 15, 20, 25, 30, 35]
for j in speeds:
    history_v = []
    print('Speed {} Km/H'.format(j))
    for i in range(100):
        sim_v = simulation()
        sim_v.initialize(G_small, 2000, j)
        history_v.append(sim_v.run())
    m_history_v.append([np.mean(history_v), np.percentile(history_v, 95)-np.percentile(history_v, 5)])


# In[381]:


meas = np.array(m_history_v)[:,0]
err = np.array(m_history_v)[:,1]
plt.figure(figsize=(12, 6))
plt.errorbar(speeds, meas, yerr=err, capsize=3, ecolor='red', color='black')

plt.xlabel('Ambulance average speed (Km/h)', fontsize=15)
plt.ylabel('Average response time (s)', fontsize=15)
plt.title('Ambulance average speed vs response time', fontsize=22)
plt.show()


# ### The effect of London Tower Bridge closure on response-time
# In late 2016, London's Tower Bridge was closed for three months to undergo critical maintenance. The closure started from 1st of October to 30th December 2016.

# #### Approach 1: Remove the edges that represnt the closed bridge

# In[382]:


G_small_bridge = G_small.copy()
G_small_bridge.remove_edges_from(bridges['tower_bridge'])
print(nx.info(G_small_bridge))


# In[383]:


'''
Simulating the distribution of the response time of ambulances
- Number of simulations = 1000
- Average ambulance speed = 20 Km/h
- Number of requests = 1000
'''
history_brdg1 = []
for i in range(500):
    if (i+1)%125==0: print('Progress {}%'.format((i+1)/5))
    sim_brdg1 = simulation()
    sim_brdg1.initialize(G_small_bridge, 2000, 20)
    history_brdg1.append(sim_brdg1.run())


# In[384]:


print('Mean of response time: ', format_timespan(np.mean(history_brdg1)),
      '\n\nConfidence interval:\n', 
      format_timespan(np.percentile(history_brdg1, 5)), ' | ',
      format_timespan(np.percentile(history_brdg1, 95)))

plt.figure(figsize=(10, 5))
plt.hist(history_brdg1, density=True, bins=20)
plt.axvline(np.mean(history_brdg1), color='red', Linewidth=.5)
plt.title('Distribution of ambulance response-time', fontsize=22)
plt.xlabel('Response time (seconds)', fontsize=15)
plt.show()


# #### Approach 2: Work with the updated Uber Movement data

# In[385]:


extract_trvl_time(G_small, 'london-lsoa-2016-4-All-MonthlyAggregate.csv')


# In[386]:


'''
Simulating the distribution of the response time of ambulances
- Number of simulations = 1000
- Average ambulance speed = 20 Km/h
- Number of requests = 1000
'''
history_brdg2 = []
for i in range(500):
    if (i+1)%125==0: print('Progress {}%'.format((i+1)/5))
    sim_brdg2 = simulation()
    sim_brdg2.initialize(G_small, 2000, 20)
    history_brdg2.append(sim_brdg2.run())


# In[450]:


print('BEFORE CLOSING THE LONDON TOWER BRIDGE: \nMean of response time: \n',
      format_timespan(np.mean(history)),
      '\nConfidence interval:\n', 
      format_timespan(np.percentile(history, 5)), ' | ',
      format_timespan(np.percentile(history, 95)))

print('\n\nAFTER CLOSING THE LONDON TOWER BRIDGE: \nMean of response time: \n',
      format_timespan(np.mean(history_brdg2)),
      '\nConfidence interval:\n', 
      format_timespan(np.percentile(history_brdg2, 5)), ' | ',
      format_timespan(np.percentile(history_brdg2, 95)))

plt.figure(figsize=(15, 6))

plt.hist(history, density=True, bins=20, alpha=.5, label='Before Closure')
plt.axvline(np.mean(history), color='red', Linewidth=.5)

plt.hist(history_brdg2, density=True, bins=20, alpha=.5, label='After Closure')
plt.axvline(np.mean(history_brdg2), color='red', Linewidth=.5)

plt.title('Distribution of ambulance response-time\nbefore an after Tower Bridge closure', fontsize=22)
plt.xlabel('Response time (seconds)', fontsize=15); plt.ylim(0, .08)
plt.legend()
plt.show()


# In[ ]:


'''
Mean of response time:  9 minutes and 43.84 seconds 

Confidence interval:
 9 minutes and 25.53 seconds  |  10 minutes and 1.67 second
 
 
 Mean of response time:  9 minutes and 44.06 seconds 

Confidence interval:
 9 minutes and 30.5 seconds  |  9 minutes and 56.82 seconds
 
'''


# ### Effect of adding an ambulance base

# In[393]:


extract_trvl_time(G_small, 'london-lsoa-2016-3-All-MonthlyAggregate.csv')
sim.region_requests().sort_values(by=['Average response time'], ascending=False).head()


# In[453]:


avrg_res_time = sim.region_requests().sort_values(by=['Region'])

plt.figure(figsize=(15, 7))

nx.draw(G_hospital, pos = hospital['location'], edgelist = [],
        node_size = 200, cmap='Blues', with_labels=False, vmax=10, vmin=-2,
        node_color = [G_hospital.nodes[i]['rating'] for i in G_hospital.nodes])

nx.draw(G_small, pos = data_small['centroid'], edgelist = edges_small, 
        # Size of the node represent the area of the region
        node_size = 500*np.array([G_small.nodes[i]['area_km2'] for i in G_small.nodes]), 
        with_labels=True, edge_cmap=plt.cm.binary,
        # The color of the edge represent the average travelling time
        edge_color=[G_small.edges[i, j]['m_travel_time'] for i, j in G_small.edges],
        # The color of the node represents the crime rate in the region
        node_color= avrg_res_time['Average response time'], cmap='coolwarm')

plt.show()


# In[404]:


'''
Simulating the distribution of the response time of ambulances
- Number of simulations = 500
- Average ambulance speed = 20 Km/h
- Number of requests = 2000
'''
history_new_base = []
for i in range(500):
    if (i+1)%125==0: print('Progress {}%'.format((i+1)/5))
    sim_new_base = simulation()
    sim_new_base.hospital_nodes.append(155)
    sim_new_base.initialize(G_small, 2000, 20)
    history_new_base.append(sim_new_base.run())


# In[452]:


print('BEFORE ADDING A NEW BASE: \nMean of response time: \n',
      format_timespan(np.mean(history)),
      '\nConfidence interval:\n', 
      format_timespan(np.percentile(history, 5)), ' | ',
      format_timespan(np.percentile(history, 95)))

print('\n\nAFTER ADDING A NEW BASE: \nMean of response time: \n',
      format_timespan(np.mean(history_new_base)),
      '\nConfidence interval:\n', 
      format_timespan(np.percentile(history_new_base, 5)), ' | ',
      format_timespan(np.percentile(history_new_base, 95)))

plt.figure(figsize=(15, 6))

plt.hist(history, density=True, bins=20, label='Before new base')
plt.axvline(np.mean(history), color='red', Linewidth=.5)

plt.hist(history_new_base, density=True, bins=20, label='After new base')
plt.axvline(np.mean(history_new_base), color='red', Linewidth=.5)

plt.title('Distribution of ambulance response-time\nbefore an after adding a base', fontsize=22)
plt.xlabel('Response time (seconds)', fontsize=15); plt.ylim(0, .08)
plt.legend()
plt.show()


# In[406]:


avrg_res_time = sim_new_base.region_requests().sort_values(by=['Region'])

plt.figure(figsize=(15, 7))

nx.draw(G_hospital, pos = hospital['location'], edgelist = [],
        node_size = 200, cmap='Greens', with_labels=False, vmax=10, vmin=-2,
        node_color = [G_hospital.nodes[i]['rating'] for i in G_hospital.nodes])

nx.draw(G_small, pos = data_small['centroid'], edgelist = edges_small, 
        # Size of the node represent the area of the region
        node_size = 500*np.array([G_small.nodes[i]['area_km2'] for i in G_small.nodes]), 
        with_labels=True, edge_cmap=plt.cm.binary,
        # The color of the edge represent the average travelling time
        edge_color=[G_small.edges[i, j]['m_travel_time'] for i, j in G_small.edges],
        # The color of the node represents the crime rate in the region
        node_color= avrg_res_time['Average response time'], cmap='coolwarm')

plt.show()

