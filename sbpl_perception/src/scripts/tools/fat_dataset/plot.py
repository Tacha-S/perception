import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

filepath = "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/src/perception/sbpl_perception/src/scripts/tools/fat_dataset/log_gpu"

# filepath = "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/src/perception/sbpl_perception/visualization/data_0050_000001-color/log.txt"
# filepath = "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/src/perception/sbpl_perception/visualization/data_0054_000001-color/log.txt"
# filepath = "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/src/perception/sbpl_perception/visualization/data_0059_000001-color/log.txt"

print(filepath)
state_data = {}
low_cost_ids = []
with open(filepath) as fp:
   line = fp.readline()
   cnt = 1
   while line: 
        line = line.strip()
        if "State " in line:
            m = re.findall(r"\d+ |\d+,", line)
            if len(m) > 0:
                m = line.split()
                # print(m)
                state_data[int(m[1])] = {
                    "preicp_target_cost" : int(m[3]),
                    "preicp_source_cost" : int(m[4]),
                    "target_cost" : int(m[5]),
                    "source_cost" : int(m[6]),
                    "preicp_total_cost" : int(m[8]),
                    "total_cost" : int(m[9]),
                }
            
        if "Cost of lowest" in line:
            m = re.findall(r": (\d+)", line)
            low_cost_ids.append(int(m[1]))    
        line = fp.readline()

# print(state_data)
df_overall_stats = \
        pd.DataFrame.from_dict(state_data, orient='index')
print(df_overall_stats)
print(low_cost_ids)
plt.figure()
plt.scatter(df_overall_stats['preicp_target_cost'], 
            df_overall_stats['preicp_source_cost'])
plt.scatter(state_data[low_cost_ids[0]]['preicp_target_cost'], 
            state_data[low_cost_ids[0]]['preicp_source_cost'], c="black")
plt.xticks(np.arange(0, 100, 10)) 
plt.yticks(np.arange(0, 100, 10)) 
plt.xlim(0,100)
plt.ylim(0,100)

plt.figure()
plt.scatter(df_overall_stats['target_cost'], 
            df_overall_stats['source_cost'])
plt.scatter(state_data[low_cost_ids[1]]['target_cost'], 
            state_data[low_cost_ids[1]]['source_cost'], 
            c="black")
plt.xticks(np.arange(0, 100, 10)) 
plt.yticks(np.arange(0, 100, 10)) 
plt.xlim(0,100)
plt.ylim(0,100)

colmap = {1: 'b', 2: 'g'}
kmeans = KMeans(n_clusters=2)
df_preicp = df_overall_stats.filter(['preicp_target_cost', 'preicp_source_cost'], axis=1)
kmeans.fit(df_preicp)

# Cluster using pre ICP costs
labels = kmeans.predict(df_preicp)
centroids = kmeans.cluster_centers_
centroids_cost = np.sum(centroids, axis=1)
min_cost_centroid = np.argmin(centroids_cost)

min_cost_cluster_size = len(np.argwhere(labels == min_cost_centroid).flatten())
print("Size of min cost cluster : {}".format(min_cost_cluster_size))

fig = plt.figure(figsize=(5, 5))
colors = list(map(lambda x: colmap[x+1], labels))

plt.scatter(df_preicp['preicp_target_cost'], 
            df_preicp['preicp_source_cost'], 
            color=colors, alpha=0.25)
# Check where the actual solution lies in the Pre ICP cost cluster
plt.scatter(state_data[low_cost_ids[1]]['preicp_target_cost'], 
            state_data[low_cost_ids[1]]['preicp_source_cost'], 
            c="black")

for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color="black")
plt.xlim(0, 100)
plt.ylim(0, 100)


# Check what happened to low cost cluster after ICP
min_cost_cluster_ids = np.argwhere(labels == min_cost_centroid).flatten()
min_cost_cluster = {key: state_data[key] for key in min_cost_cluster_ids}
df_min_cost_cluster = \
        pd.DataFrame.from_dict(min_cost_cluster, orient='index')

print(df_min_cost_cluster)
print("Minimum pre ICP cost of min cost cluster : {}, ID : {}, minimum post ICP cost of min cost cluster : {}, ID : {}".format(
    df_min_cost_cluster['preicp_total_cost'].min(),
    df_min_cost_cluster['preicp_total_cost'].idxmin(),
    df_min_cost_cluster['total_cost'].min(),
    df_min_cost_cluster['total_cost'].idxmin()
))
plt.show()

