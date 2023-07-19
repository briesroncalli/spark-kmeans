# Ben Ries-Roncalli
# February 25, 2023

from pyspark import SparkContext
#import numpy as np
import sys

# compute the squared distance between two (latitude, longitude) tuples
def dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# find the closest center from list centers to a point with coords=(latitude, longitude)
def findClosestCenter(coords, centers):
    # compute distance to each center point
    min_dist = sys.float_info.max
    min_index = 10
    
    for i in range(len(centers)):
        current_dist = dist(coords, centers[i][1])
        if current_dist < min_dist:
            min_dist = current_dist
            min_index = i
    
    # return the label of the center that is the smallest distance from the point
    return centers[min_index][0]

# compute the mean center of mass of points in cluster
def meanCenter(cluster):
    avg_latitude = 0
    avg_longitude = 0
    for coord in cluster:
        avg_latitude += coord[0]
        avg_longitude += coord[1]
    return avg_latitude/len(cluster), avg_longitude/len(cluster)

if __name__ == "__main__":
    # ensure correct usage
    if (len(sys.argv) != 3):
        print("Usage: kmeans.py <input> <output>")
        exit(-1)
        
    # parameters of k-means
    K = 5 # number of clusters
    converge_dist = 0.1 # converge distance tolerance

    # read preprocessed data file into pyspark rdd
    # file should have lines for each point with format: label,latitude,longitude
    sc = SparkContext()
    point_data = sc.textFile(sys.argv[1]) 

    # create pair rdd of points in data file (label, (latitude, longitude))
    point_rdd = point_data.map(lambda line : \
            (line.split(",")[0], (float(line.split(",")[1]), float(line.split(",")[2]))))\
            .filter(lambda row : dist(row[1], [0,0]) !=0 )\
            .persist()

    # use a random sample of K points as initial centers of the clusters
    center_pts = point_rdd.takeSample(False, K, 34)

    # do k-means iteration until the sum of the distance the centers move in one iteration is less than converge_dist
    move_dist = converge_dist + 1 
    while (move_dist > converge_dist):
        # compute the mean center of mass of each cluster
        meanCenters = point_rdd.map(lambda row : (findClosestCenter(row[1], center_pts), row[1]))\
                        .groupByKey(1)\
                        .mapValues(lambda row : meanCenter(row))

        # compute the sum total distance all the centers moved in this iteration
        move_dist = sc.parallelize(center_pts)\
                .join(meanCenters)\
                .map(lambda row : dist(row[1][1], row[1][0]))\
                .sum()
        
        # update center points to be mean center of mass of its cluster
        center_pts = meanCenters.collect()
        
        
    # write only the coordinate tuples to output file
    meanCenters.map(lambda row : row[1]).saveAsTextFile(sys.argv[2])
    sc.stop()
