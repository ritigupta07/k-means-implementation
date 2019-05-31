import math
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd 
import numpy as np
import random
from random import shuffle

class K_Means:

	def __init__(self, k =3, random_state=1):
		self.k = k
		self.tolerance = 0.0001
		self.max_iterations = 5000
		self.random_state = random_state
		random.seed(self.random_state)

	def Euclidean_distance(feat_one, feat_two):
		squared_distance = 0
		for i in range(len(feat_one)):
			squared_distance += (feat_one[i]-feat_two[i])**2
		ed = sqrt(squared_distances)
		return ed;

	def fit(self, X):	
		if(self.k == 0 or self.k > len(X)):
			raise ValueError('Invalid number of clusters.')
		if(len(X) == 0):
			raise ValueError('No Elements in data')
		#initialize the centroids, shuffle the indices and take first k
		self.centroids = {}
		y = [[i] for i in range(len(X))]
		random.shuffle(y)
		
		for i in range(self.k):
			self.centroids[i] = X[y[i]]

		for i in range(self.max_iterations):
			self.classes = {}
			for i in range(self.k):
				self.classes[i] = []

			#find the distance between the point and cluster; choose the nearest centroid
			for features in X:
				distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classes[classification].append(features)

			previous = dict(self.centroids)

			#average the cluster datapoints to re-calculate the centroids
			for classification in self.classes:
				self.centroids[classification] = np.average(self.classes[classification], axis = 0)

			isOptimal = True

			for centroid in self.centroids:
				original_centroid = previous[centroid]
				curr = self.centroids[centroid]
				if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
					isOptimal = False
				#break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
	
			if isOptimal:
				break

if __name__ == '__main__':
	km = K_Means(2)
	X= -2 * np.random.rand(200,2)
	X1 = 1 + 2 * np.random.rand(100,2)
	X[100:200, :] = X1
	#X = np.array([[2,4],[2,6], [2,8], [10,4], [10,6], [10,8]])
	km.fit(X)
	
	# Visualization
	colors = 10*["m","y","r", "g", "c", "b", "k"]
	for classification in km.classes:
		color = colors[classification]
		plt.scatter(km.centroids[classification][0], km.centroids[classification][1], s = 130, marker = "x", color = color)
		for features in km.classes[classification]:
			plt.scatter(features[0], features[1], color = color,s = 30)
	plt.show()
