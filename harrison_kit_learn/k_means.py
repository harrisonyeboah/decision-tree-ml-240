# Extracted verbatim from kmeans.ipynb (code, docstrings, comments and markdown notes) so it can be imported.


# Introduction
#
# This is my second project from CS240 Introduction to Machine Learning
#
# K-Means Algorithm Implementation
#
# Data set: Windows XP Homepage and Harrison Yeboah LinkedIn picture
#
# Auhthor: Harrison Yeboah

# Hypothesis
#
# We are expecting k number of centroids (colors) that accurately represents the rbg points in the windowsXPImage and LinkedInImage.
# Through the proofs from last week. We are also expecting inertia to converge with an elbow shape. Where an ideal k is the elbow portion.

# This are my imports the necessary imports for this is numpy and python pillow 
#Numpy will be for my calculations 
import numpy as np

#PIL will be for turning the image into a vector shape
from PIL import Image

#Random will be for selecting random point, but this will be optimized with kmean ++ soon 
import random

# This library will be for showing the colors
import matplotlib.pyplot as plt


# Normalization
# Because K means makes disantace based calculations. In general it would be good to normalize each feature column, so they are equally weighted in scale.
#
# This is necessary becuase some features have a larger numerical range than others. This is not that important in the RBG scenario where each feature of R, G, and B go from 0 to 255. Relative to other dataset where we would using columns like number of bedrooms and number of sq ft. Where bedrooms have a range of 1 to about 8 (usaually) while sqft can go from 500 to 100,000 in some cases.
#
# Min max normalization will put each from 0 to 1

# Implementation normalize data not necessarily needed 
def normalize_data(data):
    '''
    Preconditions ->data -> (numpy array) 
    Postconditions -> initizalized points of norm_data

    This funciton takes in data and normalizes it. 
    '''

    if not isinstance(data, np.ndarray):
        raise TypeError("parameter:d data must be a numpy array")
        
    # Normalize each feature column separately (axis=0), since columns like area and smoothness have very different scales
    norm_data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    return norm_data


# Initializing points with k means ++
# In kmeans there are points that are randomly assigned to the dataspace based on k. Based on this class there are two main methods that we
# learned aboyut in k means point initialzation. One being that assigning the initial clusters to random points, and the other being assigning initial clusters
# based on k means. The problem behind assigning to initial clusters is that some clusters do not get assigned to to any points since that cluster could be in a really
# far part of the instance graph. Therefore you use kmeans ++
#
#
# How does k means ++ work
# In the first point chose a random point in the instance space.
# Run for loop until centers = k
# For every pixel, compute the squared Euclidean distance to its nearest existing center (D(x)²).
#
# Make a probability distribution with dists_sq / dists_sq.sum()
# Use the distribution to sample the next point.
# centers.append(pixels[next_index]) will append the next center then we repeat until len(centers) = k

# Initialize data
def initialize_points(k, data):
    '''
    Preconditions -> k (int), data -> (numpy array)
    Postconditions -> initial clusters of data.

    This function currently uses randomization to initialize points, but will be optimized to k means++.

    My source comes from this video https://www.youtube.com/watch?v=4qJWhvFQb9g
    to lead me towards implementation of k means ++ . I will come back to this right now just random points
    '''
    # Ensure defense programming
    if k <= 0:
        raise ValueError("parameter: k must be greater than 0")
    if not isinstance(data, np.ndarray):
        raise TypeError("parameter: data must be a numpy array")

    # Flatten my dataset for easy managment 
    number_of_colors = data.shape[-1]            
    pixels = data.reshape(-1, number_of_colors)
    pixels = pixels.astype(float)               
    num_pixels = pixels.shape[0]                

    # Then the first center will be chose randomly. 
    random_index = np.random.randint(num_pixels) 
    first_center = pixels[random_index]         
    centers = [first_center]                     

    # Keep doing this until centers = k
    while len(centers) < k:

        # Get the distance squared
        dists_sq = np.full(num_pixels, np.inf)

        # Getting euclidean distance from each center.
        for c in centers:
            diff = pixels - c                    
            squared = diff ** 2                  
            d = np.sum(squared, axis=1)          
            dists_sq = np.minimum(dists_sq, d)

        # Build our probability distribtion
        probs = dists_sq / dists_sq.sum()        

        # Use it to get our new center
        next_index = np.random.choice(num_pixels, p=probs)
        centers.append(pixels[next_index])

    return centers


# Calc Inertia
# Inertia is basically the number of squared distances from each points assigned center. This is basically our loss function.
#
# How does calcuating inertia work
# Start of with inertia = 0
# Loop through every center (to peform calucaltion on all of points assigned to it)
# Loop through each point assigned to such center and get the squared distance
# Add it to the current inertia until we get through all points.

def calc_inertia(centers, closest_points_dictionary):
    '''
    This function will calculate the inertia. 

    Preconditions: centers: np.array of len(k) , closesp_points_dicitonary : dict

    Post condition: returns an float value all the acumulated absolute value square distances for inertia. 
    '''
    inertia = 0 
    for i in range (len(centers)):
        for each_point in closest_points_dictionary[i]:
            current_difference = np.sum((centers[i] - each_point) ** 2)
            inertia += current_difference
    return inertia


# Assignining Points to center
#
#
# How does assign_points_to_centers work
# Create an empty closest points dictionary which will have the centroid as the key
# Looop thrhough each trainging point (This will be a vector of features)
# Calcauate the distance to all the centers and append it to a list.
# Get the index of the minimum and assign it that key of the dictionary
# Loop until you loop through all points. Once you loop through all points you return the the dictionary.

def assign_points_to_centers(k, data, centers):
    '''
    Preconditions -> k (int) -> the number of centers,
                     centers -> list 
    Postconditions -> returns closest_points_dictionary (This will hold each cluster with the closest points. 
    '''
    rows, columns = data.shape

    closest_points_dictionary = {}
    key = 0
    while key < k:
        closest_points_dictionary[key] = []
        key += 1

    # This will assign each training point to a center
    for i in range(rows):
        # Each row is one training point (a vector of features)
        current_point = data[i]
        c = 0
        distances = []
        while c < len(centers):
            distance = np.linalg.norm(current_point - centers[c])
            distances.append(distance)
            c += 1
        min_index = distances.index(min(distances))
        closest_points_dictionary[min_index].append(current_point)

    return closest_points_dictionary


# Updating centers.
#
#
# How does update_centers work
# You take in the closest_points_dictionary and its centers.
# Loop through the closest_points_dictionary to get each cluster key.
# Case 1: If the cluster is a point then we get the mean of all the assigned values and make that the new center
# Case 2: If the cluster is not a point then just append it to the new centers.
# Repeat until you loop thorugh all key value pairs of closest_points_dictionary, then return the new centers.

def update_centers(closest_points_dictionary, centers):
    '''
    Preconditions -> closest_points_dictionary -> dicct,
                     centers -> list
    Postconditions -> return new_centers -> list 
    '''
    # New centers are currently empty.
    new_centers = []

    # Loops through the closest_points dictionry to assign new centers. 
    for cluster_key in closest_points_dictionary:
        assigned = closest_points_dictionary[cluster_key]
        if len(assigned) > 0:
            new_centers.append(np.mean(assigned, axis=0).tolist())
        else:
            new_centers.append(list(centers[cluster_key]))
            
    return new_centers


# Vector QuantImage .
#
#
# How does vector_quant_image work
# You take in the centers and the image. centers is basically the colors and the image is the data aquired from the pillow library.
# Perform some defense programming to make sure that my image is an np.array.
# Create a empty flattened numpy array to paint our picutre.
# Loop through each (r, g, b) vector and paint the final_image with the closet centroid.
# At the end of the loop reshape the flattened vector to its original. shape.
# Then use matplot lib to display it

def vector_quant_image(centers, image):
    '''
    Preconditions: centers [color 1, ...., color n], image: np.array
    Postconditions: Shows the vector quant image. 
    '''


    if not isinstance(image, np.ndarray):
        raise TypeError("parameter:d data must be a numpy array")

    # Image shape
    height, width, channels = image.shape
    
    # Flattens the image into 1D
    final_image = []
    
    # Loop through every rgb value
    for i in range(height):
        for j in range(width):
            # Create a list to slowly aquire distances
            distances = []
            # Loop through each center and get the distaance and get the minimum distance index. 
            for k in range(len(centers)):
                distance = np.linalg.norm(image[i][j] - centers[k])
                distances.append(distance)
            min_index = distances.index(min(distances))
            final_image.append([centers[min_index]])

    final_image = np.array(final_image)
    final_image = final_image.reshape(height, width, channels)


    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    
    plt.imshow(final_image)
    plt.axis("off")
    plt.show()
    
    # optional: save it
    plt.imsave("quantized.png", final_image)
        
    return final_image


def display_colors(colors):
    '''
    Preconditions: -> Takes in colors within list of colors (rgb) 
    Postconditions -> returns None just shows the colors
    '''

    plt.figure(figsize=(12, 2))
    plt.imshow([np.array(colors) / 255.0])
    plt.axis("off")
    plt.title("Initial Points")
    plt.show()

    return None


def display_inertia_graph(inertias):
    '''
    Preconditions: -> Takes in inertias which is a list (r
    Postconditions -> returns None just displayes the inertia graph. 
    '''
    # This is the inertias graph.
    # I am starting at three because index 0 and 1 have negative values
    k_values = range(1, len(inertias[2:]) + 1)
    
    
    plt.figure(figsize=(8, 5))
    # FIX: Changed 'inertias' to 'inertias[2:]' to match the length of k_values
    plt.plot(k_values, inertias[2:], marker='o', linestyle='-', color='b')
    
    # Add labels and title
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Inertia with Elbow for optimal k')
    plt.xticks(k_values)
    plt.grid(True)
    
    # Display the plot
    plt.show()


def perform_iteration_until_convergence(k, data, initial_points):
    '''
    Preconditions -> k (int), data -> (numpy array) 
    Postconditions -> initial clusters of data.

    This function currently uses randomization to initialize points, but will be optimized to k means++. 

    My source comes from this video https://www.youtube.com/watch?v=4qJWhvFQb9g
    to lead me towards implementation of k means ++ . I will come back to this right now just random points
    '''
    # Get the shape of the data
    rows, columns = data.shape

    first_time = True
    
    #Globally declaring vector_quant_image this will be reshaped later but right now for simplicity I will flatten it into 1d array.
    vector_quant_image = []
    
    # Making inertia -1 and -2 since we can't have -1 and -2 inertia this will then help us determine weather it converged or not. 
    inertias = [-2, -1]
    # (normalized inertia is small, so compare with a tolerance instead of int())
    while abs(inertias[-1] - inertias[-2]) > 1e-9:
        # Empty vector quant image that we will paint overtime.
        vector_quant_image = []
        
        if first_time == True:
            centers = initial_points
            first_time = False
        else: 
            centers = new_centers

        # This is the assignment funciton 
        closest_points_dictionary = assign_points_to_centers(k , data, centers)

        # This will calculate the inertia and build it in a list.       
        inertia = calc_inertia(centers, closest_points_dictionary)
        
        inertias.append(inertia)

        # This will do the calculation for the new centers. 
        new_centers = update_centers(closest_points_dictionary, centers)

    # The reshape will reshape the flattened 1d array back into the original picture shape
    return new_centers, inertias, data


# Based on the the overleaf proof pseudo code from last week.
# K-Means
#
# Pseudocode Algorithm
#
# **Algorithm: K-Means**
#
# 1. Select a value for $K$, the number of centers
# 2. Normalize the dataset.
# 3. Initialize centers by selecting $K$ random training points
# 4. **repeat**
# 5.     *// Assignment phase assign_points_to_centers() funciton that I wrote at the top*
# 6.     **for** each training point $s_i$ **do**
# 7.         Compute the distance of $s_i$ to each center
# 8.         Assign $s_i$ to the closest center
# 9.     **end for**
# 10.     *// Adjustment phase update_centers() function that I wrote at the top *
# 11.    **for** each center $C$ **do**
# 12.        Compute the mean of all samples assigned to $C$
# 13.        Move $C$ to the location of the mean
# 14.    **end for**
# 15. **until** no centers move

# The official K means function the main final one
def k_means(k, data):
    '''
    This is just the pseduocode for the k means 
    
    # This wiill give us a normalized version of the data
    norm_data = norm(data)
    
    # Based off this we might use k means ++ in order to initialize the points of length k and return them as numpy arrays 
    current_points = initialize_the_points(k, norm_data) 
    
    '''

    # Normlize the data 
    norm_data = normalize_data(data)
    
    # Intializing the points 
    initial_points = initialize_points(k, norm_data)

    # Performing the k means until convergence
    centers, inertias, data = perform_iteration_until_convergence(k, norm_data, initial_points)


    return centers, inertias
