#!/usr/local/bin/python3
#
# Authors: [Prashanth sateesh, Aditya Kartikeya, Vansh] User ID: [psateesh, admall, vanshah]
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2019
#

from PIL import Image
from numpy import *
import numpy as np
from scipy.ndimage import filters
import sys
import imageio
import copy
import scipy.stats
import math
# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)
#This is a function used to assign emission probabilities.This fucntion works only for the viterbi algorithm with human input
def emission_probability_calculator(edge_strength,x,y):
    emission_prob=copy.deepcopy(edge_strength)
    emission_prob=np.transpose(emission_prob)
    #lo
    for i in range(len(emission_prob)):
                   emission_prob[i]=np.log(emission_prob[i]/np.sum(emission_prob[i])+0.0001)
    emission_prob[x]=0
    emission_prob[x][y]=1
    
    return emission_prob
#This is to generate emission probabilities for the viterbi model without human input
def emission_probability_calculator_simple(edge_strength):
    emission_prob=copy.deepcopy(edge_strength)
    emission_prob=np.transpose(emission_prob)
    for i in range(len(emission_prob)):
                   emission_prob[i]=np.log(emission_prob[i]/np.sum(emission_prob[i])+0.0001)
    
    
    return emission_prob


def draw_edge(image, y_coordinates, color, thickness,input_image):
    for (x, y) in enumerate(y_coordinates):
        a=int(max(y-int(thickness/2), 0))
        b=int(min(y+int(thickness/2), image.shape[1]-1))
        for t in range(a,b):
            input_image.putpixel((x, t), color)
    return np.asarray(input_image)
#This function allocates transition probabilities.It calculates the difference in rows between adjacent columns.If the difference is large ,it allocates a lower
#probability as transition a large distance in one column change is not possible.
def transition_prob_allocator(ind,x):
    return 1/(abs(ind-x)+(1-0))
#The below function is for viterbi algorithm with human input.
#Here we will start the algortihm from 20 different points at the start and the develop 20 paths.Then we would compare the distance of every path to the human input.
#We choose the point with the lowest distance.
def main(edge_strength,x,y):
    emission_probs=emission_probability_calculator(np.square(edge_strength),x,y)
    emission_probs=emission_probs
    a=sort(emission_probs[0])
    #Taking 20 start points with highest gradients in the first column.To find the highest gradient points we use the sort function.
    probs_start=a[-20:]
    
    probs_indices=[np.where(emission_probs[0] == i)[0][0] for i in probs_start]
    paths=[]
    for i in probs_indices:
        
        probs_1=[]
        probs_2=[]
        initial_start_index=i
        initial_start_prob=emission_probs[0][initial_start_index]
        probs_1+=[initial_start_prob]
        path=[initial_start_index]
        probs=[initial_start_prob]
        for j in range(1,len(emission_probs)):
            temp_arr=[]
            for i in range(0,len(emission_probs[0])):
                a=initial_start_prob
                b=emission_probs[j][i]
                c=math.log(transition_prob_allocator(initial_start_index,i))
                #if probability of transitioning is low then give more importance to gradient.Adding a number gives more importance to gradient.We add a number because we took logarithm.
                if abs(initial_start_index-i)+(1-0)>3:
                    temp_arr+=[a+b+c+5.5]
                else:
                    #Here we give more importance to transition probability.This we do by multiplying the transition probability term.
                    #A negative term has been multiplied because negative*negative=positive.So importance increases.
                    temp_arr+=[a+b-5*c]
            initial_start_index=temp_arr.index(max(temp_arr))
            initial_start_prob=max(temp_arr)
            probs_1+=[initial_start_prob]
            path+=[initial_start_index]
            

            
        paths+=[path]
    paths_2=[]
    for i in probs_indices:
        
        
        initial_start_index=i
        initial_start_prob=emission_probs[0][initial_start_index]
        path=[initial_start_index]
        probs=[initial_start_prob]
        probs_2+=[initial_start_prob]
        for j in range(1,len(emission_probs)):
            temp_arr=[]
            for i in range(0,len(emission_probs[0])):
                a=initial_start_prob
                b=emission_probs[j][i]
                c=math.log(transition_prob_allocator(initial_start_index,i))
                if abs(initial_start_index-i)+(1-0)>3:
                    if transition_prob_allocator(initial_start_index,i)<1/2:
                       temp_arr+=[a+b+c*5.5]
                else:
                    temp_arr+=[a+b-5*c]
            initial_start_index=temp_arr.index(max(temp_arr))
            initial_start_prob=max(temp_arr)
            probs_2+=[initial_start_prob]
            path+=[initial_start_index]
            
   
            
        paths_2+=[path]
    probs_indices=probs_indices+probs_indices           
    return paths+paths_2,probs_indices
    

    
    
    


# main program
#
#(input_filename, gt_row, gt_col) = sys.argv[1:]
(input_filename) = sys.argv[1]
# load in image 
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)


imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))

ridge=[np.argmax(i) for i in np.transpose(edge_strength)]

#x=int(input("x-coordinate:"))
#y=int(input("y-coordinate:"))
x=int(sys.argv[2])
y=int(sys.argv[3])
a,probs_indices=main(edge_strength,x,y)

dist_1=[]
dist_2=[]
for ele in a:
    dist_1+=[abs(ele[x]-y)]
for ele in probs_indices:
    dist_2+=[abs(ele-y)]
total_dist=np.array(dist_1)+np.array(dist_2)
min_ele=np.argmin(total_dist) 

np_img=np.asarray(input_image)
#np_img=np_img[0:9,:]
for i in range(len(a)):
    
    imageio.imwrite("output.jpg", draw_edge(np_img,np.array(a[min_ele])-1, (0, 255,0), 5,input_image))
    
#The below function is for viterbi without human interaction.The only difference from the other one above here is that we calculate starting from only one point.
def main_just_viterbi(edge_strength,x,y):
    emission_probs=emission_probability_calculator(np.square(edge_strength),x,y)

    emission_probs=emission_probs
    a=sort(emission_probs[0])
    
    probs_start=a[-1:]
    
    probs_indices=[np.where(emission_probs[0] == i)[0][0] for i in probs_start]
    paths=[]
    for i in probs_indices:
        
        probs_1=[]
        probs_2=[]
        initial_start_index=i
        initial_start_prob=emission_probs[0][initial_start_index]
        probs_1+=[initial_start_prob]
        path=[initial_start_index]
        probs=[initial_start_prob]
        for j in range(1,len(emission_probs)):
            temp_arr=[]
            for i in range(0,len(emission_probs[0])):
                a=initial_start_prob
                b=emission_probs[j][i]
                c=math.log(transition_prob_allocator(initial_start_index,i))
                if abs(initial_start_index-i)+(1-0)>3:
                    temp_arr+=[a+b+c+5.5]
                else:
                    temp_arr+=[a+b-5*c]
            initial_start_index=temp_arr.index(max(temp_arr))
            initial_start_prob=max(temp_arr)
            probs_1+=[initial_start_prob]
            path+=[initial_start_index]
            

            
        paths+=[path]
    paths_2=[]
    for i in probs_indices:
        
        
        initial_start_index=i
        initial_start_prob=emission_probs[0][initial_start_index]
        path=[initial_start_index]
        probs=[initial_start_prob]
        probs_2+=[initial_start_prob]
        for j in range(1,len(emission_probs)):
            temp_arr=[]
            for i in range(0,len(emission_probs[0])):
                a=initial_start_prob
                b=emission_probs[j][i]
                c=math.log(transition_prob_allocator(initial_start_index,i))
                if abs(initial_start_index-i)+(1-0)>3:
                    if transition_prob_allocator(initial_start_index,i)<1/2:
                       temp_arr+=[a+b+c*5.5]
                else:
                    temp_arr+=[a+b-5*c]
            initial_start_index=temp_arr.index(max(temp_arr))
            initial_start_prob=max(temp_arr)
            probs_2+=[initial_start_prob]
            path+=[initial_start_index]
            
   
            
        paths_2+=[path]
    probs_indices=probs_indices+probs_indices           
    return paths+paths_2,probs_indices
    

    
    
    

(input_filename) = sys.argv[1]
input_image = Image.open(input_filename)
imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))

ridge=[np.argmax(i) for i in np.transpose(edge_strength)]

a,probs_indices=main_just_viterbi(edge_strength,x,y)

dist_1=[]
dist_2=[]
#Here we are calculating the path to which e have minimum distance to from the human given point.
for ele in a:
    dist_1+=[abs(ele[x]-y)]
for ele in probs_indices:
    dist_2+=[abs(ele-y)]
total_dist=np.array(dist_1)+np.array(dist_2)
min_ele=np.argmin(total_dist) 

np_img=np.asarray(input_image)
for i in range(len(a)):
    
    imageio.imwrite("output_map.jpg", draw_edge(np_img,np.array(a[min_ele])-1, (255, 0, 0), 5,input_image))
emission_probs_simple=emission_probability_calculator_simple(np.square(edge_strength))
rows_high=np.argmax(emission_probs_simple, axis=1)
(input_filename) = sys.argv[1]
input_image = Image.open(input_filename)
for i in range(len(a)):
    
    imageio.imwrite("output_simple.jpg", draw_edge(np_img,rows_high-1, (0, 0, 255), 5,input_image))
