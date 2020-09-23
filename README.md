
# Kmeans using CUDA and Thrust

## Concept
Kmeans is an iterative algorithm that attempts to split a dataset into K distinct clusters. This is achieved by assigning data points to its nearest cluster's centroid. <br />
Performing the algorithm sequentially takes a long time as the CPU has to loop through all the points to find their closest cluster. Instead, using parallel programming can split each point into their own thread to be assigned to the closest cluster.<br />
The aim of this project is to determine how much faster does parallelizing the algorithm take compared to sequentially computing the algorithm.<br />
Parallelization will be done in CUDA and Thrust to compare their differences. <br />
For more information: https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a<br />

## Build and run
Visual Studio with CUDA required.<br />
To run the project, open the solution in Visual Studio.<br />
Change the k value, number of iterations, and file name in the main functions of each code.<br />
--Default--
k = 5<br />
iter = 300<br />
filename = test100000.csv<br />
To get the best performance, the code should be run in Release mode. Thrust might also error in debug mode.<br />

## Project structure

```
.
kmeans
│   
│   kmeans.sln
|
└───kmeans_cuda
│   │   kernel.cu
│   
└───kmeans_cuda_atomic
│   │   kernel.cu
│ 
└───kmeans_seq
│   │   kmeans_seq.cpp
│     
└───kmeans_thrust
│   │   kernel.cu
│   
└───kmeans_thrust_atomic
│   │   kernel.cu
```

## Sequential
  * **Main** <br />
    Main function where the user can change the k value, iterations, and filename. Loads values into a vector of points and computes kmeans with them. Exits while loop when all clusters converges to local minima.<br />
    
  * **getNearestCluster** <br />
    Given a point and the clusters, find the nearest cluster corresponding to the point. Determined finding the minimum distance to each cluster's centroid. <br />

  * **Class: Point** <br />
    The point class stores the point values in each axis, the point and cluster Id, and its dimensions.<br />
    
  * **Class: Cluster** <br />
    The cluster class holds a vector of points that are associated with the cluster and the centroid of the cluster.<br />


## Parallel - Atomic
  * **Main** <br />
    Main function where the user can change the k value, iterations, and filename. Loads values into a vector of points and initializes device variables. Calls kernel functions in each iteration to get the cluster's sum and centroid (means). Exits while loop when all clusters converges to local minima by using a variable done that will check if any changes to the clusters were made in kernel.<br />
    
  * **setCluster** <br />
    Atomically add each point's value to the nearest cluster's sum and increment their counter by 1. Stores clusters' means into shared memory to minimize the need of accessing global memory. Inefficient as each thread has to queue to perform atomicadd to the cluster's sum. <br />

  * **getNewMeans** <br />
    For each cluster, calculate the new means based on the sum and counter calculated in setCluster.<br />
    

## Parallel - Shared Memory and Reduction
  * **Main** <br />
    Main function where the user can change the k value, iterations, and filename. Loads values into a vector of points and initializes device variables. Calls kernel functions in each iteration to get the cluster's sum and centroid (means). Exits while loop when all clusters converges to local minima by using a variable done that will check if any changes to the clusters were made in kernel.<br />
    
  * **setCluster** <br />
    Uses 3 layers of shared memory to hold all point values (x and y) and counter in each block. For each cluster, find all points in the block that belongs to the cluster and add their values by using parallel reduction. Thread Id 0 will hold the total sum of values and the counter for each cluster, which will be saved into global memory. <br />

  * **getNewMeans** <br />
    Uses 3 layers of shared memory to hold all point values (x and y) and counter in each block. There are k * gridSize number of threads which adds their corresponding values (x and y) and counter into shared memory. Each threads with Id smaller than k will represent a cluster and add all the sums calculated in setCluster, which will be divided to get the new cluster's mean (centroid).<br />


## Results
### Hardware
| | Component | Component used |
|--|--|--|
|1.| Processor |  Intel Core i5-4690k CPU @ 3.50GHz |
|2.|Memory |  8GB |
|3.| GPU | NVIDIA GTX970 |

### Experiment

|  | n = 1000 | n = 100000 |
|--|--|--|
| Sequential | 12762μs | 7367568μs |
| Cuda Atomic | **929μs** |  22802μs |
| Thrust Atomic | 2715μs | 30885μs |
| Cuda Shared & Reduce | 1066μs | 11681μs |
| Thrust Shared & Reduce| 1265μs | **11298μs** |

  
## Conclusion
Using CUDA or Thrust to perform parallel computation with n > 1000 results in a speedup of more than 5x compared to sequential computation. <br />
For n = 1000, using shared memory instead of atomicadd() resulted in more time taken. This might be due to the short queue to atomically add values not requiring a long wait time, which outperformed the time taken to initialize shared memory.
However, for n = 100000, using shared memory and parallel reduction resulted in a speedup of 2x over the atomic functions. This is due to initializing shared memory in each block and using parallel reduction to calculate each block's sums of each cluster, which outperforms the queue to atomically add into global memory.



## Reference
* https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a<br />
* https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/<br />
* http://www.goldsborough.me/c++/python/cuda/2017/09/10/20-32-46-exploring_k-means_in_python,_c++_and_cuda/<br />
