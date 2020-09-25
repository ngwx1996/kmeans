#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;

__global__ void setClusters(thrust::device_ptr<float> pointVec, thrust::device_ptr<int> cur_cluster,
                            thrust::device_ptr<float> means, thrust::device_ptr<float> sums, 
                            int vecSize, int k, int dimensions, int gridSize, 
                            thrust::device_ptr<int> counter, thrust::device_ptr<int> done) {
    // Shared contains all dimensions, each with size blockDim.x
    extern __shared__ float shared[];
    int* shared_count = (int*)&shared[blockDim.x * dimensions];

    int localIdx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= vecSize)
        return;

    // Use shared mem to store means
    if (localIdx < k * dimensions) {
        // Put all means into shared memory to reduce time to take from global memory
        shared[threadIdx.x] = means[threadIdx.x];
    }
    __syncthreads();

    float minDist = FLT_MAX;
    int bestCluster = INT_MAX;
    float distance;

    // Each thread = point. Calculate min distance to the best cluster from each point.
    for (int i = 0; i < k; i++) {
        distance = 0;
        for (int j = 0; j < dimensions; j++)
        {
            distance += (pointVec[idx + vecSize * j] - shared[i + k * j]) * (pointVec[idx + vecSize * j] - shared[i + k * j]);
        }
        if (distance < minDist) {
            minDist = distance;
            bestCluster = i;
        }
    }
    // If new cluster is found, mark done as false (1) so host can continue iteration
    if (cur_cluster[idx] != bestCluster) {
        cur_cluster[idx] = bestCluster;
        done[0] = 1;
    }

    __syncthreads();

    // For each cluster, find all points corresponding to it and add their values to a single sum
    for (int j = 0; j < k; j++) {
        // If point in cur cluster, put it in shared mem
        for (int curAxis = 0; curAxis < dimensions; curAxis++) {
            shared[localIdx + curAxis * blockDim.x] = (bestCluster == j) ? pointVec[idx + vecSize * curAxis] : 0;
        }
        shared_count[localIdx] = (bestCluster == j) ? 1 : 0;
        __syncthreads();

        // Sum all values in cur cluster
        // Reduction to get sum at tid 0
        for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
            if (localIdx < d && idx + d < vecSize) {
                for (int curAxis = 0; curAxis < dimensions; curAxis++) {
                    shared[localIdx + curAxis * blockDim.x] += shared[localIdx + curAxis * blockDim.x + d];
                }
                shared_count[localIdx] += shared_count[localIdx + d];
            }
            __syncthreads();
        }

        // Value at tid 0 has the block's ans
        if (localIdx == 0) {
            int clusterIdx = j + blockIdx.x * k;
            for (int curAxis = 0; curAxis < dimensions; curAxis++) {
                sums[clusterIdx + curAxis * k * gridSize] = shared[localIdx + curAxis * blockDim.x];
            }
            counter[clusterIdx] = shared_count[localIdx];
        }
        __syncthreads();
    }
}

__global__ void getNewMeans(thrust::device_ptr<float> means, thrust::device_ptr<float> sums, 
                            thrust::device_ptr<int> counter, int k, int dimensions) {
    // Shared will contain sums from all blocks in each dimension
    extern __shared__ float shared[];
    int* shared_count = (int*)&shared[dimensions * blockDim.x];

    int idx = threadIdx.x;
    int blocks = blockDim.x / k;
    // Put sums into shared memory
    for (int curAxis = 0; curAxis < dimensions; curAxis++) {
        shared[idx + blockDim.x * curAxis] = sums[idx + blockDim.x * curAxis];
    }
    shared_count[idx] = counter[idx];
    
    __syncthreads();

    if (idx < k) {
        // Add all sums for each cluster into the cluster (0 < x < k) idx
        for (int j = 1; j < blocks; j++) {
            for (int curAxis = 0; curAxis < dimensions; curAxis++) {
                shared[idx + blockDim.x * curAxis] += shared[idx + j * k + blockDim.x * curAxis];
            }
            shared_count[idx] += shared_count[idx + j * k];
        }
    }
    __syncthreads;

    if (idx < k) {
        // Divide sum by count and put into means array
        // Reset sum array
        int count = (shared_count[idx] > 0) ? shared_count[idx] : 1;

        for (int curAxis = 0; curAxis < dimensions; curAxis++) {
            means[idx + k * curAxis] = shared[idx + blockDim.x * curAxis] / count;
            sums[idx + blockDim.x * curAxis] = 0;
        }
        counter[idx] = 0;
    }
}

int main(int argc, char* argv[]) {
    cout << "---Thrust---" << endl;
    int k = 5;
    int iters = 300;
    int dimensions = 3;
    int vecSize = 100000;
    string filename = "test3d100000.csv";
    ifstream infile(filename.c_str());
    string line;

    if (!infile.is_open()) {
        cout << "Error: Failed to open file" << endl;
        return 1;
    }

    thrust::host_vector<float> h_pointVec(dimensions * vecSize);

    float val;
    char eater;
    int curPoint = 0;
    int offset;
    // Add point to vector
    while (getline(infile, line)) {
        stringstream is(line);
        offset = 0;
        while (is >> val) {
            h_pointVec[curPoint + vecSize * offset] = val;
            is >> eater;
            offset++;
        }
        curPoint++;
    }
    infile.close();
    cout << "Fetched data successfully" << endl;

    thrust::device_vector<float> d_pointVec = h_pointVec;

    thrust::host_vector<float> h_means(k * dimensions);

    int check;
    // Initialize clusters
    for (int i = 0; i < k; i++) {
        while (true) {
            int idx = rand() % vecSize;
            check = 0;
            for (int j = 0; j < dimensions; j++) {
                if (thrust::find(h_means.begin() + k * j, h_means.begin() + k * (j + 1), h_pointVec[idx + vecSize * j])
                    == h_means.begin() + k * (j + 1)) {
                    check++;
                }
                h_means[i + j * k] = h_pointVec[idx + vecSize * j];
            }
            if (check > 0) {
                break;
            }
        }
    }

    cout << k << " clusters initialized" << endl;

    cout << "Running K-means clustering" << endl;

    int blockSize = 1024;
    int gridSize = (vecSize - 1) / blockSize + 1;
    // shared mem has 3 layers
    int sharedSizeCluster = blockSize * (dimensions * sizeof(float) + sizeof(int));
    int sharedSizeMeans = k * gridSize * (dimensions * sizeof(float) + sizeof(int));

    thrust::device_vector<float> d_means = h_means;
    thrust::device_vector<int> d_cur_cluster(vecSize, INT_MAX);
    thrust::device_vector<float> d_sums(k * gridSize * dimensions);
    thrust::device_vector<int> d_counter(k * gridSize,0);
    thrust::device_vector<int> d_done(1, 0);

    auto start = chrono::high_resolution_clock::now();

    int iter;
    for (iter = 0; iter < iters; iter++) {
        // Clear done to prepare for new iteration
        thrust::fill(d_done.begin(), d_done.end(), 0);

        setClusters << <gridSize, blockSize, sharedSizeCluster >> > (d_pointVec.data(), d_cur_cluster.data(), d_means.data(), 
                                                    d_sums.data(), vecSize, k, dimensions, gridSize, d_counter.data(), d_done.data());

        getNewMeans << <1, k * gridSize, sharedSizeMeans >> > (d_means.data(), d_sums.data(), d_counter.data(), k, dimensions);

        if (d_done[0] == 0)
            break;
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "Clustering completed in iteration : " << iter << endl << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

    for (int i = 0; i < k; i++) {
        cout << "Centroid in cluster " << i << " : ";
        for (int j = 0; j < dimensions; j++) {
            cout << d_means[i + k * j] << " ";
        }
        cout << endl;
    }
}