#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace std;

__global__ void setClusters(float* pointVec, int* cur_cluster, float* means,
                            float* sums, int vecSize, int k, int dimensions, int gridSize, int* counter, int* done) {
    extern __shared__ float shared[];
    int* shared_count = (int*)&shared[blockDim.x * dimensions];

    int localIdx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= vecSize)
        return;

    // Use shared mem to store means
    if (localIdx < k * dimensions) {
        // Put means into shared memory to reduce time to take from global memory
        shared[threadIdx.x] = means[threadIdx.x];
        //printf("Cluster %d, axis %d has mean val %f\n", localIdx % k, localIdx / k, means[localIdx]);
    }
    __syncthreads();

    float minDist = FLT_MAX;
    int bestCluster = INT_MAX;
    float distance;

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
    if (cur_cluster[idx] != bestCluster) {
        cur_cluster[idx] = bestCluster;
        done[0] = 1;
    }
    
    __syncthreads();

    for (int j = 0; j < k; j++) {
        for (int curAxis = 0; curAxis < dimensions; curAxis++) {
            shared[localIdx + curAxis * blockDim.x] = (bestCluster == j) ? pointVec[idx + vecSize * curAxis] : 0;
        }
        shared_count[localIdx] = (bestCluster == j) ? 1 : 0;
        //printf("point %d at cluster %d has val %f , %f, %f with bestcluster %d actual %d\n", idx, j, shared[localIdx], shared[localIdx + blockDim.x], shared[localIdx + 2 * blockDim.x],shared_count[localIdx], bestCluster);
        __syncthreads();

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

        // Value at tid 0
        if (localIdx == 0) {
            //printf("cluster %d has shared sum %f, %f and count %d\n", j, shared[localIdx], shared[localIdx + blockDim.x], shared_count[localIdx]);

            //printf("cluster %d total count %d\n", j, shared_count[localIdx]);
            int clusterIdx = j + blockIdx.x * k;
            for (int curAxis = 0; curAxis < dimensions; curAxis++) {
                sums[clusterIdx + curAxis * k * gridSize] = shared[localIdx + curAxis * blockDim.x];
            }
            counter[clusterIdx] = shared_count[localIdx];
            //printf("cluster %d has sum %f, %f and count %d\n", j, sums[clusterIdx], sums[clusterIdx + k * gridSize], counter[clusterIdx]);
        }
        __syncthreads();
    }
}

__global__ void getNewMeans(float* means, float* sums, int* counter, int k, int dimensions) {
    extern __shared__ float shared[];
    int* shared_count = (int*)&shared[dimensions * blockDim.x];


    int idx = threadIdx.x;
    int blocks = blockDim.x / k;
    for (int curAxis = 0; curAxis < dimensions; curAxis++) {
        shared[idx + blockDim.x * curAxis] = sums[idx + blockDim.x * curAxis];
    }
    shared_count[idx] = counter[idx];
    __syncthreads();

    //printf("idx %d for cluster %d has %f , %f with count %d\n", idx, idx % 5, shared[idx], shared[idx + blockDim.x], shared_count[idx]);
    if (idx < k) {
        for (int j = 1; j < blocks; j++) {
            for (int curAxis = 0; curAxis < dimensions; curAxis++) {
                shared[idx + blockDim.x * curAxis] += shared[idx + j * k + blockDim.x * curAxis];
            }
            shared_count[idx] += shared_count[idx + j * k];
        }
    }
    __syncthreads;

    if (idx < k) {
        int count = (shared_count[idx] > 0) ? shared_count[idx] : 1;

        for (int curAxis = 0; curAxis < dimensions; curAxis++) {
            means[idx + k * curAxis] = shared[idx + blockDim.x * curAxis] / count;
            
            //printf("idx %d has sum %f , %f and count %d\n", idx, shared[idx], shared[idx + blockDim.x], count);

            sums[idx + blockDim.x * curAxis] = 0;
        }
        //printf("idx %d has means %f , %f and count %d\n", idx, means[idx], means[idx + k], count);

        counter[idx] = 0;
    }
}

int main(int argc, char* argv[]) {
    cout << "---CUDA---" << endl;
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

    vector<float> h_pointVec(dimensions * vecSize);

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
    cout << "Fetched data successfully from " << filename << endl;

    float* d_pointVec;
    int* h_done = new int(0);

    cudaMalloc(&d_pointVec, dimensions * vecSize * sizeof(float));
    cudaMemcpy(d_pointVec, h_pointVec.data(), dimensions * vecSize * sizeof(float), cudaMemcpyHostToDevice);

    // each dimension has k size
    vector<float> h_means(k * dimensions);

    int check;
    // Initialize clusters
    for (int i = 0; i < k; i++) {
        while (true) {
            int idx = rand() % vecSize;
            check = 0;
            for (int j = 0; j < dimensions; j++) {
                if (find(h_means.begin() + k * j, h_means.begin() + k * (j + 1), h_pointVec[idx + vecSize * j])
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

    int* d_cur_cluster;
    float* d_means;
    float* d_sums;
    int* d_counter;
    int* d_done;

    cudaMalloc(&d_cur_cluster, vecSize * sizeof(int));
    cudaMalloc(&d_means, k * dimensions * sizeof(float));
    cudaMemcpy(d_means, h_means.data(), k * dimensions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_sums, gridSize * k * dimensions * sizeof(float));
    cudaMalloc(&d_counter, gridSize * k * sizeof(int));
    cudaMalloc(&d_done, sizeof(int));
    // Clear sum and counter array to prepare for iteration
    cudaMemset(d_sums, 0, gridSize * k * dimensions * sizeof(float));
    cudaMemset(d_counter, 0, gridSize * k * sizeof(int));

    auto start = chrono::high_resolution_clock::now();

    int iter;
    for (iter = 0; iter < iters; iter++) {
        cudaMemset(d_done, 0, sizeof(int));

        setClusters << <gridSize, blockSize, sharedSizeCluster >> > 
            (d_pointVec, d_cur_cluster, d_means, d_sums, vecSize, k, dimensions, gridSize, d_counter, d_done);

        getNewMeans << <1, k * gridSize, sharedSizeMeans >> > (d_means, d_sums, d_counter, k, dimensions);

        cudaMemcpy(h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_done[0] == 0)
            break;
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "Clustering completed in iteration : " << iter << endl << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

    cudaMemcpy(h_means.data(), d_means, k * dimensions * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < k; i++) {
        cout << "Centroid in cluster " << i << " : ";
        for (int j = 0; j < dimensions; j++) {
            cout << h_means[i + k * j] << " ";
        }
        cout << endl;
    }
}