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
                            float* sums, int vecSize, int k, int dimensions, int* counter, int* done) {
    extern __shared__ float shared_means[];
    // idx corresponds to point id
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vecSize)
        return;

    if (threadIdx.x < k * dimensions) {
        // Put means into shared memory to reduce time to take from global memory
        shared_means[threadIdx.x] = means[threadIdx.x];
    }
    __syncthreads();

    float minDist = FLT_MAX;
    int bestCluster = INT_MAX;
    float distance;

    // Find best cluster for each point
    for (int i = 0; i < k; i++) {
        distance = 0;
        for (int j = 0; j < dimensions; j++)
        {
            distance += (pointVec[idx + vecSize * j] - shared_means[i + k * j]) * (pointVec[idx + vecSize * j] - shared_means[i + k * j]);
        }
        if (distance < minDist) {
            minDist = distance;
            bestCluster = i;
        }
    }

    // If cluster changed, update cluster id list and set done to false (1)
    if (cur_cluster[idx] != bestCluster) {
        cur_cluster[idx] = bestCluster;
        done[0] = 1;
    }

    // Atomically add sums and counter with cluster
    for (int i = 0; i < dimensions; i++) {
        atomicAdd(&sums[bestCluster + k * i], pointVec[idx + vecSize * i]);
    }
    atomicAdd(&counter[bestCluster], 1);
}

__global__ void getNewMeans(float* means, float* sums, int* counter, int k, int dimensions) {
    // Get new mean for each cluster
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = (counter[idx] != 0) ? counter[idx] : 1;

    for (int i = 0; i < dimensions; i++) {
        means[idx + k * i] = sums[idx + k * i] / count;
    }
}

int main(int argc, char* argv[]) {
    cout << "---CUDA Atomic---" << endl;
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

    int* d_cur_cluster;
    float* d_means;
    float* d_sums;
    int* d_counter;
    int* d_done;

    cudaMalloc(&d_cur_cluster, vecSize * sizeof(int));
    cudaMalloc(&d_means, k * dimensions *sizeof(float));
    cudaMemcpy(d_means, h_means.data(), k * dimensions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_sums, k * dimensions * sizeof(float));
    cudaMalloc(&d_counter, k * sizeof(int));
    cudaMalloc(&d_done, sizeof(int));

    int blockSize = 1024;
    int gridSize = (vecSize - 1) / blockSize + 1;
    int sharedSize = dimensions * k * sizeof(float);

    auto start = chrono::high_resolution_clock::now();

    int iter;
    for (iter = 0; iter < iters; iter++) {
        // Clear sum and counter array to prepare for new iteration
        cudaMemset(d_sums, 0, k * dimensions * sizeof(int));
        cudaMemset(d_counter, 0, k * sizeof(int));
        cudaMemset(d_done, 0, sizeof(int));

        // For each point, find nearest cluster and add itself to the cluster's sum
        setClusters << <gridSize, blockSize, sharedSize >> > (d_pointVec, d_cur_cluster, d_means, d_sums, vecSize, k, dimensions, d_counter, d_done);

        // Get new mean of each cluster
        getNewMeans << <1, k >> > (d_means, d_sums, d_counter, k, dimensions);

        // Check if done became false(1), if so continue
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