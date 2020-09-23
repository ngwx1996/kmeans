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

__global__ void setClusters(float* x, float* y, int* cur_cluster, float* means_x, float* means_y, 
                            float* sums_x, float* sums_y, int vecSize, int k, int* counter, int* done) {
    extern __shared__ float shared_means[];
    // idx corresponds to point id
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vecSize)
        return;

    if (threadIdx.x < k) {
        // Put means into shared memory to reduce time to take from global memory
        shared_means[threadIdx.x] = means_x[threadIdx.x];
        shared_means[threadIdx.x + k] = means_y[threadIdx.x];
    }
    __syncthreads();

    float cur_x = x[idx];
    float cur_y = y[idx];
    float minDist = FLT_MAX;
    int bestCluster = INT_MAX;
    float distance;

    // Find best cluster for each point
    for (int i = 0; i < k; i++) {
        distance = (cur_x - shared_means[i]) * (cur_x - shared_means[i]) + (cur_y - shared_means[i + k]) * (cur_y - shared_means[i + k]);
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
    atomicAdd(&sums_x[bestCluster], cur_x);
    atomicAdd(&sums_y[bestCluster], cur_y);
    atomicAdd(&counter[bestCluster], 1);
}

__global__ void getNewMeans(float* means_x, float* means_y, float* sums_x, float* sums_y, int* counter) {
    // Get new mean for each cluster
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = (counter[idx] != 0) ? counter[idx] : 1;

    means_x[idx] = sums_x[idx] / count;
    means_y[idx] = sums_y[idx] / count;
}

int main(int argc, char* argv[]) {
    cout << "---CUDA Atomic---" << endl;
    int k = 5;
    int iters = 300;
    string filename = "test100000.csv";
    ifstream infile(filename.c_str());
    string line;

    if (!infile.is_open()) {
        cout << "Error: Failed to open file" << endl;
        return 1;
    }

    vector<float> h_x;
    vector<float> h_y;
    float val;
    char eater;
    // Add point to vector
    while (getline(infile, line)) {
        stringstream is(line);
        is >> val;
        h_x.push_back(val);
        is >> eater;
        is >> val;
        h_y.push_back(val);
    }
    infile.close();
    cout << "Fetched data successfully from " << filename << endl;

    int vecSize = h_x.size();
    float* d_x;
    float* d_y;
    int* h_done = new int(0);

    cudaMalloc(&d_x, vecSize * sizeof(float));
    cudaMalloc(&d_y, vecSize * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), vecSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), vecSize * sizeof(float), cudaMemcpyHostToDevice);

    vector<float> h_means_x;
    vector<float> h_means_y;

    // Initialize clusters
    for (int i = 0; i < k; i++) {
        while (true) {
            int idx = rand() % vecSize;
            vector<float>::iterator it = find(h_means_x.begin(), h_means_x.end(), h_x[idx]);
            if (it == h_means_x.end() || h_means_y[it - h_means_x.begin()] != h_y[idx]) {
                h_means_x.push_back(h_x[idx]);
                h_means_y.push_back(h_y[idx]);
                break;
            }
        }
    }

    cout << k << " clusters initialized" << endl;

    cout << "Running K-means clustering" << endl;

    int* d_cur_cluster;
    float* d_means_x;
    float* d_means_y;
    float* d_sums_x;
    float* d_sums_y;
    int* d_counter;
    int* d_done;

    cudaMalloc(&d_cur_cluster, vecSize * sizeof(int));
    cudaMalloc(&d_means_x, k * sizeof(float));
    cudaMalloc(&d_means_y, k * sizeof(float));
    cudaMemcpy(d_means_x, h_means_x.data(), k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_means_y, h_means_y.data(), k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_sums_x, k * sizeof(float));
    cudaMalloc(&d_sums_y, k * sizeof(float));
    cudaMalloc(&d_counter, k * sizeof(int));
    cudaMalloc(&d_done, sizeof(int));

    int blockSize = 1024;
    int gridSize = (vecSize - 1) / blockSize + 1;
    int sharedSize = 2 * k * sizeof(float);

    auto start = chrono::high_resolution_clock::now();

    int iter;
    for (iter = 0; iter < iters; iter++) {
        // Clear sum and counter array to prepare for new iteration
        cudaMemset(d_sums_x, 0, k * sizeof(int));
        cudaMemset(d_sums_y, 0, k * sizeof(int));
        cudaMemset(d_counter, 0, k * sizeof(int));
        cudaMemset(d_done, 0, sizeof(int));

        // For each point, find nearest cluster and add itself to the cluster's sum
        setClusters << <gridSize, blockSize, sharedSize >> > (d_x, d_y, d_cur_cluster, d_means_x, d_means_y, d_sums_x, d_sums_y, vecSize, k, d_counter, d_done);

        // Get new mean of each cluster
        getNewMeans << <1, k >> > (d_means_x, d_means_y, d_sums_x, d_sums_y, d_counter);

        // Check if done became false(1), if so continue
        cudaMemcpy(h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_done[0] == 0)
            break;
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "Clustering completed in iteration : " << iter << endl << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

    cudaMemcpy(h_means_x.data(), d_means_x, k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_means_y.data(), d_means_y, k * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < k; i++) {
        cout << "Centroid in cluster " << i << " : ";
        cout << h_means_x[i] << " " << h_means_y[i];
        cout << endl;
    }
}