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

__global__ void setClusters(thrust::device_ptr<float> x, thrust::device_ptr<float> y,
                            thrust::device_ptr<int> cur_cluster,
                            thrust::device_ptr<float> means_x, thrust::device_ptr<float> means_y, 
                            thrust::device_ptr<float> sums_x, thrust::device_ptr<float> sums_y, 
                            int vecSize, int k, thrust::device_ptr<int> counter, thrust::device_ptr<int> done) {
    extern __shared__ float shared[];
    float* shared_x = shared;
    float* shared_y = &shared_x[blockDim.x];
    int* shared_count = (int*)&shared_y[blockDim.x];

    int localIdx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= vecSize)
        return;

    // Use shared mem to store means
    if (localIdx < k) {
        shared[localIdx] = means_x[localIdx];
        shared[localIdx + k] = means_y[localIdx];
    }
    __syncthreads();

    float minDist = FLT_MAX;
    int bestCluster = INT_MAX;
    float distance;
    float cur_x;
    float cur_y;

    cur_x = x[idx];
    cur_y = y[idx];

    for (int i = 0; i < k; i++) {
        distance = (cur_x - shared[i]) * (cur_x - shared[i]) + (cur_y - shared[i + k]) * (cur_y - shared[i + k]);
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
        shared_x[localIdx] = (bestCluster == j) ? cur_x : 0;
        shared_y[localIdx] = (bestCluster == j) ? cur_y : 0;
        shared_count[localIdx] = (bestCluster == j) ? 1 : 0;

        __syncthreads();

        // Reduction to get sum at tid 0
        for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
            if (localIdx < d && idx + d < vecSize) {
                shared_x[localIdx] += shared_x[localIdx + d];
                shared_y[localIdx] += shared_y[localIdx + d];
                shared_count[localIdx] += shared_count[localIdx + d];
            }
            __syncthreads();
        }

        // Value at tid 0
        if (localIdx == 0) {
            int clusterIdx = blockIdx.x * k + j;
            sums_x[clusterIdx] = shared_x[localIdx];
            sums_y[clusterIdx] = shared_y[localIdx];
            counter[clusterIdx] = shared_count[localIdx];
        }
        __syncthreads();
    }
}

__global__ void getNewMeans(thrust::device_ptr<float> means_x, thrust::device_ptr<float> means_y, 
                            thrust::device_ptr<float> sums_x, thrust::device_ptr<float> sums_y, 
                            thrust::device_ptr<int> counter, int k) {
    extern __shared__ float shared[];
    float* shared_x = shared;
    float* shared_y = &shared_x[blockDim.x];
    int* shared_count = (int*)&shared_y[blockDim.x];


    int idx = threadIdx.x;
    int blocks = blockDim.x / k;
    shared_x[idx] = sums_x[idx];
    shared_y[idx] = sums_y[idx];
    shared_count[idx] = counter[idx];

    __syncthreads();

    if (idx < k) {
        for (int j = 1; j < blocks; j++) {
            shared_x[idx] += shared_x[idx + j * k];
            shared_y[idx] += shared_y[idx + j * k];
            shared_count[idx] += shared_count[idx + j * k];
        }
        __syncthreads;
    }

    if (idx < k) {
        int count = (shared_count[idx] > 0) ? shared_count[idx] : 1;
        means_x[idx] = shared_x[idx] / count;
        means_y[idx] = shared_y[idx] / count;
        sums_x[idx] = 0;
        sums_y[idx] = 0;
        counter[idx] = 0;
    }
}

int main(int argc, char* argv[]) {
    cout << "---Thrust---" << endl;
    int k = 5;
    int iters = 300;
    string filename = "test100000.csv";
    ifstream infile(filename.c_str());
    string line;

    if (!infile.is_open()) {
        cout << "Error: Failed to open file" << endl;
        return 1;
    }

    thrust::host_vector<float> h_x;
    thrust::host_vector<float> h_y;
    float val;
    char eater;
    while (getline(infile, line)) {
        stringstream is(line);
        is >> val;
        h_x.push_back(val);
        is >> eater;
        is >> val;
        h_y.push_back(val);
    }
    infile.close();
    cout << "Fetched data successfully" << endl;

    int vecSize = h_x.size();
    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;

    thrust::host_vector<float> h_means_x;
    thrust::host_vector<float> h_means_y;

    for (int i = 0; i < k; i++) {
        while (true) {
            int idx = rand() % vecSize;
            thrust::host_vector<float>::iterator it = thrust::find(h_means_x.begin(), h_means_x.end(), h_x[idx]);
            if (it == h_means_x.end() || h_means_y[it - h_means_x.begin()] != h_y[idx]) {
                h_means_x.push_back(h_x[idx]);
                h_means_y.push_back(h_y[idx]);
                break;
            }
        }
    }

    cout << k << " clusters initialized" << endl;

    cout << "Running K-means clustering" << endl;

    int blockSize = 1024;
    int gridSize = (vecSize - 1) / blockSize + 1;
    int sharedSizeCluster = blockSize * (2 * sizeof(float) + sizeof(int));
    int sharedSizeMeans = k * gridSize * (2 * sizeof(float) + sizeof(int));

    thrust::device_vector<float> d_means_x = h_means_x;
    thrust::device_vector<float> d_means_y = h_means_y;
    thrust::device_vector<int> d_cur_cluster(vecSize, FLT_MAX);
    thrust::device_vector<float> d_sums_x(k * gridSize);
    thrust::device_vector<float> d_sums_y(k * gridSize);
    thrust::device_vector<int> d_counter(k * gridSize,0);
    thrust::device_vector<int> d_done(1, 0);

    auto start = chrono::high_resolution_clock::now();

    int iter;
    for (iter = 0; iter < iters; iter++) {
        // Clear done to prepare for new iteration
        thrust::fill(d_done.begin(), d_done.end(), 0);

        setClusters << <gridSize, blockSize, sharedSizeCluster >> > (d_x.data(), d_y.data(), d_cur_cluster.data(), d_means_x.data(), d_means_y.data(), 
                                                    d_sums_x.data(), d_sums_y.data(), vecSize, k, d_counter.data(), d_done.data());

        getNewMeans << <1, k * gridSize, sharedSizeMeans >> > (d_means_x.data(), d_means_y.data(), d_sums_x.data(), 
                                    d_sums_y.data(), d_counter.data(), k);

        if (d_done[0] == 0)
            break;
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "Clustering completed in iteration : " << iter << endl << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

    for (int i = 0; i < k; i++) {
        cout << "Centroid in cluster " << i << " : ";
        cout << d_means_x[i] << " " << d_means_y[i];
        cout << endl;
    }
}