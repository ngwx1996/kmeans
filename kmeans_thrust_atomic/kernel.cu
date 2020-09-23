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
    extern __shared__ float shared_means[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vecSize)
        return;

    if (threadIdx.x < k) {
        shared_means[threadIdx.x] = means_x[threadIdx.x];
        shared_means[threadIdx.x + k] = means_y[threadIdx.x];
    }
    __syncthreads();


    float cur_x = x[idx];
    float cur_y = y[idx];
    float minDist = FLT_MAX;
    int bestCluster = INT_MAX;
    float distance;
    for (int i = 0; i < k; i++) {
        distance = (cur_x - shared_means[i]) * (cur_x - shared_means[i]) + (cur_y - shared_means[i + k]) * (cur_y - shared_means[i + k]);
        if (distance < minDist) {
            minDist = distance;
            bestCluster = i;
        }
    }
    if (cur_cluster[idx] != bestCluster) {
        cur_cluster[idx] = bestCluster;
        done[0] = 1;
    }

    atomicAdd(thrust::raw_pointer_cast(sums_x + bestCluster), cur_x);
    atomicAdd(thrust::raw_pointer_cast(sums_y + bestCluster), cur_y);
    atomicAdd(thrust::raw_pointer_cast(counter + bestCluster), 1);
}

__global__ void getNewMeans(thrust::device_ptr<float> means_x, thrust::device_ptr<float> means_y, 
                            thrust::device_ptr<float> sums_x, thrust::device_ptr<float> sums_y, 
                            thrust::device_ptr<int> counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = (counter[idx] != 0) ? counter[idx] : 1;

    means_x[idx] = sums_x[idx] / count;
    means_y[idx] = sums_y[idx] / count;
}

int main(int argc, char* argv[]) {
    cout << "---Thrust Atomic---" << endl;
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
    cout << "Fetched data successfully from " << filename << endl;

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

    thrust::device_vector<float> d_means_x = h_means_x;
    thrust::device_vector<float> d_means_y = h_means_y;
    thrust::device_vector<int> d_cur_cluster(vecSize, FLT_MAX);
    thrust::device_vector<float> d_sums_x(k);
    thrust::device_vector<float> d_sums_y(k);
    thrust::device_vector<int> d_counter(k,0);
    thrust::device_vector<int> d_done(1, 0);

    int blockSize = 1024;
    int gridSize = (vecSize - 1) / blockSize + 1;
    int sharedSize = 2 * k * sizeof(float);

    auto start = chrono::high_resolution_clock::now();

    int iter;
    for (iter = 0; iter < iters; iter++) {
        // Clear sum and counter array to prepare for new iteration
        thrust::fill(d_sums_x.begin(), d_sums_x.end(), 0);
        thrust::fill(d_sums_y.begin(), d_sums_y.end(), 0);
        thrust::fill(d_counter.begin(), d_counter.end(), 0);
        thrust::fill(d_done.begin(), d_done.end(), 0);

        setClusters << <gridSize, blockSize, sharedSize >> > (d_x.data(), d_y.data(), d_cur_cluster.data(), d_means_x.data(), d_means_y.data(), 
                                                    d_sums_x.data(), d_sums_y.data(), vecSize, k, d_counter.data(), d_done.data());

        getNewMeans << <1, k >> > (d_means_x.data(), d_means_y.data(), d_sums_x.data(), 
                                    d_sums_y.data(), d_counter.data());

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