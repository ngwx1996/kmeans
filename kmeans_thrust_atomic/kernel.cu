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
                            int vecSize, int k, int dimensions,
                            thrust::device_ptr<int> counter, thrust::device_ptr<int> done) {
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
        atomicAdd(thrust::raw_pointer_cast(sums + bestCluster + k * i), pointVec[idx + vecSize * i]);
    }
    atomicAdd(thrust::raw_pointer_cast(counter + bestCluster), 1);
}

__global__ void getNewMeans(thrust::device_ptr<float> means, thrust::device_ptr<float> sums, 
                            thrust::device_ptr<int> counter, int k, int dimensions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = (counter[idx] != 0) ? counter[idx] : 1;
    for (int i = 0; i < dimensions; i++) {
        means[idx + k * i] = sums[idx + k * i] / count;
    }
}

int main(int argc, char* argv[]) {
    cout << "---Thrust Atomic---" << endl;
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
    cout << "Fetched data successfully from " << filename << endl;

    thrust::device_vector<float> d_pointVec = h_pointVec;

    // each dimension has k size
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

    thrust::device_vector<float> d_means = h_means;
    thrust::device_vector<int> d_cur_cluster(vecSize, INT_MAX);
    thrust::device_vector<float> d_sums(k * dimensions);
    thrust::device_vector<int> d_counter(k,0);
    thrust::device_vector<int> d_done(1, 0);

    int blockSize = 1024;
    int gridSize = (vecSize - 1) / blockSize + 1;
    int sharedSize = dimensions * k * sizeof(float);

    auto start = chrono::high_resolution_clock::now();

    int iter;
    for (iter = 0; iter < iters; iter++) {
        // Clear sum and counter array to prepare for new iteration
        thrust::fill(d_sums.begin(), d_sums.end(), 0);
        thrust::fill(d_counter.begin(), d_counter.end(), 0);
        thrust::fill(d_done.begin(), d_done.end(), 0);

        setClusters << <gridSize, blockSize, sharedSize >> > (d_pointVec.data(), d_cur_cluster.data(), d_means.data(),
                                                    d_sums.data(), vecSize, k, dimensions, d_counter.data(), d_done.data());

        getNewMeans << <1, k >> > (d_means.data(), d_sums.data(), d_counter.data(), k, dimensions);

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