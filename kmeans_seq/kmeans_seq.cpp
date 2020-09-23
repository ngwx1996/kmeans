#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace std;

class Point {
    public:
        Point(int id, string line) {
            dimensions = 0;
            pointId = id;
            clusterId = INT32_MAX;
            stringstream is(line);
            float val;
            char eater;
            // Eater used to absorb comma
            while (is >> val) {
                values.push_back(val);
                is >> eater;
                dimensions++;
            }
        }

        int getDimensions() {
            return dimensions;
        }

        int getCluster() {
            return clusterId;
        }

        int getId() {
            return pointId;
        }

        void setCluster(int id) {
            clusterId = id;
        }

        float getVal(int pos) {
            return values[pos];
        }

    private:
        int pointId;
        int clusterId;
        int dimensions;
        vector<float> values;
};

class Cluster{
    public:
        Cluster(int id, Point &centerPt) {
            clusterId = id;
            for (int i = 0; i < centerPt.getDimensions(); i++) {
                centroid.push_back(centerPt.getVal(i));
            }
            addPoint(centerPt);
        }

        void addPoint(Point &pt) {
            pt.setCluster(this->clusterId);
            points.push_back(pt);
        }

        bool removePoint(int pointId) {
            for (int i = 0; i < points.size(); i++) {
                if (points[i].getId() == pointId) {
                    points.erase(points.begin() + i);
                    return true;
                }
            }
            return false;
        }

        int getClusterId() {
            return clusterId;
        }

        Point getPoint(int pos) {
            return points[pos];
        }

        int getSize() {
            return points.size();
        }
        
        float getCentroidByPos (int pos) {
            return centroid[pos];
        }

        void setCentroidByPos (int pos, double val) {
            centroid[pos] = val;
        }

    private:
        int clusterId;
        vector<float> centroid;
        vector<Point> points;
};

int getNearestCluster(vector<Cluster> &clusters, Point point, int k) {
    float dist;
    float minDist = FLT_MAX;
    int nearestClusterId = INT_MAX;
    int dimension = point.getDimensions();
    
    // Get distance from point to nearest centroid
    for(int i = 0; i < k; i++) {
        dist = 0.0;

        for(int j = 0; j < dimension; j++)
        {
            dist += pow(clusters[i].getCentroidByPos(j) - point.getVal(j), 2.0);
        }

        if(dist < minDist)
        {
            minDist = dist;
            nearestClusterId = clusters[i].getClusterId();
        }
    }
    return nearestClusterId;
}

int main(int argc, char* argv[]) {
    cout << "---Sequential---" << endl;
    int k = 5;
    int iters = 300;
    string filename = "test100000.csv";
    ifstream infile(filename.c_str());

    if (!infile.is_open()) {
        cout << "Error: Failed to open file" << endl;
        return 1;
    }

    vector<Point> pointVec;
    int id = 0;
    string line;

    // Create point for each line
    while (getline(infile, line)) {
        Point point(id, line);
        pointVec.push_back(point);
        id++;
    }
    infile.close();
    cout << "Fetched data successfully from " << filename << endl;

    int pointVecSize = pointVec.size();
    int dimension = pointVec[0].getDimensions();
    // Store points that are used in initializing clusters. Makes sure no repeat.
    vector<int> usedPoints;
    vector<Cluster> clusters;

    for (int i = 0; i < k; i++) {
        while (true) {
            int idx = rand() % pointVecSize;

            if (find(usedPoints.begin(), usedPoints.end(), idx) == usedPoints.end()) {
                usedPoints.push_back(idx);
                pointVec[idx].setCluster(i);
                Cluster cluster(i, pointVec[idx]);
                clusters.push_back(cluster);
                break;
            }
        }
    }
    cout << clusters.size() << " clusters initialized" << endl;

    cout << "Running K-means clustering" << endl;
    auto start = chrono::high_resolution_clock::now();

    int iter = 0;

    while (iter < iters) {
        // check is true if point alr at best cluster
        bool check = true;

        // For each point, find nearest cluster and update cluster with point if move to new cluster
        for (int i = 0; i < pointVecSize; i++) {
            int curClusterId = pointVec[i].getCluster();
            int nearestClusterId = getNearestCluster(clusters, pointVec[i], k);

            if (curClusterId != nearestClusterId) {
                if (curClusterId != INT32_MAX) {
                    clusters[curClusterId].removePoint(pointVec[i].getId());
                }
                clusters[nearestClusterId].addPoint(pointVec[i]);
                check = false;
            }
        }

        // For each cluster, get new centroid
        for (int i = 0; i < k; i++) {
            int clusterSize = clusters[i].getSize();

            for (int j = 0; j < dimension; j++) {
                float sum = 0.0;
                if(clusterSize > 0) {
                    for (int p = 0; p < clusterSize; p++)
                        sum += clusters[i].getPoint(p).getVal(j);
                    clusters[i].setCentroidByPos(j, sum / clusterSize);
                }
            }
        }

        // Break if no change in all clusters
        if(check) {
            break;
        } else {
        iter++;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "Clustering completed in iteration : " << iter << endl << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

     for (int i = 0; i < k; i++){
        cout<<"Centroid in cluster "<<clusters[i].getClusterId()<<" : ";
        for(int j = 0; j < dimension; j++){
            cout<<clusters[i].getCentroidByPos(j)<<" ";
        }
        cout<<endl;
     }
}