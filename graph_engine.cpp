/*
 * graph_engine.cpp
 *
 * High-performance graph engine implemented in modern C++ utilizing the
 * Standard Template Library (STL) to manage graph topology and shortest paths.
 *
 * STL components utilized:
 * - std::vector for dense adjacency matrix, adjacency list, and dynamic storage
 * - std::priority_queue for Dijkstra's shortest path algorithm (min-heap)
 * - std::string, std::ifstream, std::ofstream for formatted I/O
 *
 * Reads: eeg_adjacency_matrix.csv (64x64)
 * Exports: topological_metrics.csv
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <string>
#include <iomanip>
#include <limits>
#include <cmath>

// Directed weighted edge representation
struct Edge {
    int to;
    double weight;
};

// Topological metrics for each node
struct NodeMetrics {
    int id;
    int degree;
    double degree_centrality;
    double weighted_degree;
    double avg_shortest_path;
    double closeness_centrality;
    int reachable_nodes;
};

// Graph class managing topology and algorithms using STL
class GraphEngine {
public:
    static constexpr int NUM_NODES = 64;
    static constexpr double INF = std::numeric_limits<double>::infinity();

    GraphEngine(double threshold = 0.5)
        : threshold_(threshold),
          dense_matrix_(NUM_NODES, std::vector<double>(NUM_NODES, 0.0)),
          adj_list_(NUM_NODES) {}

    // Load 64x64 dense correlation matrix from CSV into std::vector<std::vector<double>>
    bool loadDenseMatrix(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file '" << filepath << "'\n";
            return false;
        }

        std::string line;
        int row = 0;
        while (row < NUM_NODES && std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string cell;
            int col = 0;

            while (col < NUM_NODES && std::getline(ss, cell, ',')) {
                // Trim leading and trailing whitespace
                size_t start = cell.find_first_not_of(" \t\r\n");
                size_t end = cell.find_last_not_of(" \t\r\n");
                if (start != std::string::npos) {
                    dense_matrix_[row][col] = std::stod(cell.substr(start, end - start + 1));
                }
                col++;
            }
            row++;
        }

        return (row == NUM_NODES);
    }

    // Apply numerical threshold to create weighted directed graph
    void buildWeightedGraph() {
        adj_list_.assign(NUM_NODES, std::vector<Edge>());
        total_edges_ = 0;

        for (int i = 0; i < NUM_NODES; ++i) {
            for (int j = 0; j < NUM_NODES; ++j) {
                // Avoid self-loops and filter by correlation threshold
                if (i != j && dense_matrix_[i][j] >= threshold_) {
                    adj_list_[i].push_back({j, dense_matrix_[i][j]});
                    total_edges_++;
                }
            }
        }
    }

    // Dijkstra's algorithm implemented using std::priority_queue (min-priority queue)
    std::vector<double> dijkstra(int startNode) const {
        std::vector<double> dist(NUM_NODES, INF);
        dist[startNode] = 0.0;

        // Min-heap storing pair<distance, node_id>
        using QueueElement = std::pair<double, int>;
        std::priority_queue<QueueElement,
                            std::vector<QueueElement>,
                            std::greater<QueueElement>> pq;

        pq.push({0.0, startNode});

        while (!pq.empty()) {
            QueueElement top = pq.top();
            pq.pop();

            double current_dist = top.first;
            int u = top.second;

            if (current_dist > dist[u]) {
                continue;
            }

            for (const auto& edge : adj_list_[u]) {
                int v = edge.to;
                double weight = edge.weight;

                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.push({dist[v], v});
                }
            }
        }

        return dist;
    }

    // Compute topological metrics for all nodes
    std::vector<NodeMetrics> computeMetrics() const {
        std::vector<NodeMetrics> metrics(NUM_NODES);
        double denom = static_cast<double>(NUM_NODES - 1);

        for (int i = 0; i < NUM_NODES; ++i) {
            metrics[i].id = i;
            metrics[i].degree = static_cast<int>(adj_list_[i].size());
            metrics[i].degree_centrality = (denom > 0.0) ? (metrics[i].degree / denom) : 0.0;

            double strength = 0.0;
            for (const auto& edge : adj_list_[i]) {
                strength += edge.weight;
            }
            metrics[i].weighted_degree = strength;

            // Shortest paths via Dijkstra
            std::vector<double> dist = dijkstra(i);
            double sum_dist = 0.0;
            int reachable = 0;

            for (int j = 0; j < NUM_NODES; ++j) {
                if (i != j && !std::isinf(dist[j])) {
                    sum_dist += dist[j];
                    reachable++;
                }
            }

            metrics[i].reachable_nodes = reachable;
            if (reachable > 0) {
                metrics[i].avg_shortest_path = sum_dist / static_cast<double>(reachable);
                metrics[i].closeness_centrality = static_cast<double>(reachable) / sum_dist;
            } else {
                metrics[i].avg_shortest_path = 0.0;
                metrics[i].closeness_centrality = 0.0;
            }
        }

        return metrics;
    }

    // Export topological metrics to CSV
    static bool exportToCSV(const std::string& filepath, const std::vector<NodeMetrics>& metrics) {
        std::ofstream out(filepath);
        if (!out.is_open()) {
            std::cerr << "Error: Could not open output file '" << filepath << "' for writing\n";
            return false;
        }

        out << "node_id,degree,degree_centrality,weighted_degree,avg_shortest_path,closeness_centrality,reachable_nodes\n";
        out << std::fixed << std::setprecision(6);

        for (const auto& m : metrics) {
            out << m.id << ","
                << m.degree << ","
                << m.degree_centrality << ","
                << m.weighted_degree << ","
                << m.avg_shortest_path << ","
                << m.closeness_centrality << ","
                << m.reachable_nodes << "\n";
        }

        return true;
    }

    int getTotalEdges() const { return total_edges_; }
    double getThreshold() const { return threshold_; }

private:
    double threshold_;
    int total_edges_ = 0;
    std::vector<std::vector<double>> dense_matrix_;
    std::vector<std::vector<Edge>> adj_list_;
};

int main(int argc, char* argv[]) {
    double threshold = 0.5;
    std::string input_file = "eeg_adjacency_matrix.csv";
    std::string output_file = "topological_metrics.csv";

    // CLI Arguments: [threshold] [input_file] [output_file]
    if (argc > 1) {
        threshold = std::stod(argv[1]);
    }
    if (argc > 2) {
        input_file = argv[2];
    }
    if (argc > 3) {
        output_file = argv[3];
    }

    std::cout << "=========================================\n";
    std::cout << "       NeuroGraph Engine (C++ STL)       \n";
    std::cout << "  (std::vector & std::priority_queue)    \n";
    std::cout << "=========================================\n";
    std::cout << "Parameters:\n";
    std::cout << "  Nodes:      " << GraphEngine::NUM_NODES << "\n";
    std::cout << "  Threshold:  " << std::fixed << std::setprecision(4) << threshold << "\n";
    std::cout << "  Input CSV:  " << input_file << "\n";
    std::cout << "  Output CSV: " << output_file << "\n\n";

    GraphEngine engine(threshold);

    // 1. Load dense matrix into std::vector
    std::cout << "Loading dense matrix from '" << input_file << "'...\n";
    if (!engine.loadDenseMatrix(input_file)) {
        return 1;
    }
    std::cout << "Dense 64x64 adjacency matrix populated with std::vector.\n";

    // 2. Build weighted directed graph
    engine.buildWeightedGraph();
    std::cout << "Created weighted directed graph with " << engine.getTotalEdges()
              << " edges (threshold >= " << threshold << ").\n";

    // 3. Compute metrics via Dijkstra's algorithm using std::priority_queue
    std::cout << "Running Dijkstra's algorithm using std::priority_queue...\n";
    std::vector<NodeMetrics> metrics = engine.computeMetrics();
    std::cout << "Topological metrics computed successfully for all nodes.\n";

    // 4. Export metrics to CSV
    std::cout << "Exporting topological metrics to '" << output_file << "'...\n";
    if (!GraphEngine::exportToCSV(output_file, metrics)) {
        return 1;
    }
    std::cout << "Topological metrics exported successfully to '" << output_file << "'.\n";

    return 0;
}
