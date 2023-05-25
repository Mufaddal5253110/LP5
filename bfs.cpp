/*
1.	Design and implement Parallel Breadth First Search based on existing algorithms using OpenMP.
Use a Tree or an undirected graph for BFS. in C++
*/

#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

void parallelBFS(std::vector<std::vector<int>>& graph, int startVertex) {
    std::queue<int> q;
    std::vector<bool> visited(graph.size(), false);

    // Mark the start vertex as visited and enqueue it
    visited[startVertex] = true;
    q.push(startVertex);

    while (!q.empty()) {
        int currentVertex = q.front();
        q.pop();

        #pragma omp parallel for
        for (int i = 0; i < graph[currentVertex].size(); ++i) {
            int neighbor = graph[currentVertex][i];

            // Check if the neighbor is already visited
            // and if not, mark it as visited and enqueue it
            if (!visited[neighbor]) {
                #pragma omp critical
                {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
    }
}

int main() {
    int numVertices = 6;  // Number of vertices in the graph
    std::vector<std::vector<int>> graph(numVertices);

    // Add edges to the undirected graph
    graph[0].push_back(1);
    graph[0].push_back(2);
    graph[1].push_back(3);
    graph[1].push_back(4);
    graph[2].push_back(5);

    int startVertex = 0;  // Starting vertex for BFS

    parallelBFS(graph, startVertex);

    return 0;
}
