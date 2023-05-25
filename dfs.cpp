/*
Design and implement Parallel Depth First Search based on existing algorithms using OpenMP.
Use a Tree or an undirected graph for DFS.
*/

#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>

void parallelDFS(std::vector<std::vector<int>>& graph, int startVertex) {
    std::stack<int> s;
    std::vector<bool> visited(graph.size(), false);

    // Push the start vertex onto the stack and mark it as visited
    s.push(startVertex);
    visited[startVertex] = true;

    while (!s.empty()) {
        int currentVertex = s.top();
        s.pop();

        #pragma omp parallel for
        for (int i = 0; i < graph[currentVertex].size(); ++i) {
            int neighbor = graph[currentVertex][i];

            // Check if the neighbor is already visited
            // and if not, mark it as visited and push it onto the stack
            if (!visited[neighbor]) {
                #pragma omp critical
                {
                    visited[neighbor] = true;
                    s.push(neighbor);
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

    int startVertex = 0;  // Starting vertex for DFS

    parallelDFS(graph, startVertex);

    return 0;
}
