#ifndef MYGRAPHS_HH
#define MYGRAPHS_HH

#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include "bliss/graph.hh"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <set>
#include <utility>



#include <random>
#include <cassert>
#include "indicators.hpp"
#include <optional>

using namespace std;


template <typename T>
int permutation_sign(const std::vector<T>& p) {
    int sign = 1;
    for (size_t i = 0; i < p.size(); ++i) {
        for (size_t j = i + 1; j < p.size(); ++j) {
            if (p[i] > p[j]) sign *= -1;
        }
    }
    return sign;
}

template <typename T>
vector<T> inverse_permutation(const std::vector<T>& p) {
    vector<T> inv(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        inv[p[i]] = i;
    }
    return inv;
}

template <typename T>
void print_perm(const std::vector<T>& p) {
    for (size_t i = 0; i < p.size(); ++i) {
        std::cout << (int)p[i] << " ";
    }
    std::cout << "\n";
}

// Permute the pair (u, v) to the left of the vertex_range and return the induced permutation (inverse)
inline std::vector<uint8_t> permute_to_left(uint8_t u, uint8_t v, size_t n) {
    std::vector<uint8_t> p(n);
    p[0] = u;
    p[1] = v;
    size_t idx = 2;
    for (uint8_t j = 0; j < n; ++j) {
        if (j == u || j == v) continue;
        p[idx++] = j;
    }
    // Return the inverse permutation
    return inverse_permutation(p);
}

struct Edge {
    uint8_t u, v;
    int data = 0;
    Edge(uint8_t u_, uint8_t v_, int data_ = 0) : u(u_), v(v_), data(data_) {}
    bool operator<(const Edge& other) const {
        return std::tie(u, v) < std::tie(other.u, other.v);
    }
    bool operator==(const Edge& other) const {
        return u == other.u && v == other.v && data == other.data;
    }
    friend bool operator>(const Edge& lhs, const Edge& rhs) {
        return rhs < lhs;
    }
    friend bool operator<=(const Edge& lhs, const Edge& rhs) {
        return !(rhs < lhs);
    }
    friend bool operator>=(const Edge& lhs, const Edge& rhs) {
        return !(lhs < rhs);
    }
};

class Graph {
public:
    uint8_t num_vertices;
    std::vector<Edge> edges;

    Graph(uint8_t n) : num_vertices(n) {}

    Graph(uint8_t n, const std::vector<Edge>& e)
        : num_vertices(n), edges(e) {}

    void add_edge(uint8_t u, uint8_t v, int data = 0) {
        if (u < v)
            edges.emplace_back(u, v, data);
        else
            edges.emplace_back(v, u, data);
    }

    std::string to_g6() const {
        uint8_t n = num_vertices;
        if (n > 62) throw std::runtime_error("Only supports graphs with at most 62 vertices.");
        std::string result;
        result.push_back(static_cast<char>(n + 63));
        std::vector<uint8_t> bitvec;
        for (uint8_t j = 1; j < n; ++j) {
            for (uint8_t i = 0; i < j; ++i) {
                bool found = false;
                for (const auto& e : edges) {
                    if ((e.u == i && e.v == j) || (e.u == j && e.v == i)) {
                        found = true;
                        break;
                    }
                }
                bitvec.push_back(found ? 1 : 0);
            }
        }
        while (bitvec.size() % 6 != 0) bitvec.push_back(0);
        for (size_t k = 0; k < bitvec.size(); k += 6) {
            uint8_t value = 0;
            for (size_t i = 0; i < 6; ++i) {
                value |= (bitvec[k + i] << (5 - i));
            }
            result.push_back(static_cast<char>(value + 63));
        }
        return result;
    }

    string to_canon_g6() const {
        // use bliss to get the canonical labeling of the graph and return its g6
        bliss::Graph blissG = to_bliss_graph();
        bliss::Stats stats;
        const unsigned int* perm = blissG.canonical_form(stats);
        std::vector<uint8_t> new_labels(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) {
            new_labels[i] = perm[i];
        }
        Graph canonG = Graph(num_vertices, edges);
        canonG.relabel(new_labels);
        return canonG.to_g6();
    }

    std::pair<string, int> to_canon_g6_sgn(bool even_edges) const {
        // use bliss to get the canonical labeling of the graph and return its g6
        bliss::Graph blissG = to_bliss_graph();
        bliss::Stats stats;
        const unsigned int* perm = blissG.canonical_form(stats);
        std::vector<uint8_t> new_labels(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) {
            new_labels[i] = perm[i];
        }
        int sign = perm_sign(new_labels, even_edges);
        Graph canonG = Graph(num_vertices, edges);
        canonG.relabel(new_labels);
        return {canonG.to_g6(), sign};
    }

    bool has_odd_automorphism(bool even_edges) const {
        // cout << "bliss0 "<< to_g6() << endl;
        bliss::Graph blissG = to_bliss_graph();
        bool ret = false;

        //std::vector<std::vector<unsigned>> generators;
        auto callback = [&](unsigned n, const unsigned* perm) {
            vector<uint8_t> p(n);
            for (size_t i = 0; i < n; ++i) {
                p[i] = perm[i];
            }
            //p = inverse_permutation(p);
            if (perm_sign(p, even_edges) != 1) {
                ret = true;
                // cout << "Permutation: ";
                // print_perm(p);
            }
            // ret = ret || (perm_sign(p, even_edges) != 1);
            //generators.emplace_back(perm, perm + n);
        };
        // cout << "bliss "<< to_g6() << endl;
        bliss::Stats stats;
        blissG.find_automorphisms(stats, callback);  // Modern Bliss: expects std::function

        // for (const auto& perm : generators) {
        //     for (auto x : perm) std::cout << x << " ";
        //     std::cout << "\n";
        // }
        return ret;
    }


    Graph add_edge_across(size_t e1idx, size_t e2idx) const {
        if (e1idx == e2idx) throw std::invalid_argument("Edges must be distinct");
        uint8_t new_n = num_vertices + 2;
        uint8_t v1 = num_vertices;
        uint8_t v2 = num_vertices + 1;
        std::vector<Edge> new_edges;
        for (size_t i = 0; i < edges.size(); ++i) {
            auto [u, v, data] = edges[i];
            if (i == e1idx) {
                new_edges.emplace_back(u, v1, data);
                new_edges.emplace_back(v, v1, data);
                new_edges.emplace_back(v1, v2, 0);
            } else if (i == e2idx) {
                new_edges.emplace_back(u, v2, data);
                new_edges.emplace_back(v, v2, data);
            } else {
                new_edges.emplace_back(u, v, data);
            }
        }
        std::sort(new_edges.begin(), new_edges.end());
        return Graph(new_n, new_edges);
    }

    Graph replace_edge_by_tetra(size_t eidx) const {
        uint8_t new_n = num_vertices + 4;
        auto [u, v, data] = edges[eidx];
        if (!(u < v)) throw std::invalid_argument("Edge must be (u < v)");
        std::vector<Edge> new_edges;
        for (size_t i = 0; i < edges.size(); ++i) {
            auto [a, b, d] = edges[i];
            if (i == eidx) {
                uint8_t v1 = new_n - 4;
                uint8_t v2 = new_n - 3;
                uint8_t v3 = new_n - 2;
                uint8_t v4 = new_n - 1;
                new_edges.emplace_back(u, v1, data);
                new_edges.emplace_back(v1, v2, 0);
                new_edges.emplace_back(v1, v3, 0);
                new_edges.emplace_back(v2, v4, 0);
                new_edges.emplace_back(v3, v4, 0);
                new_edges.emplace_back(v2, v3, 0);
                new_edges.emplace_back(v, v4, data);
            } else {
                new_edges.emplace_back(a, b, d);
            }
        }
        std::sort(new_edges.begin(), new_edges.end());
        return Graph(new_n, new_edges);
    }

    Graph union_with(const Graph& other) const {
        uint8_t new_n = num_vertices + other.num_vertices;
        std::vector<Edge> new_edges = edges;
        for (const auto& e : other.edges) {
            new_edges.emplace_back(e.u + num_vertices, e.v + num_vertices, e.data);
        }
        return Graph(new_n, new_edges);
    }

    Graph contract_edge(size_t eidx) const {
        if (num_vertices < 2) throw std::invalid_argument("Not enough vertices to contract");
        uint8_t new_n = num_vertices - 1;
        auto [u, v, data] = edges[eidx];
        if (!(u < v)) throw std::invalid_argument("Edge must be (u < v)");
        std::set<Edge> edge_set;
        for (size_t i = 0; i < edges.size(); ++i) {
            if (i == eidx) continue;
            auto [a, b, d] = edges[i];
            uint8_t aa = (a < v) ? a : (a == v ? u : a - 1);
            uint8_t bb = (b < v) ? b : (b == v ? u : b - 1);
            if (aa < bb)
                edge_set.emplace(aa, bb, d);
            else if (bb < aa)
                edge_set.emplace(bb, aa, d);
        }
        std::vector<Edge> new_edges(edge_set.begin(), edge_set.end());
        return Graph(new_n, new_edges);
    }

    set<u_int8_t> get_neighbors(uint8_t v) const {
        set<uint8_t> neighbors;
        for (const auto& e : edges) {
            if (e.u == v) {
                neighbors.insert(e.v);
            } else if (e.v == v) {
                neighbors.insert(e.u);
            }
        }
        return neighbors;
    }

    vector<Graph> get_all_splits() const {
        // Generate all possible vertex splits
        vector<Graph> result;
        for (uint8_t v = 0; v < num_vertices; ++v) {
            vector<Graph> splits = vertex_splits(v);
            result.insert(result.end(), splits.begin(), splits.end());
        }
        return result;
    }

    vector<Graph> vertex_splits(uint8_t v) const {
        auto nb = get_neighbors(v);
        if (nb.size() < 4) return {}; // cannot split (into trivalent vertices) if less than 4 neighbors
        vector<Graph> result;
        // iterate over subsets of nb of size >=2
        size_t n = nb.size();
        vector<uint8_t> neighbors(nb.begin(), nb.end());
        for (size_t i = 0; i < (1 << n); ++i) {
            if (__builtin_popcount(i) < 2 || n-__builtin_popcount(i)<2) continue; // skip subsets with less than 2 vertices or a complement of less than 2 vertices
            vector<uint8_t> subset, complement;
            for (size_t j = 0; j < n; ++j) {
                if (i & (1 << j)) {
                    subset.push_back(neighbors[j]);
                } else {
                    complement.push_back(neighbors[j]);
                }
            }
            if (subset.size() < 2 ||  complement.size() < 2) continue; // skip subsets with less than 3 vertices
            Graph g_split = split_vertex(v, subset, complement);
            result.push_back(g_split);
        }
        return result;
    }

    Graph split_vertex(uint8_t v, const vector<uint8_t>& subset, const vector<uint8_t>& complement) const {
        if (subset.size() < 2) throw std::invalid_argument("Subset must have at least 2 vertices");
        uint8_t new_v = num_vertices; // one vertex is replaced by the subset
        std::vector<Edge> new_edges;
        // all other edges remain the same
        for (const auto& e : edges) {
            if (e.u != v && e.v != v) {
                new_edges.push_back(e); // keep edges not involving v
            } 
        }
        for (const auto& u : subset) {
            // if (u == v) continue; // skip the vertex itself
            if (u<v) {
                new_edges.emplace_back(u, v, 0); // add edges from v to each vertex in the subset
            } else if (u>v){
                new_edges.emplace_back(v, u, 0); // add edges from v to each vertex in the subset
            }
        }
        for (const auto& u : complement) {
            new_edges.emplace_back(u, new_v, 0); // add edges from v to each vertex in the complement
        }
        new_edges.emplace_back(v, new_v, 0); // add edge from new vertex to v

        return Graph(num_vertices+1, new_edges);
    }


    static Graph from_g6(const std::string& g6) {
        if (g6.empty()) throw std::invalid_argument("Empty g6 string");
        uint8_t first = static_cast<uint8_t>(g6[0]);
        if (first < 63) throw std::invalid_argument("Invalid graph6 string");
        uint8_t n = (first >= 63 && first <= 126) ? (first - 63) : throw std::invalid_argument("Only supports n ≤ 62");
        size_t num_bits = n * (n - 1) / 2;
        size_t num_bytes = (num_bits + 5) / 6;
        if (g6.size() < 1 + num_bytes) throw std::invalid_argument("g6 string too short");
        std::vector<uint8_t> bits;
        for (size_t i = 0; i < num_bytes; ++i) {
            uint8_t val = static_cast<uint8_t>(g6[1 + i]);
            if (val < 63) throw std::invalid_argument("Invalid graph6 data byte");
            val -= 63;
            for (int j = 5; j >= 0; --j) {
                bits.push_back((val >> j) & 1);
            }
        }
        bits.resize(num_bits);
        std::vector<Edge> edges;
        size_t k = 0;
        for (uint8_t j = 1; j < n; ++j) {
            for (uint8_t i = 0; i < j; ++i) {
                if (bits[k++] == 1) {
                    edges.emplace_back(i, j, 0);
                }
            }
        }
        return Graph(n, edges);
    }

    bool is_biconnected(std::optional<uint8_t> ignore_vertex = std::nullopt) const {
        if (num_vertices <= 2) return false;

        std::vector<std::vector<uint8_t>> adj(num_vertices);
        for (const auto& e : edges) {
            if (ignore_vertex && (e.u == *ignore_vertex || e.v == *ignore_vertex))
                continue;
            adj[e.u].push_back(e.v);
            adj[e.v].push_back(e.u);
        }

        std::vector<int> disc(num_vertices, -1); // discovery times of visited vertices
        std::vector<int> low(num_vertices, -1);  // earliest visited vertex reachable
        std::vector<int> parent(num_vertices, -1);
        int time = 0;
        bool has_articulation = false;

        auto start = [&]() -> std::optional<uint8_t> {
            for (uint8_t i = 0; i < num_vertices; ++i)
                if (!ignore_vertex || i != *ignore_vertex)
                    return i;
            return std::nullopt;
        }();

        if (!start) return false;

        std::function<void(uint8_t)> dfs = [&](uint8_t u) {
            disc[u] = low[u] = time++;
            int children = 0;

            for (uint8_t v : adj[u]) {
                if (ignore_vertex && v == *ignore_vertex) continue;
                if (disc[v] == -1) {
                    parent[v] = u;
                    children++;
                    dfs(v);
                    low[u] = std::min(low[u], low[v]);

                    if ((parent[u] != -1 && low[v] >= disc[u]) ||
                        (parent[u] == -1 && children > 1)) {
                        has_articulation = true;
                    }
                } else if (v != parent[u]) {
                    low[u] = std::min(low[u], disc[v]);
                }
            }
        };

        dfs(*start);

        for (uint8_t i = 0; i < num_vertices; ++i) {
            if (ignore_vertex && i == *ignore_vertex) continue;
            if (disc[i] == -1) return false;
        }

        return !has_articulation;
    } // is_biconnected

    bool is_triconnected() const {
        if (!is_biconnected() || num_vertices < 4)
            return false;

        for (uint8_t v = 0; v < num_vertices; ++v) {
            if (!is_biconnected(v))
                return false;
        }

        return true;
    }

    template <typename Iterable>
    static void save_to_file(const Iterable& g6_list, const std::string& filename) {
        std::ofstream file(filename);
        if (!file) throw std::runtime_error("Failed to open file for writing");
        file << std::distance(std::begin(g6_list), std::end(g6_list)) << "\n";
        for (const auto& g6 : g6_list) {
            file << g6 << "\n";
        }
    }
    // static void save_to_file(const std::vector<std::string>& g6_list, const std::string& filename) {
    //     std::ofstream file(filename);
    //     if (!file) throw std::runtime_error("Failed to open file for writing");
    //     file << g6_list.size() << "\n";
    //     for (const auto& g6 : g6_list) {
    //         file << g6 << "\n";
    //     }
    // }

    static std::vector<std::string> load_from_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Failed to open file for reading " + filename);
        size_t num_graphs;
        file >> num_graphs;
        std::string line;
        std::getline(file, line); // consume rest of first line
        std::vector<std::string> g6_list;
        while (std::getline(file, line)) {
            if (!line.empty())
                g6_list.push_back(line);
        }
        if (g6_list.size() != num_graphs)
            throw std::runtime_error("Number of graphs in file does not match the first line");
        return g6_list;
    }

    static std::vector<std::string> load_from_file_nohdr(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Failed to open file for reading " + filename);
        std::vector<std::string> g6_list;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty())
                g6_list.push_back(line);
        }
        return g6_list;
    }

    static Graph tetrahedron_graph() {
        return Graph(4, {Edge(0,1), Edge(0,2), Edge(0,3), Edge(1,2), Edge(1,3), Edge(2,3)});
    }

    static Graph tetrastring_graph(uint8_t n_blocks) {
        uint8_t n = 4 * n_blocks;
        std::vector<Edge> edges;
        for (uint8_t i = 0; i < n_blocks; ++i) {
            edges.emplace_back(4 * i, 4 * i + 1);
            edges.emplace_back(4 * i, 4 * i + 2);
            edges.emplace_back(4 * i + 1, 4 * i + 2);
            edges.emplace_back(4 * i + 1, 4 * i + 3);
            edges.emplace_back(4 * i + 2, 4 * i + 3);
            if (i == n_blocks - 1) {
                edges.emplace_back(0, 4 * i + 3);
            } else {
                edges.emplace_back(4 * i + 3, 4 * i + 4);
            }
        }
        return Graph(n, edges);
    }

    void print() const {
        std::cout << "Graph with " << (int)num_vertices << " vertices and " << edges.size()
                  << " edges. G6 code: " << to_g6() << ".\n";
        for (const auto& e : edges) {
            std::cout << (int)e.u << " " << (int)e.v << " " << e.data << "\n";
        }
    }

    bliss::Graph to_bliss_graph() const {
        bliss::Graph g(num_vertices);
        for (const auto& e : edges) {
            g.add_edge(e.u, e.v);
        }
        return g;
    }

    void relabel(const std::vector<uint8_t>& new_labels) {
        if (new_labels.size() != num_vertices) throw std::invalid_argument("Invalid relabeling vector size");
        for (auto& e : edges) {
            e.u = new_labels[e.u];
            e.v = new_labels[e.v];
            // swap u and v if u > v
            if (e.u > e.v) {
                std::swap(e.u, e.v);
            }
        }
        sort_edges();
    }

    void sort_edges() {
        std::sort(edges.begin(), edges.end());
    }

    void number_edges() { 
        // sort edges and assign data the position in the ordered list
        sort_edges();
        for (size_t i = 0; i < edges.size(); ++i) {
            edges[i].data = i;
        }
    }

    int perm_sign(const std::vector<uint8_t>& p, bool even_edges) const {
        if (even_edges) {
            // Sign of the vertex permutation
            int sign = permutation_sign(p);
            // Multiply by sign flips from edge orientation
            for (const auto& e : edges) {
                uint8_t u = e.u, v = e.v;
                if ((u < v && p[u] > p[v]) || (u > v && p[u] < p[v])) {
                    sign *= -1;
                }
            }
            return sign;
        } else {
            Graph G1 = Graph(num_vertices, edges);
            G1.number_edges();
            // Graph G2(G1.num_vertices, G1.edges);
            G1.relabel(p);
            G1.sort_edges();
            std::vector<int> perm;
            for (const auto& e : G1.edges) {
                perm.push_back(e.data);
            }
            int sign = permutation_sign(perm);
            // if (sign !=1) {
            //     cout << "Permutation (perm_sign): ";
            //     print_perm(perm);
            //     cout << "graph: " << G1.to_g6() << endl;
            //     G1.print_edges();
            //     cout << "graph: " << G2.to_g6() << endl;
            //     G2.print_edges();

            // }
            return sign;
        }
    }

    void print_edges() {
        for (const auto& e : edges) {
            std::cout << (int)e.u << " " << (int)e.v << " " << (int)e.data << "\n";
        }
    }
    // int contract_edge_with_sign(size_t eidx, bool even_edges) const {

    // }

    vector<pair<Graph, int>> get_contractions_with_sign(bool even_edges) const {
        vector<pair<Graph, int>> image;
        for (size_t i = 0; i < edges.size(); ++i) {
            // Contract edge i
            auto [u, v, data] = edges[i];
            // Create permutation that brings u,v to 0,1
            vector<uint8_t> pp(num_vertices);
            iota(pp.begin(), pp.end(), 0);
            if (u != 0) swap(pp[0], pp[u]);
            if (v == 0) swap(pp[1], pp[u]);
            else if (v != 1) swap(pp[1], pp[v]);

            pp = permute_to_left(u, v, num_vertices);
            // Compute sign
            int sgn = perm_sign(pp, even_edges);
            // Relabel and contract
            Graph G1 = *this;
            G1.relabel(pp);
            G1.number_edges();
            size_t prev_size = G1.edges.size();
            //try {
            G1 = G1.contract_edge(0);
            //} catch (...) {
            //continue;
            //}
            if (prev_size - G1.edges.size() != 1) continue;
            // Relabel to canonical order
            // vector<uint8_t> relab(G1.num_vertices);
            // iota(relab.begin(), relab.end(), 0);
            // G1.relabel(relab);
            if (!even_edges) {
                // Compute sign from edge permutation
                vector<int> p;
                G1.sort_edges();
                for (const auto& e : G1.edges) p.push_back(e.data-1); // the edge with label 1 was contracted
                sgn *= permutation_sign(p);
            } else {
                sgn *= -1;
            }
            image.emplace_back(G1, sgn);
        }
        return image;
        
        // vector<pair<Graph, int>> contractions;
        // for (size_t i = 0; i < edges.size(); ++i) {
        //     Graph g = contract_edge(i);
        //     int sign = g.perm_sign({0, 1, 2, 3}, even_edges);
        //     contractions.emplace_back(g, sign);
        // }
        // return contractions;
    }

    bool check_valid(size_t defect, string err_msg) const {
        // check whether all vertex indices are < num_vertices
        for (const auto& e : edges) {
            if (e.u >= num_vertices || e.v >= num_vertices) {
                std::cerr << err_msg << " Graph " << to_g6() << " has vertex index >= num_vertices\n";
                return false;
            }
        }
        // check whether all vertices are $\geq 3-valent
        for (size_t i = 0; i < num_vertices; ++i) {
            size_t degree = 0;
            for (const auto& e : edges) {
                if (e.u == i || e.v == i) degree++;
            }
            if (degree < 3) {
                std::cerr << err_msg << " Graph " << to_g6() << " Vertex " << i << " has degree " << degree << "\n";
                return false;
            }
        }
        // there are no multiple edges or self-edges
        for (auto [u,v,data] : edges) {
            if (u == v) {
                std::cerr << err_msg << " Graph " << to_g6() << " has self-edge " << (int) u << "\n";
                return false;
            }
            if (u > v) {
                std::cerr << err_msg << " Graph " << to_g6() << " has wrongly ordered edge " << (int) u << " " << (int) v << "\n";
                return false;
            }
            int cnt = 0;
            for (auto [u2,v2,data2] : edges) {
                if (u == u2 && v == v2) {
                    cnt++;
                    if (cnt > 1){
                        std::cerr << err_msg << " Graph " << to_g6() << " has multiple edges " << (int) u << " " << (int) v << "\n";
                        return false;
                    }
                }
            }
        }
        // check whether the graph is connected
        std::vector<bool> visited(num_vertices, false);
        std::vector<uint8_t> stack;
        stack.push_back(0);
        while (!stack.empty()) {
            uint8_t v = stack.back();
            stack.pop_back();
            if (visited[v]) continue;
            visited[v] = true;
            for (const auto& e : edges) {
                if (e.u == v && !visited[e.v]) {
                    stack.push_back(e.v);
                } else if (e.v == v && !visited[e.u]) {
                    stack.push_back(e.u);
                }
            }
        }
        for (size_t i = 0; i < num_vertices; ++i) {
            if (!visited[i]) {
                std::cerr << err_msg << " Graph " << to_g6() << " is not connected\n";
                return false;
            }
        }
        // check if defect = 2edges -3vertices
        if (defect +3*num_vertices != 2*edges.size()) {
            int true_defect = 2*edges.size() - 3*num_vertices;
            std::cerr << err_msg << " Graph " << to_g6() << " has defect " << true_defect << "(not "<<defect<< ")\n";
            return false;
        }
        return true;
    }

    static bool check_g6_valid(string g6, size_t defect, string err_msg) {
        Graph g = from_g6(g6);
        return g.check_valid(defect, err_msg);
    }
};




inline bool graphs_equal(const Graph& g1, const Graph& g2) {
    if (g1.num_vertices != g2.num_vertices) {
        return false;
    }
    auto e1 = g1.edges;
    auto e2 = g2.edges;
    std::sort(e1.begin(), e1.end());
    std::sort(e2.begin(), e2.end());
    return e1 == e2;
}


inline void test_random_graphs_g6_roundtrip() {
    std::mt19937 rng(std::random_device{}());
    for (int t = 0; t < 10; ++t) {
        uint8_t n = std::uniform_int_distribution<uint8_t>(1, 20)(rng);
        Graph g(n);
        for (uint8_t u = 0; u < n; ++u) {
            for (uint8_t v = u + 1; v < n; ++v) {
                if (std::bernoulli_distribution(0.5)(rng)) {
                    g.add_edge(u, v);
                }
            }
        }
        std::string g6 = g.to_g6();
        Graph g2 = Graph::from_g6(g6);
        std::string g6_2 = g2.to_g6();
        assert(g6 == g6_2 && "G6 roundtrip failed");
        assert(graphs_equal(g, g2) && "Graph roundtrip failed");
    }
}



#endif // MYGRAPHS_HH