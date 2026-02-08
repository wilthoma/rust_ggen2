#ifndef ORDINARYGC_HH
#define ORDINARYGC_HH

#include "mygraphs.hh"
#include "helpers.hh"
#include <vector>
#include <string>
#include "indicators.hpp"

string get_type_string(bool even_edges) {
    return even_edges ? "even_edges" : "odd_edges";
}

class OrdinaryGVS {
    public:
        uint8_t num_vertices;
        uint8_t num_loops;
        bool even_edges;
        bool use_triconnected;

        OrdinaryGVS(uint8_t nvertices, uint8_t loops, bool even, bool triconnected)
            : num_vertices(nvertices), num_loops(loops), even_edges(even), use_triconnected(triconnected) {}
        
        string get_basis_file_path(){
            if (use_triconnected) {
                return "data/ordinary/tri/" + get_type_string(even_edges) +
                    "/gra" + std::to_string(num_vertices) +
                    "_" + std::to_string(num_loops) + ".g6";
            } else {
                return "data/ordinary/" + get_type_string(even_edges) +
                    "/gra" + std::to_string(num_vertices) +
                    "_" + std::to_string(num_loops) + ".g6";
            }
        }
        string get_basis_ref_file_path(){
            if (use_triconnected) {
                return "data/ordinary/tri/ref/" + get_type_string(even_edges) +
                    "/gra_tri" + std::to_string(num_vertices) +
                    "_" + std::to_string(num_loops) + ".g6";
            } else {
                return "data/ordinary/ref/" + get_type_string(even_edges) +
                    "/gra" + std::to_string(num_vertices) +
                    "_" + std::to_string(num_loops) + ".g6";
            }
        }
        string get_input_file_path(){
            int defect = 2 * num_loops - 2 -  num_vertices;
            return "data/graphs" + std::to_string(num_loops) +
                   "_" + std::to_string(defect) + ".g6";
        }

        bool is_valid() const {
            // Vertices at least trivalent. Positive number of vertices. Non-negative number of loops.
            // At most fully connected graph, no multiple edges.
            int n_edges = num_loops + num_vertices - 1;
            return (3 * num_vertices <= 2 * n_edges) &&
                   (num_vertices > 0) &&
                   (n_edges <= num_vertices * (num_vertices - 1) / 2);
        }

        vector<string> get_basis_g6() {
            if (!is_valid()) {
                // Return empty list if graph vector space is not valid.
                //cerr << "Empty basis: not valid" << endl;
                return {};
            }
            return Graph::load_from_file(get_basis_file_path());
        }
        map<string, size_t> get_basis_dict() {
            return make_basis_dict(get_basis_g6());
        }

        void build_basis(bool ignore_existing_files = false) {
            if (!is_valid()) {
                // cerr << "Cannot build basis: not valid" << endl;
                return;
            }
            if (!ignore_existing_files && std::ifstream(get_basis_file_path())) {
                cout << "Basis file already exists: " << get_basis_file_path() << endl;
                return;
            }

            ensure_folder_of_filename_exists(get_basis_file_path());
            string infile = get_input_file_path();
            vector<string> in_g6s = Graph::load_from_file(infile);
            size_t num_graphs = in_g6s.size();
            vector<string> out_g6s;
            out_g6s.reserve(num_graphs);
            if (in_g6s.empty()) {
                cerr << "No graphs found in input file: " << infile << endl;
                return;
            }
            cout << "Building basis for " << get_basis_file_path() << endl;

            size_t progress = 0;
            indicators::ProgressBar bar {
                indicators::option::BarWidth{50},
                indicators::option::Start{"["},
                indicators::option::Fill{"="},
                indicators::option::Lead{">"},
                indicators::option::Remainder{" "},
                indicators::option::End{"]"},
                indicators::option::PostfixText{"Processing contractions"},
                indicators::option::ForegroundColor{indicators::Color::yellow},
                indicators::option::ShowElapsedTime{true},
                indicators::option::ShowRemainingTime{true},
                indicators::option::MaxProgress{num_graphs}
            };
            for (const auto& g6 : in_g6s) {
                Graph g = Graph::from_g6(g6);
                if ((use_triconnected ? g.is_triconnected() : g.is_biconnected()) && !g.has_odd_automorphism(even_edges)) {
                    string canon_g6 = g.to_canon_g6();
                    out_g6s.push_back(canon_g6);
                }
                ++progress;
                bar.set_progress(progress);
                    // Show iteration as postfix text
                    bar.set_option(indicators::option::PostfixText{
                    std::to_string(progress) + "/" + std::to_string(num_graphs)
                });
            }
            bar.mark_as_completed();

            cout << "Found " << out_g6s.size() << " graphs in the basis." << endl;
            cout << "Saving basis to file: " << get_basis_file_path() << endl;
            // save the basis to file
            Graph::save_to_file(out_g6s, get_basis_file_path());
            cout << "Done." << endl;

        }

        void test_basis_vs_ref() {
            if (!is_valid()) {
                cout << "Basis check void for " << to_string() << endl;
                return;
            }
            test_basis_vs_reference(get_basis_file_path(),
                                    get_basis_ref_file_path(), even_edges);
        }

        string to_string() const {
            return "OrdinaryGVS(" + std::to_string(num_vertices) + ", " +
                   std::to_string(num_loops) + ", " + get_type_string(even_edges) + ")";
        }
};

class OrdinaryContract {
    public:
        OrdinaryGVS domain;
        OrdinaryGVS target;
        uint8_t num_loops;
        uint8_t num_vertices;
        bool even_edges;
        bool use_triconnected;

        OrdinaryContract(uint8_t n_vertices, uint8_t n_loops, bool even_edges_, bool triconnected)
            : domain(n_vertices, n_loops, even_edges_, triconnected), 
              target(n_vertices-1, n_loops, even_edges_, triconnected), 
              num_loops(n_loops), num_vertices(n_vertices), even_edges(even_edges_), use_triconnected(triconnected) {}

        string get_matrix_file_path() {
            return std::string("data/ordinary/") + (use_triconnected ? "tri/" : "") + get_type_string(even_edges) +
                   "/contractD" + std::to_string(num_vertices) + "_" + std::to_string(num_loops) + ".txt";
        }
        string get_ref_matrix_file_path() {
            if (use_triconnected) {
                return "data/ordinary/tri/ref/" + get_type_string(even_edges) +
                       "/contractD_tri" + std::to_string(num_vertices) + "_" + std::to_string(num_loops) + ".txt";
            } else {
                return "data/ordinary/ref/" + get_type_string(even_edges) +
                    "/contractD" + std::to_string(num_vertices) + "_" + std::to_string(num_loops) + ".txt";
            }

        }

        bool is_valid() const {
            return domain.is_valid() && target.is_valid() &&
                   (domain.num_vertices == num_vertices) &&
                   (target.num_vertices == num_vertices - 1) &&
                   (domain.num_loops == num_loops) &&
                   (target.num_loops == num_loops);
        }

        void build_matrix(bool ignore_existing_files = false) {
            if (!is_valid()) {
                // cout << "Cannot build matrix: not valid" << endl;
                return;
            }

            if (!ignore_existing_files && std::ifstream(get_matrix_file_path())) {
                cout << "Matrix file already exists: " << get_matrix_file_path() << endl;
                return;
            }

            ensure_folder_of_filename_exists(get_matrix_file_path());
            vector<string> in_basis = domain.get_basis_g6();
            vector<string> out_basis = target.get_basis_g6();
            map<string, size_t> in_basis_map = make_basis_dict(in_basis);
            map<string, size_t> out_basis_map = make_basis_dict(out_basis);
            map<pair<size_t, size_t>, int> matrix;
            int num_rows = in_basis.size();
            int num_cols = out_basis.size();

            cout << "Number of input basis elements: " << num_rows << endl;
            cout << "Number of output basis elements: " << num_cols << endl;

            cout << "Contracting...." << endl;

            indicators::ProgressBar bar{
                indicators::option::BarWidth{50},
                indicators::option::Start{"["},
                indicators::option::Fill{"="},
                indicators::option::Lead{">"},
                indicators::option::Remainder{" "},
                indicators::option::End{"]"},
                indicators::option::PostfixText{"Processing contractions"},
                indicators::option::ForegroundColor{indicators::Color::yellow},
                indicators::option::ShowElapsedTime{true},
                indicators::option::ShowRemainingTime{true},
                indicators::option::MaxProgress{in_basis.size()}
            };

            size_t progress = 0;
            for (const string& s : in_basis) {
                Graph g = Graph::from_g6(s);
                auto contractions = g.get_contractions_with_sign(even_edges);
                for (const auto& [g1, sign] : contractions) {
                    auto [g1s, sign2] = g1.to_canon_g6_sgn(even_edges);
                    int val = sign * sign2;
                    if (out_basis_map.find(g1s) != out_basis_map.end()) {
                        size_t row = in_basis_map[s];
                        size_t col = out_basis_map[g1s];
                        if (matrix.find({row, col}) != matrix.end()) {
                            matrix[{row, col}] += val;
                        } else {
                            matrix[{row, col}] = val;
                        }
                    }
                }
                ++progress;
                bar.set_progress(progress);
                    // Show iteration as postfix text
                    bar.set_option(indicators::option::PostfixText{
                    std::to_string(progress) + "/" + std::to_string(in_basis.size())
                });
            }
            bar.mark_as_completed();
            cout << "Saving matrix to file: " << get_matrix_file_path() << endl;
            // save matrix to file
            save_matrix_to_sms_file(matrix, num_rows, num_cols, get_matrix_file_path());
            cout << "Done." << endl;
        }

        void test_matrix_vs_ref() {
            if (!is_valid()) {
                cout << "Matrix check void for " << to_string() << endl;
                return;
            }
            test_matrix_vs_reference(get_matrix_file_path(),
                                     get_ref_matrix_file_path(),
                                     domain.get_basis_file_path(),
                                     target.get_basis_file_path(),
                                     domain.get_basis_ref_file_path(),
                                     target.get_basis_ref_file_path(),
                                     even_edges);
        }

        string to_string() const {
            return "OrdinaryContract(" + std::to_string(num_vertices) + ", " +
                   std::to_string(num_loops) + ", " + get_type_string(even_edges) + ")";
        }
};






#endif // ORDINARYGC_HH