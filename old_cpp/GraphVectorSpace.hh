#ifndef GRAPHVECTORSPACE_HH
#define GRAPHVECTORSPACE_HH

#include <vector>
#include <string>
#include "mygraphs.hh"

using namespace std;

class GraphVectorSpace {

    public:
        virtual string get_basis_file_path();
        virtual bool is_valid();

        // Build the basis of the vector space.
        // If the vector space is not valid, or the basis file exists and ignore_existing_files is false, skip.
        void build_basis(bool ignore_existing_files = false) {
            if (!is_valid()) {
                return;
            }
            if (!ignore_existing_files && exists_basis_file()) {
                return;
            }

            string desc = "Build basis: " + get_ordered_param_dict();
            auto generating_list = get_generating_graphs();
            cout << desc << endl;
            set<string> basis_set;

            for (auto& G : generating_list) {
                vector<Automorphism> autom_list;
                Graph canonG;
                if (get_partition().empty()) {
                    autom_list = G.automorphism_group().gens();
                    canonG = G.canonical_label(Parameters::canonical_label_algorithm);
                } else {
                    autom_list = G.automorphism_group(get_partition()).gens();
                    canonG = G.canonical_label(get_partition(), Parameters::canonical_label_algorithm);
                }
                string canon6 = canonG.graph6_string();
                if (basis_set.find(canon6) == basis_set.end()) {
                    if (!_has_odd_automorphisms(G, autom_list)) {
                        basis_set.insert(canon6);
                    }
                }
            }
            vector<string> L(basis_set.begin(), basis_set.end());
            sort(L.begin(), L.end());
            _store_basis_g6(L);
        }

    protected:
        // Return whether the graph G has odd automorphisms.
        bool _has_odd_automorphisms(const Graph& G, const vector<Automorphism>& autom_list) {
            for (const auto& p : autom_list) {
                auto pd = p.dict();
                vector<int> pp(G.order());
                for (int j = 0; j < G.order(); ++j) {
                    pp[j] = pd[j];
                }
                if (perm_sign(G, pp) == -1) {
                    return true;
                }
            }
            return false;
        }

        // Return whether there exists a basis file.
        bool exists_basis_file() {
            ifstream f(get_basis_file_path());
            return f.good();
        }

        // Return the dimension of the vector space.
        int get_dimension() {
            if (!is_valid()) {
                return 0;
            }
            try {
                string header = StoreLoad::load_line(get_basis_file_path());
                return stoi(header);
            } catch (const StoreLoad::FileNotFoundError&) {
                throw StoreLoad::FileNotFoundError("Dimension unknown: No basis file");
            }
        }

        // Store the basis to the basis file.
        void _store_basis_g6(vector<string> basis_list) {
            basis_list.insert(basis_list.begin(), to_string(basis_list.size()));
            StoreLoad::store_string_list(basis_list, get_basis_file_path());
        }

        // Load the basis from the basis file.
        vector<string> _load_basis_g6() {
            if (!exists_basis_file()) {
                throw StoreLoad::FileNotFoundError("Cannot load basis, No basis file found");
            }
            vector<string> basis_list = StoreLoad::load_string_list(get_basis_file_path());
            int dim = stoi(basis_list.front());
            basis_list.erase(basis_list.begin());
            if (basis_list.size() != dim) {
                throw std::runtime_error("Basis read from file has wrong dimension");
            }
            return basis_list;
        }

        // Return the basis of the vector space as list of graph6 strings.
        vector<string> get_basis_g6() {
            if (!is_valid()) {
                // Return empty list if graph vector space is not valid.
                cerr << "Empty basis: not valid" << endl;
                return {};
            }
            return _load_basis_g6();
        }

        // Return the basis of the vector space as list of Graph objects.
        vector<Graph> get_basis() {
            vector<Graph> result;
            for (const auto& g6 : get_basis_g6()) {
                result.push_back(Graph(g6));
            }
            return result;
        }

        // The following methods must be implemented in derived classes:
        virtual vector<Graph> get_generating_graphs() = 0;
        virtual string get_ordered_param_dict() = 0;
        virtual vector<vector<int>> get_partition() = 0;
        virtual int perm_sign(const Graph& G, const vector<int>& perm) = 0;

}




#endif