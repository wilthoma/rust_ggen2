#ifndef PREORDINARYGC_HH
#define PREORDINARYGC_HH

#include "mygraphs.hh"
#include "helpers.hh"
#include <vector>
#include <string>
#include "indicators.hpp"


class PreOrdinaryGVS {
    public:
        uint8_t defect;
        uint8_t num_loops;
        uint8_t num_vertices;

        PreOrdinaryGVS(uint8_t loops, uint8_t defect_)
            : num_vertices(2*loops-2-defect), num_loops(loops), defect(defect_) {}
        
        string get_basis_file_path(){
            return "data/pre/graphs" + std::to_string(num_loops) +
                   "_" + std::to_string(defect) + ".g6";
        }
        string get_basis_ref_file_path(){
            return "data/pre/ref/graphs" + std::to_string(num_loops) +
                   "_" + std::to_string(defect) + ".g6";
        }
        string get_input_file_path(){
            int defect = 2 * num_loops - 2 -  num_vertices;
            return "data/pre/graphs" + std::to_string(num_loops) +
                   "_" + std::to_string(defect-1) + ".g6";
        }

        vector<string> get_basis_g6() {
            // if (!is_valid()) {
            //     // Return empty list if graph vector space is not valid.
            //     cerr << "Empty basis: not valid" << endl;
            //     return {};
            // }
            return Graph::load_from_file(get_basis_file_path());
        }
        map<string, size_t> get_basis_dict() {
            return make_basis_dict(get_basis_g6());
        }

        void build_basis(bool ignore_existing_files = false) {
            if (!ignore_existing_files && std::ifstream(get_basis_file_path())) {
                cout << "Basis file already exists: " << get_basis_file_path() << endl;
                return;
            }

            ensure_folder_of_filename_exists(get_basis_file_path());
            string infile = get_input_file_path();
            vector<string> in_g6s = Graph::load_from_file(infile);
            size_t num_graphs = in_g6s.size();
            set<string> out_g6s;
            size_t nedges = 3* num_loops - 3 - defect;
            //out_g6s.reserve(num_graphs*(nedges+1));
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
                // g.check_valid(defect-1, "PreOrdinaryGVS::build_basis");
                for (int i = 0; i <= nedges; ++i) {
                    Graph g1 = g.contract_edge(i);
                    if (g1.edges.size() == nedges) {
                        // g1.check_valid(defect, "PreOrdinaryGVS::build_basis(2)");
                        string canon_g6 = g1.to_canon_g6();
                        out_g6s.insert(canon_g6);
                    }
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
            // std::vector<std::string> out_g6s_vec(out_g6s.begin(), out_g6s.end());
            Graph::save_to_file(out_g6s, get_basis_file_path());
            cout << "Done." << endl;

        }

        void test_basis_vs_ref() {
            test_basis_vs_reference(get_basis_file_path(),
                                    get_basis_ref_file_path(), false, false);
        }
};







#endif // PREORDINARYGC_HH