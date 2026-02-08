#ifndef HELPERS_HH
#define HELPERS_HH

#include "mygraphs.hh"
#include <map>
#include <fstream>
#include <filesystem>


void ensure_folder_of_filename_exists(const string& filename) {
    size_t pos = filename.find_last_of("/\\");
    if (pos != string::npos) {
        string folder = filename.substr(0, pos);
        if (!std::filesystem::exists(folder)) {
            std::filesystem::create_directories(folder);
        }
    }
}


map<pair<size_t, size_t>, int> load_matrix_from_sms_file(const string& filename, int& nrows, int& ncols) {
    ifstream file(filename);
    if (!file) throw std::runtime_error("Failed to open file for reading " + filename);
    map<pair<size_t, size_t>, int> matrix;
    string dummy;
    file >> nrows >> ncols >> dummy;
    // read until 0 0 0
    while (true) {
        size_t row, col;
        int val;
        file >> row >> col >> val;
        if (row == 0 && col == 0 && val == 0) break;
        if (row == 0 || col == 0) {
            throw std::runtime_error("Invalid row or column index (namely 0) in SMS file: " + filename);
        }
        // sms file uses 1-based indexing
        matrix[{row - 1, col - 1}] = val;
    }
    file.close();
    return matrix;
}

void save_matrix_to_sms_file(const map<pair<size_t, size_t>, int>& matrix, int nrows, int ncols, const string& filename) {
    ensure_folder_of_filename_exists(filename);
    ofstream file(filename);
    if (!file) throw std::runtime_error("Failed to open file for writing");
    // first line is rows cols M
    file << nrows << " " << ncols << " " << matrix.size() << "\n";
    for (const auto& [key, value] : matrix) {
        if (value == 0) continue; // skip zero entries
        // sms file uses 1-based indexing
        file << key.first+1 << " " << key.second+1 << " " << value << "\n";
    }
    // last line is 0 0 0
    file << "0 0 0\n";
    // close file
    file.close();
}

map<string, size_t> make_basis_dict(const vector<string>& basis) {
    map<string, size_t> basis_map;
    for (size_t i = 0; i < basis.size(); ++i) {
        basis_map[basis[i]] = i;
    }
    return basis_map;
}

void test_matrix_vs_reference(string mat_file, string ref_file, string domain_basis_file, string target_basis_file, string domain_ref_file, string target_ref_file, bool even_edges) {
    // test if the matrix is correct
    cout << "Checking matrix correctness " << mat_file << " ..." << endl;  
    int nrows_ref, ncols_ref;
    map<pair<size_t, size_t>, int> ref_matrix = load_matrix_from_sms_file(ref_file, nrows_ref, ncols_ref);
    // load matrix from file
    int nrows, ncols;
    map<pair<size_t, size_t>, int> matrix = load_matrix_from_sms_file(mat_file, nrows, ncols);
    
    if (nrows != nrows_ref || ncols != ncols_ref) {
        cout << "Matrix dimensions are different: " << nrows << "x" << ncols << " vs " << nrows_ref << "x" << ncols_ref << endl;
        return;
    }
    // check if the number of entries are the same
    if (matrix.size() != ref_matrix.size()) {
        cout << "Matrix number of entries are different: " << matrix.size() << " vs " << ref_matrix.size() << endl;
        //return;
    }

    // before comparing entries, we have to account for possibly different basis orderings.
    // load the domain and tagert basis and the reference basis. canonize the reference basis and find the permutation
    // then correct the matrix indices (rows and columns) accorind to the row- and column permutations
    // get the domain and target basis
    vector<string> in_basis = Graph::load_from_file(domain_basis_file);
    vector<string> out_basis = Graph::load_from_file(target_basis_file);
    map<string, size_t> in_basis_map = make_basis_dict(in_basis);
    map<string, size_t> out_basis_map = make_basis_dict(out_basis);
    // get the reference basis
    vector<string> in_basis_ref = Graph::load_from_file(domain_ref_file);
    vector<string> out_basis_ref = Graph::load_from_file(target_ref_file);

    // canonize the reference basis
    vector<int> in_basis_ref_sgn(in_basis_ref.size());
    for (size_t i = 0; i < in_basis_ref.size(); ++i) {
        Graph g = Graph::from_g6(in_basis_ref[i]);
        auto [g1s, sgn] = g.to_canon_g6_sgn(even_edges);
        in_basis_ref[i] = g1s;
        in_basis_ref_sgn[i] = sgn;

        // sanity checks
        if (g.has_odd_automorphism(even_edges)) {
            cout << "Reference graph has odd automorphism: " << g.to_g6() << endl;
        }
    }
    // canonize the target basis
    vector<int> out_basis_ref_sgn(out_basis_ref.size());
    for (size_t i = 0; i < out_basis_ref.size(); ++i) {
        Graph g = Graph::from_g6(out_basis_ref[i]);
        auto [g1s, sgn] = g.to_canon_g6_sgn(even_edges);
        out_basis_ref[i] = g1s;
        out_basis_ref_sgn[i] = sgn;
        // sanity checks
        if (g.has_odd_automorphism(even_edges)) {
            cout << "Reference graph has odd automorphism: " << g.to_g6() << endl;
        }
    }
    // compute ref to normal basis permutations
    vector<size_t> in_perm(in_basis_ref.size());
    vector<size_t> out_perm(out_basis_ref.size());
    for (size_t i = 0; i < in_basis_ref.size(); ++i) {
        auto it = in_basis_map.find(in_basis_ref[i]);
        if (it != in_basis_map.end()) {
            in_perm[i] = it->second;
        } else {
            cout << "Error: " << in_basis_ref[i] << " not found in domain basis" << endl;
        }
    }
    for (size_t i = 0; i < out_basis_ref.size(); ++i) {
        auto it = out_basis_map.find(out_basis_ref[i]);
        if (it != out_basis_map.end()) {
            out_perm[i] = it->second;
        } else {
            cout << "Error: " << out_basis_ref[i] << " not found in target basis" << endl;
        }
    }
    // now we have the permutations, we can correct the matrix indices
    map<pair<size_t, size_t>, int> matrix2;
    for (const auto& [key, value] : ref_matrix) {
        size_t row = key.first;
        size_t col = key.second;
        // apply the permutations
        row = in_perm[row];
        col = out_perm[col];
        matrix2[{row, col}] = value * in_basis_ref_sgn[key.first] * out_basis_ref_sgn[key.second];
    }
    // check whether the entries are the same
    for (const auto& [key, value] : matrix) {
        if (matrix2.find(key) == matrix2.end()) {
            cout << "Entry " << key.first << " " << key.second << " not found in ref matrix" << endl;
            cout << "g6 code " << in_basis[key.first] << " -> " << out_basis[key.second] << " value: " << value << endl;
        } else if (matrix2[key] != value) {
            cout << "Entry " << key.first << " " << key.second << " differs: " << matrix2[key] << " vs " << value << endl;
            cout << "g6 code " << in_basis[key.first] << " -> " << out_basis[key.second] << " value: " << value << endl;
        }
    }
    cout << "Matrix check completed." << endl;

}

void test_basis_vs_reference(string basis_file, string ref_file, bool even_edges, bool check_automorphisms = true) {
    // test if the basis is correct
    cout << "Checking basis correctness "<< basis_file << "..." << endl;  
    vector<string> g6s = Graph::load_from_file(basis_file);
    vector<string> ref_g6s = Graph::load_from_file(ref_file);

    // need to re-canonize reference basis
    for (size_t i = 0; i < ref_g6s.size(); ++i) {
        Graph g = Graph::from_g6(ref_g6s[i]);
        auto g1s = g.to_canon_g6();
        ref_g6s[i] = g1s;
        // sanity checks
        if (check_automorphisms && g.has_odd_automorphism(even_edges)) {
            cout << "Reference graph has odd automorphism: " << g.to_g6() << endl;
        }

    }

    // check whether the entries are the same
    //std::sort(g6s.begin(), g6s.end());
    std::sort(ref_g6s.begin(), ref_g6s.end());
    std::set<string> g6s_set(g6s.begin(), g6s.end());
    std::set<string> ref_g6s_set(ref_g6s.begin(), ref_g6s.end());
    std::set<string> diff;
    std::set_difference(g6s_set.begin(), g6s_set.end(),
                        ref_g6s_set.begin(), ref_g6s_set.end(),
                        std::inserter(diff, diff.begin()));
    if (diff.size() > 0) {
        cout << "The following graphs are in the basis but not in the reference:" << endl;
        for (const auto& g6 : diff) {
            cout << g6 << endl;
        }
    } else {
        cout << "All graphs in the basis are in the reference" << endl;
    }
    // check whether the entries are the same
    diff.clear();
    std::set_difference(ref_g6s_set.begin(), ref_g6s_set.end(),
                        g6s_set.begin(), g6s_set.end(),
                        std::inserter(diff, diff.begin()));
    if (diff.size() > 0) {
        cout << "The following graphs are in the reference but not in the basis:" << endl;
        for (const auto& g6 : diff) {
            cout << g6 << endl;
        }
    } else {
        cout << "All graphs in the reference are in the basis" << endl;
    }
}


#endif