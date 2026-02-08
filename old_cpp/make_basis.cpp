#include "mygraphs.hh"
#include <chrono>
#include <iostream>
#include "CLI11.hpp"
#include "indicators.hpp"
#include "OrdinaryGC.hh"

std::chrono::high_resolution_clock::time_point tic_time;

void tic() {
    tic_time = std::chrono::high_resolution_clock::now();
}

void toc() {
    auto toc_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(toc_time - tic_time).count();
    std::cout << "Elapsed time: " << duration << " ms" << std::endl;
}

struct Range {
    int start = 0;
    int end = 0;

    // This tells CLI11 how to convert a string like "1:20" to a Range
    friend std::istream& operator>>(std::istream& in, Range& r) {
        std::string s;
        in >> s;
        auto pos = s.find(':');
        if (pos == std::string::npos)
            throw CLI::ConversionError("Range format must be start:end");

        r.start = std::stoi(s.substr(0, pos));
        r.end = std::stoi(s.substr(pos + 1));
        return in;
    }
};

int main(int argc, char** argv) {
    CLI::App app{"Graph and matrix generator"};

    Range r_loops;
    Range r_vertices;
    bool compute_matrices = false;
    bool compute_bases = false;
    bool even_edges = false;
    bool overwrite = false;
    bool test_basis = false;
    bool test_matrix = false;
    bool triconnected = false;

    app.add_option("range_vertices", r_vertices, "Range in format start:end")->required();
    app.add_option("range_loops", r_loops, "Range in format start:end")->required();
    app.add_flag("-m,--compute-matrices", compute_matrices, "Compute matrices");
    app.add_flag("-b,--compute-bases", compute_bases, "Compute bases");
    app.add_flag("-e,--even-edges", even_edges, "Use even edges");
    app.add_flag("-o,--overwrite", overwrite, "Overwrite existing files");
    app.add_flag("-t,--test_basis", test_basis, "compare basis with reference files");
    app.add_flag("-T,--test_matrix", test_matrix, "compare matrix with reference files");
    app.add_flag("-3, --triconnected", triconnected, "Use triconnected graphs only");


    CLI11_PARSE(app, argc, argv);

    // Check if the ranges are valid
    if (r_loops.start < 0 || r_loops.end < r_loops.start) {
        std::cerr << "Invalid range for loops: " << r_loops.start << ":" << r_loops.end << std::endl;
        return 1;
    }
    if (r_vertices.start < 0 || r_vertices.end < r_vertices.start) {
        std::cerr << "Invalid range for vertices: " << r_vertices.start << ":" << r_vertices.end << std::endl;
        return 1;
    }

    for (int l =r_loops.start; l <= r_loops.end; ++l) {
        for (int k = r_vertices.start; k <= r_vertices.end; ++k) {
            // for (bool even_edges : {true}) {
            OrdinaryGVS gvs(k, l, even_edges, triconnected);
            OrdinaryContract D(k, l, even_edges, triconnected);
            
            if (compute_bases) {
                tic();
                gvs.build_basis(overwrite);
                toc();
            }
            if (test_basis) {
                tic();
                gvs.test_basis_vs_ref();
                toc();
            }
            
            if (compute_matrices) {
                tic();
                D.build_matrix(overwrite);
                toc();
            }
            if (test_matrix) {
                tic();
                D.test_matrix_vs_ref();
                toc();
            }
            
        }
    }

    return 0;
}