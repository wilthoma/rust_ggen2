#include "mygraphs.hh"
#include <chrono>
#include <iostream>
#include "CLI11.hpp"
#include "indicators.hpp"
#include "PreOrdinaryGC.hh"

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
    Range r_defect;
    bool compute_bases = false;
    bool overwrite = false;
    bool test_basis = false;


    app.add_option("range_loops", r_loops, "Range in format start:end")->required();
    app.add_option("range_defect", r_defect, "Range in format start:end")->required();
    app.add_flag("-b,--compute-bases", compute_bases, "Compute bases");
    app.add_flag("-o,--overwrite", overwrite, "Overwrite existing files");
    app.add_flag("-t,--test", test_basis, "compare basis with reference files");


    CLI11_PARSE(app, argc, argv);

    // Check if the ranges are valid
    if (r_loops.start < 0 || r_loops.end < r_loops.start) {
        std::cerr << "Invalid range for loops: " << r_loops.start << ":" << r_loops.end << std::endl;
        return 1;
    }
    if (r_defect.start < 0 || r_defect.end < r_defect.start) {
        std::cerr << "Invalid range for defect: " << r_defect.start << ":" << r_defect.end << std::endl;
        return 1;
    }

    for (int l =r_loops.start; l <= r_loops.end; ++l) {
        for (int k = r_defect.start; k <= r_defect.end; ++k) {
            cout << "Processing loops: " << l << ", defect: " << k << std::endl;
            // for (bool even_edges : {true}) {
            PreOrdinaryGVS gvs(l, k);
            
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
            
            
        }
    }

    return 0;
}