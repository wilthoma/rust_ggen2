mod graphs;
mod helpers;
use helpers::*;
// mod kneissler;
// mod nauty_interface;
// use nauty_interface::*;
// mod nauty_interface2;
// mod bliss_interface;
// use nauty_interface2::*;
// use bliss_interface::*;
mod densegraph;
mod gc_graph;
use gc_graph::{Graph as GCGraph, OrdinaryGVS, OrdinaryContract};
use graphs::*;
// use kneissler::*;
use clap::{Command, Arg};
use zstd::zstd_safe::CompressionLevel;

const COMPRESS_LEVEL: i32 = 5; // 1-9, 1 is fastest, 9 is best compression

fn main() {
    let matches = Command::new("Graph list generator")
        .version("1.0")
        .author("Thomas Willwacher")
        .about("Creates a list of isomorphism classes of graphs in g6 format.")
        .after_help("EXAMPLES:\n    cargo run --release -- plain 1 3 0 2\n    cargo run --release -- even 2 4 1 3 --compress\n    cargo run --release -- odd 1 2 0 1 -t 8")
        .arg(
            Arg::new("mode")
            .help("Graph generation mode: plain, even, odd, oddprune, alltest (which runs all tests in sequence) or allclean (deletes all generated files).")
            .value_parser(["plain", "even", "odd", "alltest", "allclean", "oddprune"])
            .required(true)
            .index(1),
        )
        .arg(
            Arg::new("min_loops")
                .help("The minimum loop order.")
                .value_parser(clap::value_parser!(usize))
                .required(true)
                .index(2),
        )
        .arg(
            Arg::new("max_loops")
                .help("The maximum loop order.")
                .value_parser(clap::value_parser!(usize))
                .required(true)
                .index(3),
        )
        .arg(
            Arg::new("min_defect")
                .help("The minimum defect d. The number of edges is =3(loops)-3-d.")
                .value_parser(clap::value_parser!(usize))
                .required(true)
                .index(4),
        )
        .arg(
            Arg::new("max_defect")
                .help("The maximum defect d. The number of edges is =3(loops)-3-d.")
                .value_parser(clap::value_parser!(usize))
                .required(true)
                .index(5),
        )
        .arg(
            Arg::new("overwrite")
                .short('o')
                .long("overwrite")
                .required(false)
                .help("Overwrite existing files")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("ref")
                .long("ref")
                .required(false)
                .help("Generate geng reference files")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("M")
                .short('M')
                .long("matrices")
                .required(false)
                .help("Generate or test matrix files")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("nobuild")
                .long("nobuild")
                .required(false)
                .help("Do not generate graphs.")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("3")
                .short('3')
                .long("triconnected")
                .required(false)
                .help("Generate only triconnected graphs.")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("compress")
                .short('c')
                .long("compress")
                .required(false)
                .help("Compress output files.")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("import")
                .short('i')
                .long("import")
                .required(false)
                .help("Imports .scd files from genreg instead of generating defect 0 plain graphs. Scd files are expected in data/genreg/, using genreg's default file naming convention.")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("test")
                .long("test")
                .required(false)
                .help("Compare to reference files for verification")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("num_threads")
                .short('t')
                .long("threads")
                .help("Size of shared threadpool to use (excluding the worker threads)")
                .value_parser(clap::value_parser!(usize))
                .value_name("NUM")
                .default_value("4")
        )
        .arg(
            Arg::new("geng")
                .long("geng")
                .help("Path to the geng executable")
                .value_name("PATH")
                .required(false)
                .default_value("geng")
        )

        .get_matches();

    let mode = matches.get_one::<String>("mode").expect("Mode is required");
    let overwrite = *matches.get_one::<bool>("overwrite").unwrap_or(&false);
    let test = *matches.get_one::<bool>("test").unwrap_or(&false);
    let gen_ref = *matches.get_one::<bool>("ref").unwrap_or(&false);
    let nobuild = *matches.get_one::<bool>("nobuild").unwrap_or(&false);
    let gen_matrices = *matches.get_one::<bool>("M").unwrap_or(&false);
    let use_triconnected = *matches.get_one::<bool>("3").unwrap_or(&false);
    let compress = *matches.get_one::<bool>("compress").unwrap_or(&false);
    let import = *matches.get_one::<bool>("import").unwrap_or(&false);

    let num_threads: usize = *matches.get_one::<usize>("num_threads").expect("Invalid number of threads");
    let n_loops_min: usize = *matches.get_one::<usize>("min_loops").expect("Invalid number of min loops");
    let n_loops_max: usize = *matches.get_one::<usize>("max_loops").expect("Invalid number of max loops");
    let n_defect_min: usize = *matches.get_one::<usize>("min_defect").expect("Invalid number of min defect");
    let n_defect_max: usize = *matches.get_one::<usize>("max_defect").expect("Invalid number of max defect");
    let geng_path = matches.get_one::<String>("geng").map(|s| s.as_str()).unwrap();

    let compress_level = if compress { COMPRESS_LEVEL } else { 0 };

    if num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .expect("Failed to create thread pool");
    }

    if mode == "alltest" {
        test_everything(n_loops_min, n_loops_max, n_defect_min, n_defect_max, compress_level);
        return;
    }

    if mode == "allclean" {
        clean_all_generated_files(n_loops_min, n_loops_max, n_defect_min, n_defect_max);
        return;
    }

    if mode == "oddprune" {
        println!("Generating odd pruned basis graphs for loops in {}..={} and defect 0 (defect ignored for now)", n_loops_min, n_loops_max);
        let n_defect = 0; // for now we ignore the defect since we don't have the necessary basis files for nonzero defect
        for n_loops in n_loops_min..=n_loops_max {
            if !is_satisfiable(n_loops, n_defect) {
                continue;
            }
            let n_vertices = 2 * n_loops - 2 - n_defect;
            let OGC = OrdinaryGVS::new(n_vertices as u8, n_loops as u8, false, use_triconnected);
            let start = std::time::Instant::now();
            println!("Generating odd pruneable graphs for {} loops and defect {}", n_loops, n_defect);
            OGC.prune_basis(overwrite, compress_level).expect("Build odd pruneable basis failed");
            let duration = start.elapsed();
            println!("Time elapsed: {:?}", duration);
        }
        return;
    }

    for n_loops in n_loops_min..=n_loops_max {
        for n_defect in n_defect_min..=n_defect_max {
            if !is_satisfiable(n_loops, n_defect) {
                continue;
            }
            if mode == "plain" {
                if gen_ref {
                    let start = std::time::Instant::now();
                    println!("Generating reference file for {} loops and defect {}", n_loops, n_defect);
                    create_geng_ref(n_loops, n_defect);
                    let duration = start.elapsed();
                    println!("Time elapsed: {:?}", duration);
                }

                if test {
                    compare_file_to_ref(n_loops, n_defect);
                }

                if !gen_ref && !nobuild && !test{
                    // check if we should import genreg
                    if import && n_defect == 0 {
                        let start = std::time::Instant::now();
                        println!("Importing graphs from genreg for {} loops and defect {}", n_loops, n_defect);
                        import_genreg_graphs(n_loops, compress_level).unwrap();
                        let duration = start.elapsed();
                        println!("Time elapsed: {:?}", duration);
                        continue;
                    } else {
                        // normal plain graph generation
                        let start = std::time::Instant::now();
                        println!("Generating graphs for {} loops and defect {}", n_loops, n_defect);
                        generate_graphs(n_loops, n_defect, compress_level).unwrap();
                        let duration = start.elapsed();
                        println!("Time elapsed: {:?}", duration);
                    }
                }

            } else { // mode == "even" or "odd"
                let even_edges = mode == "even";
                let n_vertices = 2 * n_loops - 2 - n_defect;
                if gen_ref {
                    panic!("Reference file generation is only supported in plain mode.");
                }
                let OGC = OrdinaryGVS::new(n_vertices as u8, n_loops as u8,even_edges, use_triconnected);
                let Op = OrdinaryContract::new(n_vertices as u8, n_loops as u8, even_edges, use_triconnected);
                let start = std::time::Instant::now();

                if test {
                    if gen_matrices {
                        println!("Testing matrix files against reference for {} loops and defect {}", n_loops, n_defect);
                        Op.test_matrix_vs_ref().expect("Matrix test failed");
                    } else {
                        println!("Testing basis files against reference for {} loops and defect {}", n_loops, n_defect);
                        OGC.test_basis_vs_ref().expect("Basis test failed");
                    }

                } 

                if !test {
                    if gen_matrices {
                        println!("Generating matrix files for {} loops and defect {}", n_loops, n_defect);
                        Op.build_matrix(overwrite, compress_level).expect("Build matrix failed");
                    } else {
                        println!("Generating basis files for {} loops and defect {}", n_loops, n_defect);
                        OGC.build_basis(overwrite, compress_level).expect("Build basis failed");
                    }
                }

            }
        }


    }
   

}


fn test_everything(min_loops: usize, max_loops: usize, min_defect: usize, max_defect: usize, compression_level:i32) {
    // recreates all basis and matrix files in the specified range and compares to reference
    
    println!("Testing everything for loops in {}..={} and defect in {}..={}", min_loops, max_loops, min_defect, max_defect);
    println!("Test plain graph generation...");
    for n_loops in min_loops..=max_loops {
        for n_defect in min_defect..=max_defect {
            if !is_satisfiable(n_loops, n_defect) {
                continue;
            }
            println!("Testing plain graph generation for {} loops and defect {}", n_loops, n_defect);
            generate_graphs(n_loops, n_defect, compression_level).expect("Graph generation failed");
            compare_file_to_ref(n_loops, n_defect).expect("Comparison failed");

        }
    }

    println!("Testing GC basis generation...");
    for use_triconnected in [true, false] {
        for even_edges in [true, false] {
            for n_loops in min_loops..=max_loops {
                for n_defect in min_defect..=max_defect {
                    if !is_satisfiable(n_loops, n_defect) {
                        continue;
                    }
                    println!("Testing GC basis generation for {} loops and defect {}, even edges: {}, use triconnected: {}", n_loops, n_defect, even_edges, use_triconnected);
                    let n_vertices = 2 * n_loops - 2 - n_defect;
                    let OGC = OrdinaryGVS::new(n_vertices as u8, n_loops as u8, even_edges, use_triconnected);
                    OGC.build_basis(true, compression_level).expect("Build basis failed");
                    OGC.test_basis_vs_ref().expect("Basis test failed");
                }
            }
        }
    }

    println!("Testing GC matrix generation...");
    for use_triconnected in [true, false] {
        for even_edges in [true, false] {
            for n_loops in min_loops..=max_loops {
                for n_defect in min_defect..max_defect { // one defect less since we don't necessarily have the necessary basis file
                    if !is_satisfiable(n_loops, n_defect) {
                        continue;
                    }
                    println!("Testing GC matrix generation for {} loops and defect {}, even edges: {}, use triconnected: {}", n_loops, n_defect, even_edges, use_triconnected);
                    let n_vertices = 2 * n_loops - 2 - n_defect;
                    let Op = OrdinaryContract::new(n_vertices as u8, n_loops as u8, even_edges, use_triconnected);
                    Op.build_matrix(true, compression_level).expect("Build matrix failed");
                    Op.test_matrix_vs_ref().expect("Matrix test failed");
                }
            }
        }
    }

    println!("All tests passed!");
}

fn clean_all_generated_files(min_loops: usize, max_loops: usize, min_defect: usize, max_defect: usize) {
    println!("Cleaning all generated files for loops in {}..={} and defect in {}..={}", min_loops, max_loops, min_defect, max_defect);
    // clean the plain graphs
    for n_loops in min_loops..=max_loops {
        for n_defect in min_defect..=max_defect {
            if !is_satisfiable(n_loops, n_defect) {
                continue;
            }
            let filename = plain_filename(n_loops, n_defect);
            println!("Cleaning file {}", filename);
            if std::path::Path::new(&filename).exists() {
                std::fs::remove_file(&filename).expect("Failed to remove file");
            }
        }
    }

    // clean the GC basis and matrix files
    for use_triconnected in [true, false] {
        for even_edges in [true, false] {
            for n_loops in min_loops..=max_loops {
                for n_defect in min_defect..=max_defect {
                    if !is_satisfiable(n_loops, n_defect) {
                        continue; 
                    }
                    let n_vertices = 2 * n_loops - 2 - n_defect;
                    let OGC = OrdinaryGVS::new(n_vertices as u8, n_loops as u8, even_edges, use_triconnected);
                    let filename = OGC.get_basis_file_path();
                    println!("Cleaning file {}", filename);
                    if std::path::Path::new(&filename).exists() {
                        std::fs::remove_file(&filename).expect("Failed to remove file");    
                    }
                    // also clean the compressed version if it exists
                    let compressed_filename = format!("{}{}", filename, ZSTD_EXTENSION);
                    if std::path::Path::new(&compressed_filename).exists() {
                        std::fs::remove_file(&compressed_filename).expect("Failed to remove compressed file");
                    }
                }
            }
        }
    }
    // clean the matrix files (both basis and compressed)
    for use_triconnected in [true, false] {
        for even_edges in [true, false] {
            for n_loops in min_loops..=max_loops {
                for n_defect in min_defect..max_defect {
                    if !is_satisfiable(n_loops, n_defect) {
                        continue; 
                    }
                    let n_vertices = 2 * n_loops - 2 - n_defect;
                    let Op = OrdinaryContract::new(n_vertices as u8, n_loops as u8, even_edges, use_triconnected);
                    let filename = Op.get_matrix_file_path();
                    println!("Cleaning file {}", filename);
                    if std::path::Path::new(&filename).exists() {
                        std::fs::remove_file(&filename).expect("Failed to remove file");    
                    }
                    // also clean the compressed version if it exists
                    let compressed_filename = format!("{}{}", filename, ZSTD_EXTENSION);
                    if std::path::Path::new(&compressed_filename).exists() {
                        std::fs::remove_file(&compressed_filename).expect("Failed to remove compressed file");
                    }
                }
            }
        }
    }

}