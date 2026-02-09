mod graphs;
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

fn main() {
    let matches = Command::new("Graph list generator")
        .version("1.0")
        .author("Thomas Willwacher")
        .about("Creates a list of isomorphism classes of graphs in g6 format.")
        .arg(
            Arg::new("mode")
            .help("Graph generation mode: plain, even, or odd")
            .value_parser(["plain", "even", "odd"])
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
            Arg::new("defect")
                .help("The defect d. The number of edges is =3(loops)-3-d.")
                .value_parser(clap::value_parser!(usize))
                .required(true)
                .index(4),
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
                .long("triconnected")
                .required(false)
                .help("Generate only triconnected graphs.")
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

    let num_threads: usize = *matches.get_one::<usize>("num_threads").expect("Invalid number of threads");
    let n_loops_min: usize = *matches.get_one::<usize>("min_loops").expect("Invalid number of min loops");
    let n_loops_max: usize = *matches.get_one::<usize>("max_loops").expect("Invalid number of max loops");
    let n_defect = *matches.get_one::<usize>("defect").expect("Invalid defect");
    let geng_path = matches.get_one::<String>("geng").map(|s| s.as_str()).unwrap();

    if num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .expect("Failed to create thread pool");
    }


    for n_loops in n_loops_min..=n_loops_max {
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
                let start = std::time::Instant::now();
                println!("Generating graphs for {} loops and defect {}", n_loops, n_defect);
                generate_graphs(n_loops, n_defect).unwrap();
                let duration = start.elapsed();
                println!("Time elapsed: {:?}", duration);
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
                    Op.build_matrix(overwrite).expect("Build matrix failed");
                } else {
                    println!("Generating basis files for {} loops and defect {}", n_loops, n_defect);
                    OGC.build_basis(overwrite).expect("Build basis failed");
                }
            }

        }



    }
   

}
