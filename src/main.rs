mod graphs;
// mod kneissler;
// mod nauty_interface;
// use nauty_interface::*;
// mod nauty_interface2;
// mod bliss_interface;
// use nauty_interface2::*;
// use bliss_interface::*;
mod densegraph;
use graphs::*;
// use kneissler::*;
use clap::{Command, Arg};

fn main() {
    let matches = Command::new("Graph list generator")
        .version("1.0")
        .author("Thomas Willwacher")
        .about("Creates a list of isomorphism classes of graphs in g6 format.")
        .arg(
            Arg::new("min_loops")
                .help("The minimum loop order.")
                .value_parser(clap::value_parser!(usize))
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("max_loops")
                .help("The maximum loop order.")
                .value_parser(clap::value_parser!(usize))
                .required(true)
                .index(2),
        )
        .arg(
            Arg::new("defect")
                .help("The defect d. The number of edges is =3(loops)-3-d.")
                .value_parser(clap::value_parser!(usize))
                .required(true)
                .index(3),
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
            Arg::new("nobuild")
                .long("nobuild")
                .required(false)
                .help("Do not generate graphs.")
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
            Arg::new("labelg")
                .long("labelg")
                .help("Path to the labelg executable")
                .value_name("PATH")
                .required(false)
                .default_value("labelg")
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

    let overwrite = *matches.get_one::<bool>("overwrite").unwrap_or(&false);
    let test = *matches.get_one::<bool>("test").unwrap_or(&false);
    let gen_ref = *matches.get_one::<bool>("ref").unwrap_or(&false);
    let nobuild = *matches.get_one::<bool>("nobuild").unwrap_or(&false);

    let num_threads: usize = *matches.get_one::<usize>("num_threads").expect("Invalid number of threads");
    let n_loops_min: usize = *matches.get_one::<usize>("min_loops").expect("Invalid number of min loops");
    let n_loops_max: usize = *matches.get_one::<usize>("max_loops").expect("Invalid number of max loops");
    let n_defect = *matches.get_one::<usize>("defect").expect("Invalid defect");
    let labelg_path = matches.get_one::<String>("labelg").map(|s| s.as_str()).unwrap();
    let geng_path = matches.get_one::<String>("geng").map(|s| s.as_str()).unwrap();
    

    if num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .expect("Failed to create thread pool");
    }

    // println!("Hello, world!");
    // Graph::tetrahedron_graph().print();
    // Graph::tetrastring_graph(1).print();

    // create_geng_ref(3,0);
    // create_geng_ref(4,0);
    // create_geng_ref(5,0);
    // create_geng_ref(5,0);
    // create_geng_ref(6,0);
    // create_geng_ref(7,0);
    // create_geng_ref(8,0);
    // create_geng_ref(9,0);
    // create_geng_ref(10,0);
    // create_geng_ref(11,0);



    // for l in 3..10 {
    //     for k in 1..l {
    //         if !is_satisfiable(l, k) {
    //             continue;
    //         }
    //         println!("{} {}", l, k);
    //         generate_graphs(l, k).unwrap();
    //         create_geng_ref(l,k);
    //         compare_file_to_ref(l,k);
    //     }
    // }

    for n_loops in n_loops_min..=n_loops_max {
        if !is_satisfiable(n_loops, n_defect) {
            continue;
        }
        if gen_ref {
            let start = std::time::Instant::now();
            println!("Generating reference file for {} loops and defect {}", n_loops, n_defect);
            create_geng_ref(n_loops, n_defect);
            let duration = start.elapsed();
            println!("Time elapsed: {:?}", duration);
        }
        if !nobuild {
            let start = std::time::Instant::now();
            println!("Generating graphs for {} loops and defect {}", n_loops, n_defect);
            generate_graphs(n_loops, n_defect, labelg_path).unwrap();
            let duration = start.elapsed();
            println!("Time elapsed: {:?}", duration);
        }
        if test {
            compare_file_to_ref(n_loops, n_defect);
        }
    }
    // let start = std::time::Instant::now();
    // generate_graphs(n_loops,n_defect, labelg_path).unwrap();

    // // for l in 12..13 {
    // //     // compute_all_kneissler_graphs(l, 0);
    // //     // compute_all_kneissler_graphs(l, 1);
    // //     // compute_all_kneissler_graphs(l, 2);
    // //     // compute_all_kneissler_graphs(l, 3);
    // // }
    

    // let duration = start.elapsed();
    // println!("Time elapsed: {:?}", duration);

    // let g6 = "Dhc"; // Example: star graph on 4 vertices

    // mainz();

    // nauty_interface::compute_automorphisms(g6);

    // generate_graphs(3,0).unwrap();
    // generate_graphs(4,0).unwrap();
    // generate_graphs(5,0).unwrap();
    // generate_graphs(6,0).unwrap();
    // generate_graphs(7,0).unwrap();
    // generate_graphs(8,0).unwrap();
    // generate_graphs(9,0).unwrap();
    // let start = std::time::Instant::now();
    // generate_graphs(12,0).unwrap();
    // let duration = start.elapsed();
    // println!("Time elapsed: {:?}", duration);
    // // check correctness
    // compare_file_to_ref(12,0);
    // generate_graphs(11,0).unwrap();

}
