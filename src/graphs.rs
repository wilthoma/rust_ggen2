use graph6_rs::Graph as G6Graph;
use rand::Rng;
use std::collections::HashSet;
use std::process::{Command, Stdio};
use std::io::Write;
use std::io::BufRead;
use std::io::BufReader;
use std::error::Error;
use std::thread;
use std::time::Instant;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use std::ffi::{CString, CStr};
use std::os::raw::c_char;
use std::ptr;

use crate::helpers::*;

use zeroize::Zeroize;
use crate::densegraph::DenseGraph;
use rustc_hash::FxHashSet;

use crate::gc_graph::{Graph as GCGraph, test_basis_vs_reference};

// unsafe extern "C" {
//     fn canonicalize_g6(input: *const c_char, output: *mut c_char, output_size: usize);
// }

use std::io;
use std::io::prelude::*;


#[derive(Clone)]
pub struct Graph {
    num_vertices: u8,
    edges: Vec<(u8, u8)>,
}

impl Graph {
    pub fn new(num_vertices: u8) -> Self {
        Graph {
            num_vertices,
            edges: Vec::new(),
        }
    }

    pub fn add_edge(&mut self, u: u8, v: u8) {
        if u<v {
            self.edges.push((u, v));
        } else {
            self.edges.push((v, u));
        }
    }

    pub fn to_g6(&self) -> String {
        let graph = self;
        let n = graph.num_vertices;
        assert!(n <= 62, "This encoder only supports graphs with at most 62 vertices.");

        // Write N(n)
        let mut result = String::new();
        result.push((n + 63) as char);

        // Create adjacency bit vector in the correct order
        let mut bitvec = Vec::new();
        for j in 1..n {
            for i in 0..j {
                let bit = graph.edges.contains(&(i, j)) || graph.edges.contains(&(j, i));
                bitvec.push(bit as u8);
            }
        }

        // Pad bitvec with zeros to make length multiple of 6
        while bitvec.len() % 6 != 0 {
            bitvec.push(0);
        }

        // Encode into 6-bit chunks
        for chunk in bitvec.chunks(6) {
            let mut value = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                value |= bit << (5 - i);
            }
            result.push((value + 63) as char);
        }

        result
    }

    pub fn add_edge_across(&self, e1idx : usize, e2idx: usize) -> Graph
    {
        // creates new vertices on the edges with indices e1idx and e1idx, and joins them by an edge
        assert_ne!(e1idx, e2idx, "Edges must be distinct");
        let new_n = self.num_vertices + 2;
        let n = self.num_vertices;
        let mut new_edges = vec![];
        let v1 = n;
        let v2 = n+1;
        for i in 0..self.edges.len() {
            let (u,v) = self.edges[i];
            if i==e1idx {
                new_edges.push((u,v1));
                new_edges.push((v,v1));
                new_edges.push((v1,v2));
            } else if i == e2idx {
                new_edges.push((u,v2));
                new_edges.push((v,v2));
            } else {
                new_edges.push( (u,v) );
            }
        }
        new_edges.sort();
        Graph {
            num_vertices: new_n,
            edges: new_edges,
        }
    }

    pub fn replace_edge_by_tetra(&self, eidx : usize) -> Graph {
        // replaces the edge with index eidx by a tetrahedron
        let new_n = self.num_vertices + 4;
        let (u,v) = self.edges[eidx];
        assert!(u<v);

        let mut new_edges = vec![];
        for i in 0..self.edges.len() {
            let (a,b) = self.edges[i];
            if i==eidx {
                let v1= new_n-4;
                let v2= new_n-3;
                let v3= new_n-2;
                let v4= new_n-1;
                new_edges.push((u, v1));
                new_edges.push((v1, v2));
                new_edges.push((v1, v3));
                new_edges.push((v2, v4));
                new_edges.push((v3, v4));
                new_edges.push((v2, v3));
                new_edges.push((v, v4));
            } else {
                new_edges.push( (a,b) );
            }
        }
        new_edges.sort();
        Graph {
            num_vertices: new_n,
            edges: new_edges,
        }
    }

    pub fn union(&self, other : &Graph) -> Graph {
        let new_n = self.num_vertices + other.num_vertices;
        // join the two edges vectors
        let mut edges = self.edges.clone();
        edges.extend(
            other
                .edges
                .iter()
                .map(|&(u, v)| (u + self.num_vertices, v + self.num_vertices)),
        );
        Graph {
            num_vertices: new_n,
            edges,
        }
    }


    pub fn contract_edge(&self, eidx : usize) -> Graph {
        let new_n = self.num_vertices-1;
        let (u,v) = self.edges[eidx];
        assert!(u<v);

        let new_edges = self.edges.iter()
                .enumerate()
                .filter_map(|(i, &(a, b))| {
                    if i == eidx {
                        return None;
                    }
                    let aa = if a < v { a } else if a == v { u } else { a - 1 };
                    let bb = if b < v { b } else if b == v { u } else { b - 1 };
                    if aa < bb {
                        Some((aa, bb))
                    } else if bb < aa {
                        Some((bb, aa))
                    } else {
                        None
                    }
                })
                .collect::<HashSet<_>>() // remove duplicates
                .into_iter()
                .collect::<Vec<_>>(); // convert to Vec
        
        
        Graph {
            num_vertices : new_n,
            edges : new_edges,
        }
    }

    pub fn contract_edge_opt(&self, eidx : usize) -> Option<Graph> {
        let g = self.contract_edge(eidx);
        if g.edges.len() +1  == self.edges.len() {
            Some(g)
        } else {
            None
        }
    }

    pub fn from_g6(g6: &str) -> Graph {
        let bytes = g6.as_bytes();
        assert!(!bytes.is_empty(), "Empty g6 string");

        // Decode N(n)
        let first = bytes[0];
        assert!(first >= 63, "Invalid graph6 string");

        let n = match first {
            63..=126 => (first - 63) as u8,
            _ => panic!("This decoder only supports n ≤ 62 (1-byte N(n))"),
        };

        // Compute number of bits in the upper triangle: n(n-1)/2
        let num_bits = (n as usize * (n as usize - 1)) / 2;
        let num_bytes = (num_bits + 5) / 6;

        let bit_data = &bytes[1..=num_bytes];
        let mut bits = Vec::with_capacity(num_bits);

        for &byte in bit_data {
            assert!(byte >= 63, "Invalid graph6 data byte");
            let val = byte - 63;
            for i in (0..6).rev() {
                bits.push((val >> i) & 1);
            }
        }

        // Trim any extra bits (if padding was added)
        bits.truncate(num_bits);

        // Reconstruct edge list in order: (0,1), (0,2), (1,2), (0,3), (1,3), (2,3), ...
        let mut edges = Vec::new();
        let mut k = 0;
        for j in 1..n {
            for i in 0..j {
                if bits[k] == 1 {
                    edges.push((i, j));
                }
                k += 1;
            }
        }

        Graph {
            num_vertices: n,
            edges,
        }
    }

    pub fn save_to_file(g6_list: &[String], filename: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::create(filename)?;
        // first line is number of graphs
        writeln!(file, "{}", g6_list.len())?;
        // write each graph6 string
        for g6 in g6_list {
            writeln!(file, "{}", g6)?;
        }
        Ok(())
    }


    pub fn tetrahedron_graph() -> Graph {
        Graph {
            num_vertices : 4,
            edges : vec![(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        }
    }

    pub fn tetrastring_graph(n_blocks : u8) -> Graph {
        let n = 4*n_blocks;
        let mut edges = vec![];
        for i in 0..n_blocks {
            edges.push((4*i, 4*i+1));
            edges.push((4*i, 4*i+2));
            edges.push((4*i+1, 4*i+2));
            edges.push((4*i+1, 4*i+3));
            edges.push((4*i+2, 4*i+3));
            if i== n_blocks-1 {
                edges.push((0, 4*i+3));
            } else {
                edges.push((4*i+3, 4*i+4));
            }
        }
        Graph {
            num_vertices : n,
            edges
        }

    }

    pub fn print(&self) {
        println!("Graph with {} vertices and {} edges. G6 code: {}.", self.num_vertices, self.edges.len(), self.to_g6());
        for (u,v) in &self.edges {
            println!("{} {}", u, v);
        }
    }

    pub fn to_densegraph(&self) -> DenseGraph {
        DenseGraph::new(self.num_vertices, self.edges.clone())
    }

    
}


// pub fn from_g6_ref(g6_str: &str) -> Self {
    //     let g6 = G6Graph::from_g6(g6_str).unwrap();
    //     let num_vertices = g6.n as u8;
    //     let bit_vec = g6.bit_vec;
    //     let mut edges = Vec::new();
    //     for i in 0..num_vertices{
    //         for j in 0..num_vertices{
    //             if i<j && bit_vec[  (i as usize)*g6.n+(j as usize)]>0 {
    //                 edges.push((i, j));
    //             }
    //         }
    //     }
    //     Graph { num_vertices, edges }
    // }

pub fn create_geng_ref(l: usize, d : usize) {
    let n = 2 * l - 2 - d;
    let e = 3*l-3- d;
    let filename = format!("data/ref/graphs{}_{}.geng", l, d);

    let geng_cmd = format!(
        "geng {} {}:{} -d3 -c -l > {}",
        n, e,e, filename
    );

    let status = Command::new("sh")
        .arg("-c")
        .arg(&geng_cmd)
        .status()
        .expect("Failed to run geng command");

    if !status.success() {
        panic!("geng command failed with status: {:?}", status);
    }
}

const BATCH_SIZE: usize = 64000;

// pub fn canonicalize_and_dedup_g6<I>(g6_iter: I) -> Result<HashSet<String>, Box<dyn Error>>
// where
//     I: IntoIterator<Item = String>,
// {
//     let mut seen = HashSet::new();
//     let mut out_buf = vec![0u8; 1024];

//     for g6 in g6_iter {
//         let c_input = CString::new(g6).expect("Invalid g6 input");
//         unsafe {
//             let out_ptr = out_buf.as_mut_ptr() as *mut c_char;
//             canonicalize_g6(c_input.as_ptr(), out_ptr, out_buf.len());
//             let c_output = CStr::from_ptr(out_ptr);
//             if let Ok(s) = c_output.to_str() {
//                 seen.insert(s.to_owned());
//             }
//         }
//     }

//     Ok(seen)
// }

pub fn canonicalize_and_dedup_g6(
    g6_list: &Vec<String>,
) -> FxHashSet<String>
{
    // convert g6 to Densegraph, then canonicalize, then convert back to g6, and deduplicate using a HashSet
    // do it in parallel using rayon
    let bar = ProgressBar::new(g6_list.len() as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) ETA: {eta_precise}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );
    let bar = Arc::new(bar);
    
    let canon_set: FxHashSet<String> = g6_list
        .par_iter()
        .map(|g6| {
            bar.inc(1);
            let dg = DenseGraph::from_g6(g6);
            let (canon_dg, _) = dg.canonical_label();
            canon_dg.to_g6()
        })
        .collect();
    
    bar.finish_with_message("Canonicalization complete");
    canon_set
}

/// Run g6 strings through `labelg` in batches and deduplicate the canonical results.
pub fn canonicalize_and_dedup_g6_old(
    g6_list: &Vec<String>,
    labelg_path: &str,
    // g6_iter: I,
) -> Result<HashSet<String>, Box<dyn Error>>
// where
//     I: IntoIterator<Item = String>,
{
    let mut canonical_set = HashSet::new();
    // let labelg_path = "labelg"; // Path to the labelg executable

    // let mut batch = Vec::with_capacity(BATCH_SIZE);
    // for g6 in g6_iter.into_iter() {
    //     batch.push(g6);
    //     if batch.len() >= BATCH_SIZE {
    //         let canonicals = run_labelg_batch(&batch, labelg_path)?;
    //         canonical_set.extend(canonicals);
    //         batch.clear();
    //     }
    // }
    // let mut all_g6: Vec<String> = g6_iter.into_iter().collect();
    let batches: Vec<_> = g6_list.chunks(BATCH_SIZE).map(|c| c.to_vec()).collect();

    let bar = ProgressBar::new(batches.len() as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) ETA: {eta_precise}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let results: Vec<HashSet<String>> = batches
        .par_iter()
        .map_init(
            || bar.clone(),
            |bar, batch| {
                let res = run_labelg_batch(batch, labelg_path).unwrap();
                bar.inc(1);
                res
            },
        )
        .collect();

    bar.finish_with_message("Canonicalization done");

    for res in results {
        let canonicals = res;
        canonical_set.extend(canonicals);
    }

    // Process any remaining graphs
    // if !batch.is_empty() {
    //     let canonicals = run_labelg_batch(&batch, labelg_path)?;
    //     canonical_set.extend(canonicals);
    // }

    Ok(canonical_set)
}

/// Helper to run one batch of g6 strings through labelg and collect output.
fn run_labelg_batch(
    batch: &[String],
    labelg_path: &str,
) -> Result<HashSet<String>, Box<dyn Error>> {
    let mut child = Command::new(labelg_path)
        .arg("-g")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;

    // Take ownership of stdin and move it to a thread for writing
    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    let input_batch = batch.to_owned();
    let writer_handle = thread::spawn(move || {
        for g6 in input_batch {
            // We ignore errors here; the read side handles subprocess failure
            let _ = writeln!(stdin, "{}", g6);
        }
        // Most important part: close stdin to signal EOF to labelg
        drop(stdin);
    });

    // Read from stdout in the main thread
    let stdout = child.stdout.take().ok_or("Failed to open stdout")?;
    let reader = BufReader::new(stdout);
    let mut canonicals = HashSet::new();

    for line in reader.lines() {
        let line = line?;
        canonicals.insert(line);
    }

    // Ensure the writing thread finishes
    writer_handle.join().expect("Writer thread panicked");

    let status = child.wait()?;
    if !status.success() {
        return Err(format!("labelg failed with exit code {:?}", status.code()).into());
    }

    Ok(canonicals)
}


pub fn is_satisfiable(g: usize, d: usize) -> bool {
    // d is the defect
    if g < 3 {
        return false;
    }
    let n = 2 * g - 2 - d;
    let e = 3 * g - 3 - d;

    if n * (n - 1) / 2 < e {
        return false;
    }
    true
}

pub fn generate_graphs(g : usize, d : usize) -> Result<(), Box<dyn std::error::Error>> {
    // d is the defect
    println!("Generating graphs with genus {} and defect {}...", g, d);

    if g<3  {
        println!("Warning: generate_graphs called with non-satisfiable paramters.");
        return Ok(());
    }
    let n = 2*g-2 -d;
    let e = 3*g-3-d; 

    if n*(n-1)/2 < e {
        println!("Warning: generate_graphs called with non-satisfiable paramters.");
        return Ok(());
    }

    let filename = format!("data/graphs{}_{}.g6", g, d);
    if d>0 {

        // contract an edge
        let otherfilename = format!("data/graphs{}_{}.g6", g, d-1);
        let mut g6s = load_g6_file(&otherfilename)?;
        let total = g6s.len();
        let bar = get_progress_bar(total);
        // let counter = Arc::new(AtomicUsize::new(0));
        let start = Instant::now();
        let g6_set: FxHashSet<String> = g6s
            .par_iter()
            .map_init(
                || bar.clone(),
                |bar, s| {
                    bar.inc(1);
                    Graph::from_g6(s)
                },
            )
            .flat_map_iter(|g| {
                (0..=e)
                    .filter_map(move |idx| g.contract_edge_opt(idx))
                    .map(|g| 
                        // Convert to densegraph and canonize
                        {
                            let dg = g.to_densegraph();
                            let (canon_dg, _) = dg.canonical_label();
                            canon_dg.to_g6()
                        })
            })
            .collect();

        // bar.finish();
        // let g6list: Vec<String> = g6s
        //     .par_iter()
        //     .map(|s| Graph::from_g6(s))
        //     .flat_map_iter(|g| (0..=e).filter_map(move |idx| g.contract_edge_opt(idx)))
        //     .map(|g| g.to_g6())
        //     .collect();
        // let mut g6list_pre = g6s
        // .par_iter()
        // .map_init(
        //     || {
        //         let bar = bar.clone();
        //         // let counter = counter.clone();
        //         // (bar, counter)
        //         bar
        //     },
        //     |bar, s| {
        //         let g = Graph::from_g6(s);
        //         // counter.fetch_add(1, Ordering::Relaxed);
        //         bar.inc(1);
        //         (0..=e)
        //             .filter_map(move |idx| g.contract_edge_opt(idx))
        //             .map(|g| g.to_g6())
        //             .collect::<Vec<_>>()
        //     },
        // );
        // println!("Done");
        // pause();
        // // println!("{} graphs processed in {:.2?}", total, start.elapsed());

        // let g6list: Vec<String> = g6list_pre.flatten()
        // .collect();

        g6s.zeroize(); // clear the g6s vector to free memory

        //pause();

        //Graph::save_to_file(&g6list, "temp.g6")?;
        //println!("saved to temp.g6");
        //pause();

        let msg = format!(
            "Done in {:.2?} — total graphs: {}",
            start.elapsed(),
            g6_set.len()
        );
        bar.finish_with_message(msg);



        // println!("{} graphs generated, deduplicating...", g6list.len());
        // let start = Instant::now();
        // let g6_canon: FxHashSet<String> = canonicalize_and_dedup_g6(&g6list);
        // println!("Deduplication took {:.2?}, {} unique graphs remaining.", start.elapsed(), g6_canon.len());
        let g6_vec: Vec<String> = g6_set.into_iter().collect();
        println!("Saving {} graphs to file {}", g6_vec.len(), filename);
        Graph::save_to_file(&g6_vec, &filename)?;
        println!("Done.");

    } else if d==0 {
        // start with the tetrastring
        let mut g6list = vec![];
        if g==3 //g%2 == 1 
        {
            g6list.push(Graph::tetrastring_graph(((g-1)/2) as u8).to_g6());
        }

        // connect two components
        println!("... connecting two components ...");
        for l1 in 3..g-2 {
            let l2 = g-l1;
            if l1<l2 {
                continue;
            } 
            let fname1 = format!("data/graphs{}_{}.g6", l1, 0);
            let fname2 = format!("data/graphs{}_{}.g6", l2, 0);
            println!("{} graphs loaded from {}...", l1, fname1);
            println!("{} graphs loaded from {}...", l2, fname2);
            let g6s1 = load_g6_file(&fname1)?;
            let g6s2 = load_g6_file(&fname2)?;
            let total = g6s1.len()* g6s2.len();
            let bar = get_progress_bar(total);
            let g6s1 = g6s1.into_par_iter().map(|s| Graph::from_g6(&s)).collect::<Vec<_>>();
            let g6s2 = g6s2.into_par_iter().map(|s| Graph::from_g6(&s)).collect::<Vec<_>>();
            for g1 in g6s1.iter() {
                let e1 = g1.edges.len();
                for g2 in g6s2.iter() {
                    let e2 = g2.edges.len();
                    let gg = g1.union(g2);

                    for i in 0..e1 {
                        for j in 0..e2 {
                            let ggg = gg.add_edge_across(i, j+e1);
                            g6list.push(ggg.to_g6());
                        }
                    }
                    bar.inc(1);
                }
            }
            bar.finish_with_message("Done.");
        }

        // add a tetra across an edge
        println!("... adding tetras ...");
        if g>=5 {
            let l1 = g-2;
            let otherfname = format!("data/graphs{}_0.g6", l1);
            let g6s = load_g6_file(&otherfname)?;
            println!("{} graphs loaded from {}...", g6s.len(), otherfname);
            let total = g6s.len();
            let bar = get_progress_bar(total);
            for (id,gg) in g6s.into_iter().map(|s| Graph::from_g6(&s)).enumerate(){
                for i in 0..gg.edges.len() {
                    let ggg = gg.replace_edge_by_tetra(i);
                    g6list.push(ggg.to_g6());
                }
                bar.inc(1);
            }
            bar.finish_with_message("Done.");
        }
        
        // add graphs obtained by connecting two edges
        println!("... connecting edges ...");
        if g>3 {
            let otherfname = format!("data/graphs{}_0.g6", g-1);
            let g6s = load_g6_file(&otherfname)?;
            println!("{} graphs loaded from {}...", g6s.len(), otherfname);
            let ee = e-3;
            
            let total = g6s.len();
            let bar2 = get_progress_bar(total);

            let new_g6s: Vec<String> = g6s
                .par_iter()
                .enumerate()
                .flat_map_iter(|(_id, s)| {
                    let gg = Graph::from_g6(s);
                    let mut local_vec = Vec::new();
                    for i in 0..ee {
                        for j in (i + 1)..ee {
                            let ggg = gg.add_edge_across(i, j);
                            local_vec.push(ggg.to_g6());
                        }
                    }
                    // Update progress bar for this chunk
                    bar2.inc(1);
                    local_vec
                })
                .collect();

            bar2.finish_with_message("Done.");
            // let new_g6s: Vec<String> = g6s
            //     .par_iter()
            //     .enumerate()
            //     .flat_map_iter(|(id, s)| {
            //         // if id % 1000 == 0 {
            //         //     println!("{} graphs processed...", id);
            //         // }
            //         let gg = Graph::from_g6(s);
            //         // let gg = gg.clone();
            //         (0..ee)
            //             .flat_map(move |i| {
            //                 let gg = gg.clone();
            //                 ((i + 1)..ee).map(move |j| {
            //                     let ggg = gg.add_edge_across(i, j);
            //                     ggg.to_g6()
            //                 })
            //             })
            //             .collect::<Vec<_>>()
            //     })
            //     .collect();

            g6list.extend(new_g6s);
        }
        
        // dedup g6list
        // let mut set = HashSet::new();
        // g6list.retain(|g| set.insert(g.clone()));

        println!("{} graphs generated, deduplicating...", g6list.len());
        let start = Instant::now();
        let g6_canon: FxHashSet<String> = canonicalize_and_dedup_g6(&g6list);
        println!("Deduplication took {:.2?}, {} unique graphs remaining.", start.elapsed(), g6_canon.len());

        let g6_vec: Vec<String> = g6_canon.into_iter().collect();
        Graph::save_to_file(&g6_vec, &filename)?;
    }

    Ok(())
}

pub fn compare_file_to_ref(l : usize, d : usize) -> Result<(), Box<dyn Error>>{
        let filename = format!("data/graphs{}_{}.g6", l, d);
        let refname = format!("data/ref/graphs{}_{}.geng", l, d);

        test_basis_vs_reference(&filename, &refname, false, false, false)


        // let g6s = load_g6_file(&filename).unwrap();
        // let g6_ref = load_g6_file_nohdr(&refname).unwrap();
        // // assert_eq!(g6s.len(), g6_ref.len(), "Number of graphs in {} and {} do not match", filename, refname);
        // // take the difference of g6s and g6_ref as sets
        // let g6s_set: FxHashSet<String> = g6s.into_iter().collect();
        // let g6_ref_set = canonicalize_and_dedup_g6(&g6_ref);
        // // sanity check that nothing was removed
        // if g6_ref_set.len() != g6_ref.len() {
        //     panic!("Error: canonicalization removed some graphs from the reference set. Original size: {}, after canonicalization: {}.", g6_ref.len(), g6_ref_set.len());
        // }
        // let diff: Vec<_> = g6s_set.difference(&g6_ref_set).collect();
        // if !diff.is_empty() {
        //     println!("Graphs in {} but not in {}: {:?}", filename, refname, diff);
        // }
        // assert!(diff.is_empty(), "Graphs in {} but not in {}", filename, refname);
        // // take the difference of g6_ref and g6s as sets
        // let g6s_set_owned: FxHashSet<String> = g6s_set.into_iter().collect();
        // let diff: Vec<_> = g6_ref_set.difference(&g6s_set_owned).collect();
        // if !diff.is_empty() {
        //     println!("Graphs in {} but not in {}: {:?}", refname, filename, diff);
        // }
        // assert!(diff.is_empty(), "Graphs in {} but not in {}", refname, filename);
        // println!("Graphs in {} and {} match", filename, refname);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graphs_equal(g1: &Graph, g2: &Graph) -> bool {
        if g1.num_vertices != g2.num_vertices {
            return false;
        }
        let mut e1 = g1.edges.clone();
        let mut e2 = g2.edges.clone();
        e1.sort();
        e2.sort();
        e1 == e2
    }

    #[test]
    fn test_random_graphs_g6_roundtrip() {
        let mut rng = rand::rng();
        for _ in 0..10 {
            let n = rng.random_range(1..=20); // keep n small for test speed
            let mut g = Graph::new(n);
            // Randomly add edges
            for u in 0..n {
                for v in (u + 1)..n {
                    if rng.random_bool(0.5) {
                        g.add_edge(u, v);
                    }
                }
            }
            let g6 = g.to_g6();
            let g2 = Graph::from_g6(&g6);
            let g6_2 = g2.to_g6();
            assert!(
                g6 == g6_2,
                "G6 roundtrip failed: g6={}, back={:?}",
                g6,
                g6_2
            );
            assert!(
                graphs_equal(&g, &g2),
                "Graph roundtrip failed: orig={:?}, g6={}, back={:?}",
                g.edges,
                g6,
                g2.edges
            );
        }
    }


    #[test]
    fn test_filecompare() {
        compare_file_to_ref(3, 0);
        compare_file_to_ref(4, 0);
        compare_file_to_ref(5, 0);
        compare_file_to_ref(6, 0);
        compare_file_to_ref(7, 0);
        compare_file_to_ref(8, 0);
        compare_file_to_ref(9, 0);
        compare_file_to_ref(10, 0);
    }

    // #[test]
    // fn test_canonicalize_and_dedup_g6() {
    //     // These three are isomorphic, the last is not
    //     let g6_graphs = vec![
    //         "D??".to_string(),
    //         "D_@".to_string(),
    //         "D`?".to_string(), // all isomorphic
    //         "D?@".to_string(), // triangle + pendant (non-isomorphic)
    //     ];


    //     let result: Result<FxHashSet<String>, Box<dyn Error + 'static>> = canonicalize_and_dedup_g6(&g6_graphs);
    //     match result {
    //         Ok(canon_set) => {
    //             // Should deduplicate the first three, so only 2 canonical forms
    //             println!("Canonical forms: {:?}", canon_set);
    //             assert_eq!(
    //                 canon_set.len(),
    //                 3,
    //                 "Expected 3 canonical forms, got {:?}",
    //                 canon_set
    //             );
    //         }
    //         Err(e) => {
    //             panic!("canonicalize_and_dedup_g6 failed: {}", e);
    //         }
    //     }
    // }

    #[test]
    fn test_tetrahedron_equals_tetrastring_1() {
        let g1 = Graph::tetrahedron_graph();
        let g2 = Graph::tetrastring_graph(1);
        assert!(
            graphs_equal(&g1, &g2),
            "Tetrahedron and tetrastring(1) should be equal: {:?} vs {:?}",
            g1.edges,
            g2.edges
        );
    }
}
