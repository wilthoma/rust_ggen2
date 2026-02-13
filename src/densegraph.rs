use core::num;
// use rand::seq::SliceRandom;
// use rand::{Rng, SeedableRng};
use std::time::Instant;
use std::collections::BTreeMap;
use crate::helpers::*;

use std::fs::File;
use std::io::{BufRead, BufReader};

type HashType = usize;

type GraphScore = Vec<u64>;

#[derive(Clone, Debug)]
pub struct DenseGraph {
    pub num_vertices: u8,
    pub edges: Vec<(u8, u8)>,
    pub adj: Vec<u64>, // adjacency matrix as bit-packed rows
}


// utilities
fn inverse_permutation(perm: &Vec<u8>) -> Vec<u8> {
    let mut inv = vec![0u8; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p as usize] = i as u8;
    }
    inv
}

fn compose_permutations(a: &Vec<u8>, b: &Vec<u8>) -> Vec<u8> {
    a.iter().map(|&x| b[x as usize]).collect()
}


impl DenseGraph {
    pub fn new(num_vertices: u8, edges: Vec<(u8, u8)>) -> Self {
        let mut adj = vec![0u64; num_vertices as usize];
        for &(u, v) in &edges {
            adj[u as usize] |= 1u64 << v;
            adj[v as usize] |= 1u64 << u;
        }
        DenseGraph {
            num_vertices,
            edges: edges,
            adj,
        }
    }


    /// Returns a vector `res` such that `res[i]` is the number of vertices
    /// at graph distance exactly `i` from vertex `v`.
    pub fn distance_histogram(&self, v: u8) -> Vec<usize> {
        let n = self.num_vertices as usize;
        let mut res = vec![0usize; n + 1];

        let mut seen: u64 = 0;
        let mut frontier: u64 = 1u64 << (v as usize);
        seen |= frontier;
        let mut dist = 0usize;

        while frontier != 0 {
            // count bits in frontier -> number of vertices at distance dist
            res[dist] = frontier.count_ones() as usize;

            // next frontier = neighbors(frontier) & !seen
            let mut nbrs: u64 = 0;
            let mut mm = frontier;
            while mm != 0 {
                let u = mm.trailing_zeros() as usize;
                nbrs |= self.adj[u];
                mm &= mm - 1;
            }
            let next_frontier = nbrs & !seen;
            seen |= next_frontier;
            frontier = next_frontier;
            dist += 1;
        }

        // trim trailing zeros
        res.truncate(dist);
        res
    }

    pub fn distance_histogram_keys(&self) -> Vec<usize> {
        let weight_factor = self.num_vertices as usize;
        let mut histograms = Vec::new();
        for v in 0..self.num_vertices {
            let hist = self.distance_histogram(v);
            let mut sum = 0;
            let mut factor = 1;
            for (_, &count) in hist.iter().rev().enumerate() {
                sum += count * factor;
                factor *= weight_factor;
            }
            histograms.push(sum);
        }

        histograms
    }

    pub fn permute(&self, perm: &[u8]) -> DenseGraph {
        let mut edges: Vec<(u8, u8)> = self
            .edges
            .iter()
            .map(|&(u, v)| {
                let (a, b) = (perm[u as usize], perm[v as usize]);
                if a < b { (a, b) } else { (b, a) }
            })
            .collect();
        edges.sort();
        DenseGraph::new(self.num_vertices, edges)
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


    #[inline(always)]
    pub fn canonical_label(&self) -> (DenseGraph, Vec<u8>) {
        let (g, pp) = self.canonical_labels();
        return (g, pp[0].clone());
    }

    #[inline(always)]
    pub fn canonical_label_col(&self, init_colors: &[usize]) -> (DenseGraph, Vec<u8>) {
        let (g, pp) = self.canonical_labels_col(init_colors);
        return (g, pp[0].clone());
    }

    #[inline(always)]
    pub fn canonical_labels(&self) -> (DenseGraph, Vec<Vec<u8>>) {
        let zero_colors = vec![0usize; self.num_vertices as usize];
        self.canonical_labels_col(&zero_colors)
    }

    pub fn canonical_labels_col(&self, init_colors: &[usize]) -> (DenseGraph, Vec<Vec<u8>>) {
        let n = self.num_vertices as usize;
        // let mut classes: Vec<u64> = vec![]; //vec![(1<<n)-1]; // start with one big class
        let mut classes: Vec<u64> = vec![(1<<n)-1]; // start with one big class
        let start = Instant::now();
        // if let Some(colors) = init_colors {
        assert_eq!(init_colors.len(), n);
        // get some initial coloring by applying relatively strong vertex invariants
        classes = self.refined_coloring(&classes, init_colors);
 
        let hash = self.distance_histogram_keys();
        classes = self.refined_coloring(&classes, &hash);

        self.refine(&mut classes);
        // let hash = self.myhash(&classes);
        // classes = self.refined_coloring(&classes, &hash);
        let elapsed = start.elapsed();
        // println!("initial coloring took {:.6} ms", elapsed.as_secs_f64() * 1e3);

        let start = Instant::now();

        let mut best: Option<(GraphScore, Vec<Vec<u8>>)> = None;
        self.search_multi_bm(&classes, &mut best);
        let elapsed2 = start.elapsed();
        // println!("search_multi_bm took {:.6} ms", elapsed2.as_secs_f64() * 1e3);

        // display timing only if one took more than .01ms
        if elapsed.as_secs_f64() * 1e3 > 0.01 || elapsed2.as_secs_f64() * 1e3 > 0.01 {
            // let n_autos = self.automorphisms().len();
            // println!("Refinement took {:.6} ms, search took {:.6} ms", elapsed.as_secs_f64() * 1e3, elapsed2.as_secs_f64() * 1e3);
        }

        let (_ , perms) = best.unwrap();
        let gcanon = self.permute(&perms[0]);
        (gcanon, perms)
    }


    #[inline(always)]
    pub fn automorphisms(&self) -> Vec<Vec<u8>> {
        let zero_colors = vec![0usize; self.num_vertices as usize];
        return self.automorphisms_col(&zero_colors);
    }
    pub fn automorphisms_col(&self, init_colors: &[usize]) -> Vec<Vec<u8>> {
        let (_canon, best_perms) = self.canonical_labels_col(init_colors);
        if best_perms.is_empty() {
            return vec![];
        }
        let base = &best_perms[0];
        best_perms
            .iter()
            .map(|p| compose_permutations(base, &inverse_permutation(p)))
            .collect()
    }

    pub fn from_g6(g6: &str) -> DenseGraph {
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

        DenseGraph::new(n, edges)
    }
    
    // pub fn load_from_file(filename: &str) -> std::io::Result<Vec<String>> {
    //     let file = std::fs::File::open(filename)?;
    //     let reader = std::io::BufReader::new(file);
    //     // read first line and trsnform to int
    //     let mut lines = reader.lines();
    //     let first_line = lines.next().unwrap()?;
    //     let num_graphs: usize = first_line.trim().parse().unwrap();
    //     let mut g6_list = Vec::new();
    //     for line in lines { // .take(num_graphs) {
    //         let g6 = line?;
    //         g6_list.push(g6);
    //     }
    //     if g6_list.len() != num_graphs {
    //         return Err(std::io::Error::new(
    //             std::io::ErrorKind::InvalidData,
    //             "Number of graphs in file does not match the first line",
    //         ));
    //     }
    //     Ok(g6_list)
    // }

    /// Refines a given original coloring based on given hash values provided for every vertex.
    /// Each of the original classes is (possibly) split into multiple classes of vertices of equal hash values.
    /// The new subclasses are sorted by hash value.
    #[inline(always)]
    fn refined_coloring(&self, orig_classes: &[u64], hashes: &[HashType]) -> Vec<u64> {
        let n = self.num_vertices as usize;
        let mut new_classes = Vec::with_capacity(n);
        for &class_mask in orig_classes {
            if class_mask.count_ones() <= 1 {
                new_classes.push(class_mask);
                continue;
            }
            // Map from hash value to bitmask of vertices in this class with that hash
            let mut hash_map: BTreeMap<HashType, u64> = BTreeMap::new();
            let mut mm = class_mask;
            while mm != 0 {
                let v = mm.trailing_zeros() as usize;
                mm &= mm - 1;
                let h = hashes[v];
                *hash_map.entry(h).or_default() |= 1u64 << v;
            }
            for (_h, mask) in hash_map {
                new_classes.push(mask);
            }
        }
        new_classes
    }

    #[inline(always)]
    fn adjacency_hash(&self, classes: &[u64]) -> Vec<u64> {
        let n = self.num_vertices as usize;
        let mut hashes = vec![0u64; n];
        for v in 0..n {
            let mut h: u64 = 0;
            for (i, &cm) in classes.iter().enumerate() {
                let cnt = (self.adj[v] & cm).count_ones() as u64;
                // Simple multiplicative hash; 257 is small prime
                h = h.wrapping_mul(257).wrapping_add(cnt + (i as u64) * 17);
            }
            hashes[v] = h;
        }
        hashes
    }

    /// Compute a structural hash per vertex based on the current partition.
    /// Each color class is represented by a bitmask in `part`.
    /// Assumes self.adj[v] is a u64 bitmask of neighbors of v.
    #[inline(always)]
    pub fn myhash(&self, part: &[u64]) -> Vec<HashType> {
        let adj = &self.adj;
        let n = self.num_vertices as usize;
        let mut hashes:Vec<HashType> = vec![0; n];

        // For each vertex, accumulate neighbor counts per color class
        for v in 0..n {
            let mut h: HashType = 0xcbf29ce484222325; // FNV offset basis
            let a = adj[v];

            for &mask in part {
                // number of neighbors of v in this color class
                let c = (a & mask).count_ones() as HashType;

                // mix into hash (FNV-1a style)
                h ^= c.wrapping_add(0x9e3779b97f4a7c15);
                h = h.wrapping_mul(0x100000001b3);
            }

            // also include degree (for extra discrimination)
            h ^= a.count_ones() as HashType;
            h = h.wrapping_mul(0x9e3779b97f4a7c15);

            hashes[v] = h;
        }

        hashes
    }

    /// Refines a partition of vertices (given as bitmask vector) using adjacency information.
    /// Uses integer hashes instead of Vec<u8> signatures for speed.
    fn refine(&self, classes: &mut Vec<u64>) {
        let n = self.num_vertices as usize;
        let mut sigs = Vec::with_capacity(n);
        // let mut new_parts = Vec::with_capacity(n);

        // loop {

        //     // let start = Instant::now();
        //     let hashes = self.adjacency_hash(classes);
        //     // let duration1 = start.elapsed();
        //     // let start = Instant::now();
        //     let new_classes = self.refined_coloring(classes, &hashes);
        //     // let duration2 = start.elapsed();
        //     // println!("Times {:.6} ms, {:.6} ms", duration1.as_secs_f64() * 1e3, duration2.as_secs_f64() * 1e3);
        //     if new_classes.len() == classes.len() {
        //         break;
        //     }
        //     *classes = new_classes;
        // }

        loop {
            let mut changed = false;

            // For each class, split by neighborhood signatures
            for i in 0..classes.len() {
                let class_mask = classes[i];
                if class_mask.count_ones() <= 1 {
                    continue;
                }

                // Compute integer signature for each vertex in this class
                // let mut sigs: Vec<(usize, u8)> = Vec::new();
                sigs.clear();
                let mut mm = class_mask;
                while mm != 0 {
                    let v = mm.trailing_zeros() as usize;
                    mm &= mm - 1;

                    // Compute hash signature based on neighbor counts in each class
                    let mut h: usize = 0;
                    for (j, &cm) in classes.iter().enumerate() {
                        let cnt = (self.adj[v] & cm).count_ones() as usize;
                        // Simple multiplicative hash; 257 is small prime
                        h = h.wrapping_mul(257).wrapping_add(cnt + j * 17);
                    }

                    sigs.push((h, v as u8));
                }

                // Sort vertices by signature and split if necessary
                sigs.sort_unstable_by_key(|x| x.0);

                // Build new partition pieces
                let mut new_parts: Vec<u64> = Vec::with_capacity(n);
                // new_parts.clear();
                let mut cur_mask = 0u64;
                let mut cur_sig = sigs[0].0;

                for &(sig, v) in &sigs {
                    if sig != cur_sig {
                        new_parts.push(cur_mask);
                        cur_mask = 0;
                        cur_sig = sig;
                    }
                    cur_mask |= 1u64 << v;
                }
                new_parts.push(cur_mask);

                if new_parts.len() > 1 {
                    // Replace class i by the new parts
                    classes.remove(i);
                    for (k, part) in new_parts.into_iter().enumerate() {
                        classes.insert(i + k, part);
                    }
                    changed = true;
                    break; // restart refinement because indices changed
                }
            }

            if !changed {
                break;
            }
        }
    }

    #[inline(always)]
    fn graph_score(&self, perm: &[u8]) -> GraphScore {
        let n = self.num_vertices as usize;
        let mut adj = vec![0u64; n];
        for &(u, v) in &self.edges {
            let a = perm[u as usize] as usize;
            let b = perm[v as usize] as usize;
            adj[a] |= 1u64 << b;
            adj[b] |= 1u64 << a;
        }
        adj
    }

    // #[inline(always)]
    // fn graph_score_cls(&self, classes: &[u64]) -> GraphScore {
    //     // compute the adjacency matrix of the quotient graph, where each class is a super-vertex
    //     // and we have an edge between two super-vertices if there is any edge between any of their members
    //     let n = classes.len();
    //     let mut adj = vec![0u64; n];
    //     for i in 0..n {
    //         let ci = classes[i];
    //         for j in (i+1)..n {
    //             let cj = classes[j];
    //             // check if there is any edge between ci and cj
    //             let mut found = false;
    //             let mut mm = ci;
    //             while mm != 0 && !found {
    //                 let v = mm.trailing_zeros() as usize;
    //                 mm &= mm - 1;
    //                 if (self.adj[v] & cj) != 0 {
    //                     found = true;
    //                 }
    //             }
    //             if found {
    //                 adj[i] |= 1u64 << j;
    //                 adj[j] |= 1u64 << i;
    //             }
    //         }
    //     }
    //     adj
    // }

    fn search_multi_bm(
        &self,
        classes: &Vec<u64>,
        best: &mut Option<(GraphScore, Vec<Vec<u8>>)>,
    ) {
        let mut perm = vec![0; self.num_vertices as usize];


        if classes.iter().all(|cls| cls.count_ones() == 1) {
            // we found a leaf
            let mut idx = 0;
            for cls in classes {
                let v_idx = cls.trailing_zeros() as u8;
                perm[v_idx as usize] = idx as u8;
                idx += 1;
            }
            let gperm_score = self.graph_score(&perm);
            if let Some((best_graph_score, _)) = best {
                //let best_bitstr = best_graph.bitstring();
                // let g_bitstr = g_perm.bitstring();
                if gperm_score < *best_graph_score {
                    *best = Some((gperm_score, vec![perm.clone()]));
                } else if gperm_score == *best_graph_score {
                // } else if best_bitstr == g_bitstr {
                    if let Some((_, perms)) = best.as_mut() {
                        perms.push(perm.clone());
                    }
                }
            } else {
                *best = Some((gperm_score, vec![perm.clone()]));
            }
            return;
        }

        let class_pos = classes.iter().position(|cls| cls.count_ones() > 1).unwrap();
        let class = &classes[class_pos];

        for v in 0..self.num_vertices {
            if (class & (1u64 << v)) == 0 {
                continue;
            }
            let v = v as u8;
            // print!(".");
            let mut new_classes = Vec::new();
            for (i, cls) in classes.iter().enumerate() {
                if i == class_pos {
                    let others = cls & !(1u64 << v);
                    if others != 0 {
                        new_classes.push(others);
                    }
                    new_classes.push(1u64 << v);
                } else {
                    new_classes.push(*cls);
                }
            }

            // prune this branch if the quotient graph is already worse than the best found so far
            // Experimentally, pruning here does not help much, so it is commented out for now.
            // if let Some((best_graph_score, _)) = best {
            //     let gcls_score = self.graph_score_cls(&new_classes);
            //     if gcls_score > *best_graph_score {
            //         print!("x");
            //         continue;
            //     }
            // }

            let mut refined = new_classes.clone();
            self.refine(&mut refined);

            self.search_multi_bm(&refined, best);
        }
    }

    // create a Densegraph from an scd code. The scd code encodes a 3-regular graph, 
    // with the first three bytes the neighbors of vertex 0, followed
    // by the neighbors of vertex 1 etc.
    // Also note that scd uses 1-based indexing for vertices, so we need to subtract 1 from each neighbor index.
    // The code must have length 3n/2, where n is the number of vertices, and n must be even.
    pub fn from_scd_code(code : &[u8]) -> DenseGraph {
        let num_vertices = (2*code.len()/3) as u8;
        let num_edges = code.len() as u8; // 3 edges per vertex, but each edge is counted twice
        assert!(3*num_vertices as usize == 2*code.len());

        let mut edges = Vec::with_capacity(num_edges as usize);
        let mut cur_vertex = 0u8;
        let mut nb_counters = vec![0u8; num_vertices as usize]; // number of neighbors found for each vertex
        for (i, &neighbor) in code.iter().enumerate() {
            let neighbor = neighbor - 1; // convert from 1-based to 0-based indexing
            if neighbor >= num_vertices {
                panic!("Invalid neighbor index: {}", neighbor);
            }
            edges.push((cur_vertex, neighbor));
            nb_counters[cur_vertex as usize] += 1;
            nb_counters[neighbor as usize] += 1;
            if nb_counters[cur_vertex as usize] == 3 {
                cur_vertex += 1;
            }
        }
        DenseGraph::new(num_vertices, edges)
    }
}