use std::fs::File;
use std::io::{BufRead, BufReader};
use indicatif::{ParallelProgressIterator, ProgressBar};
use std::io::Write;
use crate::densegraph::DenseGraph;
use rustc_hash::{FxHashMap, FxHashSet};
use rayon::prelude::*;
use crate::helpers::*;
use std::time::Instant;

// use hashbrown::HashSet;
// use rustc_hash::FxBuildHasher;

pub type FxHashSetHB<T> = FxHashSet<T>; //HashSet<T, FxBuildHasher>;

// f determines whether to keep an element. 
// The set is mutated in-place by removing elements for which f returns false. 
//The function f can query membership in the original set, but should not mutate it.
// the function returns the number of removed elements. The implementation is optimized for the case where only a small fraction of the elements are removed, by first collecting the elements to remove in a separate vector and then removing them from the set in a second phase.
// pub fn filter_in_place_by_removing<T>(
//     x: &mut FxHashSetHB<T>,
//     f: impl Sync + Fn(&T, &FxHashSetHB<T>) -> bool,
// ) -> usize
// where
//     T: Sync + Send + Eq + std::hash::Hash + Clone,
// {
//     // Immutable snapshot for membership queries during phase 1
//     let x_ref: &FxHashSetHB<T> = &*x;
//     let total = x.len() as u64;

//     // Expect small removal set: 1–10% of |X|
//     // let est_remove = (x.len() / 20).max(1024); // tune: assumes ~5% removed
//     let to_remove: Vec<T> = x_ref
//         .par_iter()
//         .progress_count(total)
//         .progress_with_style(get_progress_bar_style())
//         .filter_map(|item| {
//             if !f(item, x_ref) {
//                 Some(item.clone())
//             } else {
//                 None
//             }
//         })
//         .collect();

//     let num_removed = to_remove.len();
//     // Phase 2: mutate
//     for k in to_remove {
//         x.remove(&k);
//     }
//     num_removed
// }
pub fn filter_in_place_by_removing<T>(
    x: &mut FxHashSetHB<T>,
    f: impl Sync + Fn(&T, &FxHashSetHB<T>) -> bool,
) -> usize
where
    T: Sync + Send + Eq + std::hash::Hash + Clone,
{
    // Immutable snapshot for membership queries during phase 1
    let x_ref: &FxHashSetHB<T> = &*x;
    let pb = get_progress_bar(x.len());
    // pb.set_draw_delta(50_000); // tune: update progress bar every 50k items

    // Expect small removal set: 1–10% of |X|
    let est_remove = (x.len() / 20).max(1024); // tune: assumes ~5% removed
    let to_remove: Vec<T> = x_ref
        .par_iter()
        .fold(
            || (Vec::with_capacity(est_remove / rayon::current_num_threads().max(1)), 0u64),
            |(mut acc, mut ctr), item| {
                if !f(item, x_ref) {
                    acc.push(item.clone());
                }
                ctr += 1;
                if ctr == 4096 {
                    pb.inc(ctr);
                    ctr = 0;
                }
                (acc, ctr)
            },
        )
        .map(|(out, leftover)| {
            if leftover != 0 {
                pb.inc(leftover);
            }
            out
        })
        .reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            a
        });

    let num_removed = to_remove.len();
    // Phase 2: mutate
    for k in to_remove {
        x.remove(&k);
    }
    num_removed
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    u: u8,
    v: u8,
    data: i32,
}

impl Edge {
    fn new(u: u8, v: u8, data: i32) -> Self {
        Edge { u, v, data }
    }
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.u == other.u && self.v == other.v && self.data == other.data
    }
}

impl Eq for Edge {}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.u, self.v).cmp(&(other.u, other.v))
    }
}

#[derive(Clone)]
pub struct Graph {
    num_vertices: u8,
    edges: Vec<Edge>,
}

impl Graph {
    pub fn new(n: u8) -> Self {
        Graph {
            num_vertices: n,
            edges: Vec::new(),
        }
    }

    pub fn with_edges(n: u8, edges: Vec<Edge>) -> Self {
        Graph {
            num_vertices: n,
            edges,
        }
    }

    pub fn add_edge(&mut self, u: u8, v: u8, data: i32) {
        let (u, v) = if u < v { (u, v) } else { (v, u) };
        self.edges.push(Edge::new(u, v, data));
    }

    pub fn to_g6(&self) -> String {
        let n = self.num_vertices;
        if n > 62 {
            panic!("Only supports graphs with at most 62 vertices");
        }
        let mut result = String::new();
        result.push((n as u8 + 63) as char);
        
        let mut bitvec = Vec::new();
        for j in 1..n {
            for i in 0..j {
                let found = self.edges.iter().any(|e| 
                    (e.u == i && e.v == j) || (e.u == j && e.v == i)
                );
                bitvec.push(if found { 1u8 } else { 0u8 });
            }
        }
        
        while bitvec.len() % 6 != 0 {
            bitvec.push(0);
        }
        
        for chunk in bitvec.chunks(6) {
            let mut value: u8 = 0;
            for (i, &bit) in chunk.iter().enumerate() {
                value |= bit << (5 - i);
            }
            result.push((value + 63) as char);
        }
        
        result
    }

    pub fn from_g6(g6: &str) -> Self {
        if g6.is_empty() {
            panic!("Empty g6 string");
        }
        
        let first = g6.as_bytes()[0] as u8;
        if first < 63 {
            panic!("Invalid graph6 string: {}", g6);
        }
        
        let n = if first >= 63 && first <= 126 {
            first - 63
        } else {
            panic!("Only supports n ≤ 62: {}", g6);
        };
        
        let num_bits = (n as usize) * (n as usize - 1) / 2;
        let num_bytes = (num_bits + 5) / 6;
        
        if g6.len() < 1 + num_bytes {
            panic!("g6 string too short");
        }
        
        let mut bits = Vec::new();
        for i in 0..num_bytes {
            let val = (g6.as_bytes()[1 + i] as u8) - 63;
            for j in (0..=5).rev() {
                bits.push((val >> j) & 1);
            }
        }
        bits.truncate(num_bits);
        
        let mut edges = Vec::new();
        let mut k = 0;
        for j in 1..n {
            for i in 0..j {
                if bits[k] == 1 {
                    edges.push(Edge::new(i, j, 0));
                }
                k += 1;
            }
        }
        
        Graph::with_edges(n, edges)
    }

    pub fn get_neighbors(&self, v: u8) -> std::collections::HashSet<u8> {
        let mut neighbors = std::collections::HashSet::new();
        for e in &self.edges {
            if e.u == v {
                neighbors.insert(e.v);
            } else if e.v == v {
                neighbors.insert(e.u);
            }
        }
        neighbors
    }

    fn sort_edges(&mut self) {
        self.edges.sort();
    }

    fn print(&self) {
        println!(
            "Graph with {} vertices and {} edges. G6 code: {}.",
            self.num_vertices,
            self.edges.len(),
            self.to_g6()
        );
        for e in &self.edges {
            println!("{} {} {}", e.u, e.v, e.data);
        }
    }

    // fn load_from_file(filename: &str) -> Result<Vec<String>, std::io::Error> {
        
    //     let file = File::open(filename)
    //         .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Failed to open file for reading"))?;
    //     let mut reader = BufReader::new(file);
        
    //     let mut first_line = String::new();
    //     reader.read_line(&mut first_line)?;
    //     let num_graphs: usize = first_line.trim().parse()
    //         .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Invalid number in first line"))?;
        
    //     let mut g6_list = Vec::new();
    //     for line in reader.lines() {
    //         let line = line?;
    //         if !line.is_empty() {
    //             g6_list.push(line);
    //         }
    //     }
        
    //     if g6_list.len() != num_graphs {
    //         return Err(std::io::Error::new(std::io::ErrorKind::Other, "Number of graphs does not match"));
    //     }
        
    //     Ok(g6_list)
    // }

    // fn load_from_file_nohdr(filename: &str) -> Result<Vec<String>, std::io::Error> {
        
    //     let file = File::open(filename)
    //         .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Failed to open file for reading"))?;
    //     let reader = BufReader::new(file);
        
    //     let g6_list = reader.lines()
    //         .collect::<Result<Vec<_>, _>>()?
    //         .into_iter()
    //         .filter(|line| !line.is_empty())
    //         .collect();
        
    //     Ok(g6_list)
    // }


    fn contract_edge(&self, eidx: usize) -> Graph {
        if self.num_vertices < 2 {
            panic!("Not enough vertices to contract");
        }
        let new_n = self.num_vertices - 1;
        let Edge { u, v, .. } = self.edges[eidx];
        if !(u < v) {
            panic!("Edge must be (u < v)");
        }
        let mut edge_set = std::collections::BTreeSet::new();
        for (i, edge) in self.edges.iter().enumerate() {
            if i == eidx {
                continue;
            }
            let a = edge.u;
            let b = edge.v;
            let d = edge.data;
            let aa = if a < v { a } else if a == v { u } else { a - 1 };
            let bb = if b < v { b } else if b == v { u } else { b - 1 };
            if aa < bb {
                edge_set.insert(Edge::new(aa, bb, d));
            } else if bb < aa {
                edge_set.insert(Edge::new(bb, aa, d));
            }
        }
        let new_edges: Vec<Edge> = edge_set.into_iter().collect();
        Graph::with_edges(new_n, new_edges)
    }

    fn relabel(&mut self, new_labels: &[u8]) {
        if new_labels.len() != self.num_vertices as usize {
            panic!("Invalid relabeling vector size");
        }
        for e in &mut self.edges {
            e.u = new_labels[e.u as usize];
            e.v = new_labels[e.v as usize];
            if e.u > e.v {
                std::mem::swap(&mut e.u, &mut e.v);
            }
        }
        self.sort_edges();
    }

    fn number_edges(&mut self) {
        self.sort_edges();
        for (i, e) in self.edges.iter_mut().enumerate() {
            e.data = i as i32;
        }
    }

    fn perm_sign(&self, p: &[u8], even_edges: bool) -> i32 {
        if even_edges {
            let mut sign = permutation_sign(p);
            for e in &self.edges {
                let u = e.u;
                let v = e.v;
                if (u < v && p[u as usize] > p[v as usize]) || (u > v && p[u as usize] < p[v as usize]) {
                    sign *= -1;
                }
            }
            sign
        } else {
            let mut g1 = Graph::with_edges(self.num_vertices, self.edges.clone());
            g1.number_edges();
            g1.relabel(p);
            g1.sort_edges();
            let perm: Vec<i32> = g1.edges.iter().map(|e| e.data).collect();
            permutation_sign(&perm)
        }
    }

    fn print_edges(&self) {
        for e in &self.edges {
            println!("{} {} {}", e.u, e.v, e.data);
        }
    }

    fn get_contractions_with_sign(&self, even_edges: bool) -> Vec<(Graph, i32)> {
        let mut image = Vec::new();
        for i in 0..self.edges.len() {
            let Edge { u, v, .. } = self.edges[i];
            let pp = permute_to_left(u, v, self.num_vertices);
            let mut sgn = self.perm_sign(&pp, even_edges);
            let mut g1 : Graph = self.clone();
            g1.relabel(&pp);
            g1.number_edges();
            let prev_size = g1.edges.len();
            g1 = g1.contract_edge(0);
            if prev_size - g1.edges.len() != 1 {
                continue;
            }
            if !even_edges {
                let mut p = Vec::new();
                g1.sort_edges();
                for e in &g1.edges {
                    p.push(e.data - 1);
                }
                sgn *= permutation_sign(&p);
            } else {
                sgn *= -1;
            }
            image.push((g1, sgn));
        }
        image
    }


    fn contract_edge_with_sign(&self, eidx: usize, even_edges: bool) -> Option<(Graph, i32)> {
        // let mut image = Vec::new();
        let i = eidx;
        // for i in 0..self.edges.len() {
            let Edge { u, v, .. } = self.edges[i];
            let pp = permute_to_left(u, v, self.num_vertices);
            let mut sgn = self.perm_sign(&pp, even_edges);
            let mut g1 : Graph = self.clone();
            g1.relabel(&pp);
            g1.number_edges();
            let prev_size = g1.edges.len();
            g1 = g1.contract_edge(0);
            if prev_size - g1.edges.len() != 1 {
                return None;
            }
            if !even_edges {
                let mut p = Vec::new();
                g1.sort_edges();
                for e in &g1.edges {
                    p.push(e.data - 1);
                }
                sgn *= permutation_sign(&p);
            } else {
                sgn *= -1;
            }
            return Some((g1, sgn));
        // }
    }

    fn check_valid(&self, defect: usize, err_msg: &str) -> bool {
        // check whether all vertex indices are < num_vertices
        for e in &self.edges {
            if e.u >= self.num_vertices || e.v >= self.num_vertices {
                eprintln!("{} Graph {} has vertex index >= num_vertices", err_msg, self.to_g6());
                return false;
            }
        }
        
        // check whether all vertices are ≥ 3-valent
        for i in 0..self.num_vertices {
            let degree = self.edges.iter().filter(|e| e.u == i || e.v == i).count();
            if degree < 3 {
                eprintln!("{} Graph {} Vertex {} has degree {}", err_msg, self.to_g6(), i, degree);
                return false;
            }
        }
        
        // there are no multiple edges or self-edges
        for (u, v, _) in self.edges.iter().map(|e| (e.u, e.v, e.data)) {
            if u == v {
                eprintln!("{} Graph {} has self-edge {}", err_msg, self.to_g6(), u);
                return false;
            }
            if u > v {
                eprintln!("{} Graph {} has wrongly ordered edge {} {}", err_msg, self.to_g6(), u, v);
                return false;
            }
            let cnt = self.edges.iter().filter(|e| e.u == u && e.v == v).count();
            if cnt > 1 {
                eprintln!("{} Graph {} has multiple edges {} {}", err_msg, self.to_g6(), u, v);
                return false;
            }
        }
        
        // check whether the graph is connected
        let mut visited = vec![false; self.num_vertices as usize];
        let mut stack = vec![0u8];
        while let Some(v) = stack.pop() {
            if visited[v as usize] {
                continue;
            }
            visited[v as usize] = true;
            for e in &self.edges {
                if e.u == v && !visited[e.v as usize] {
                    stack.push(e.v);
                } else if e.v == v && !visited[e.u as usize] {
                    stack.push(e.u);
                }
            }
        }
        
        for i in 0..self.num_vertices {
            if !visited[i as usize] {
                eprintln!("{} Graph {} is not connected", err_msg, self.to_g6());
                return false;
            }
        }
        
        // check if defect = 2*edges - 3*vertices
        if defect + 3 * (self.num_vertices as usize) != 2 * self.edges.len() {
            let true_defect = 2 * self.edges.len() - 3 * (self.num_vertices as usize);
            eprintln!("{} Graph {} has defect {} (not {})", err_msg, self.to_g6(), true_defect, defect);
            return false;
        }
        
        true
    }

    pub fn to_densegraph(&self) -> DenseGraph {
        let edges = self.edges.iter().map(|e| (e.u, e.v)).collect();
        DenseGraph::new(self.num_vertices, edges)
    }

    fn check_g6_valid(g6: &str, defect: usize, err_msg: &str) -> bool {
        let g = Graph::from_g6(g6);
        g.check_valid(defect, err_msg)
    }

    fn is_biconnected(&self, ignore_vertex: Option<u8>) -> bool {
        if self.num_vertices <= 2 {
            return false;
        }

        let mut adj: Vec<Vec<u8>> = vec![Vec::new(); self.num_vertices as usize];
        for e in &self.edges {
            if let Some(iv) = ignore_vertex {
                if e.u == iv || e.v == iv {
                    continue;
                }
            }
            adj[e.u as usize].push(e.v);
            adj[e.v as usize].push(e.u);
        }

        let mut disc = vec![-1; self.num_vertices as usize];
        let mut low = vec![-1; self.num_vertices as usize];
        let mut parent = vec![-1i8; self.num_vertices as usize];
        let mut time = 0;
        let mut has_articulation = false;

        let start = (0..self.num_vertices).find(|i| {
            ignore_vertex.is_none() || ignore_vertex != Some(*i)
        });

        if start.is_none() {
            return false;
        }

        let mut dfs = |u: u8, disc: &mut Vec<i32>, low: &mut Vec<i32>, parent: &mut Vec<i8>, time: &mut i32, has_articulation: &mut bool| {
            fn visit(u: u8, disc: &mut Vec<i32>, low: &mut Vec<i32>, parent: &mut Vec<i8>, adj: &[Vec<u8>], time: &mut i32, has_articulation: &mut bool, ignore_vertex: Option<u8>) {
                disc[u as usize] = *time;
                low[u as usize] = *time;
                *time += 1;
                let mut children = 0;

                for &v in &adj[u as usize] {
                    if let Some(iv) = ignore_vertex {
                        if v == iv {
                            continue;
                        }
                    }
                    if disc[v as usize] == -1 {
                        parent[v as usize] = u as i8;
                        children += 1;
                        visit(v, disc, low, parent, adj, time, has_articulation, ignore_vertex);
                        low[u as usize] = low[u as usize].min(low[v as usize]);

                        if (parent[u as usize] != -1 && low[v as usize] >= disc[u as usize]) ||
                           (parent[u as usize] == -1 && children > 1) {
                            *has_articulation = true;
                        }
                    } else if (v as i8) != parent[u as usize] {
                        low[u as usize] = low[u as usize].min(disc[v as usize]);
                    }
                }
            }
            visit(u, disc, low, parent, &adj, time, has_articulation, ignore_vertex);
        };

        dfs(start.unwrap(), &mut disc, &mut low, &mut parent, &mut time, &mut has_articulation);

        for i in 0..self.num_vertices {
            if let Some(iv) = ignore_vertex {
                if i == iv {
                    continue;
                }
            }
            if disc[i as usize] == -1 {
                return false;
            }
        }

        !has_articulation
    }

    fn is_triconnected(&self) -> bool {
        if !self.is_biconnected(None) || self.num_vertices < 4 {
            return false;
        }

        for v in 0..self.num_vertices {
            if !self.is_biconnected(Some(v)) {
                return false;
            }
        }

        true
    }

    fn has_odd_automorphism(&self, even_edges: bool) -> bool {
        let dg = self.to_densegraph();
        let auts = dg.automorphisms();
        for p in auts {
            let sign = self.perm_sign(&p, even_edges);
            if sign == -1 {
                return true;
            }
        }
        false
    }

    fn to_canon_g6(&self) -> String {
        let dg = self.to_densegraph();
        let (canon, _) = dg.canonical_label();
        canon.to_g6()
    }

    fn to_canon_g6_sgn(&self, even_edges: bool) -> (String, i32) {
        let dg = self.to_densegraph();
        let (canon, perm) = dg.canonical_label();
        let sign = self.perm_sign(&perm, even_edges);
        (canon.to_g6(), sign)
    }

    fn edge_flips_with_sign(&self, eidx: usize, even_edges: bool) -> Vec<(Graph, i32)> {
        let mut ret = Vec::new();
        let Edge { u, v, .. } = self.edges[eidx];

        // build the two graphs obtained by detaching the first edge from u and the i-th edge from v
        // and reattaching them in the other way around. The sign is obtained by the permutation that sends the first graph to the second one,
        // in the cacnonical ordering of edges

        let pp = permute_to_left(u, v, self.num_vertices);
        let mut sgn = self.perm_sign(&pp, even_edges);
        let mut g1 : Graph = self.clone();
        g1.relabel(&pp);
        g1.number_edges();
        let prev_size = g1.edges.len();
        // the first edge is now (0,1).
        // let Edge { u: u1, v: v1, data: d1 } = g1.edges[1];
        // let Edge { u: u2, v: v2, data: d2 } = g1.edges[2];
        // let Edge { u: u3, v: v3, data: d3 } = g1.edges[3];

        let mut g2 = g1.clone();

        // swap
        g1.edges[1].u = 1; 
        g1.edges[3].u = 0;
        g2.edges[1].u = 1; 
        g2.edges[4].u = 0;
        if !even_edges {
            let mut p = Vec::new();
            g1.sort_edges();
            for e in &g1.edges {
                p.push(e.data - 1);
            }
            ret.push((g1, sgn * permutation_sign(&p)));
            let mut p2 = Vec::new();
            g2.sort_edges();
            for e in &g2.edges {
                p2.push(e.data - 1);
            }
            ret.push((g2, sgn * permutation_sign(&p2)));
        } else {
            // not implemented
            panic!("Not implemented for even edges");
        }

        ret
    }

    fn is_pruneable_edge(&self, eidx: usize, valid_graphs: &FxHashSet<String>, even_edges: bool) -> bool {
        // conditions for 1-pruneable are: The given edge e=(u,v) is such that 
        // 1) contracting the edge is possible and gives a nonzero graph
        // 2) no other graph in the basis has this contraction.
        //    the two other such candidate graphs are obtained by permuting the edges incident at (u,v)

        // first check that contracting the edge gives a nonzero graph
        if let Some((g1, sgn1)) = self.contract_edge_with_sign(eidx, even_edges)
        {
            if g1.has_odd_automorphism(even_edges) {
                return false;
            }

            // now produce the other two graphs
            let swaps = self.edge_flips_with_sign(eidx, even_edges);
            let mut found = 0;
            let g6 = self.to_canon_g6();
            for (g2, sgn2) in swaps {
                let g2_g6 = g2.to_canon_g6();
                if g2_g6 != g6 && valid_graphs.contains(&g2_g6) {
                    found += 1;
                }
            }
            return found == 0;
        } else {
            return false;
        }

    }

    fn is_pruneable(&self, valid_graphs: &FxHashSet<String>, even_edges: bool) -> bool {
        // checks whether the graph is one-pruneable.
        // currently only implemented for 3-regular graphs with odd edges.
        assert_eq!(self.edges.len(), 3 * self.num_vertices as usize / 2);

        // conditions for 1-pruneable are: There exists an edge e=(u,v), such that 
        // 1) contracting the edge is possible and gives a nonzero graph
        // 2) no other graph in the basis has this contraction.
        //    the two other such candidate graphs are obtained by permuting the edges incident at (u,v)

        // check if any edge is pruneable
        for eidx in 0..self.edges.len() {
            if self.is_pruneable_edge(eidx, valid_graphs, even_edges) {
                return true;
            }
        }
        false
    }

}

fn get_type_string(even_edges: bool) -> String {
    if even_edges {
        "even_edges".to_string()
    } else {
        "odd_edges".to_string()
    }
}

pub struct OrdinaryGVS {
    pub num_vertices: u8,
    pub num_loops: u8,
    pub even_edges: bool,
    pub use_triconnected: bool,
}

impl OrdinaryGVS {
    pub fn new(num_vertices: u8, num_loops: u8, even_edges: bool, use_triconnected: bool) -> Self {
        OrdinaryGVS {
            num_vertices,
            num_loops,
            even_edges,
            use_triconnected,
        }
    }

    pub fn get_basis_file_path(&self) -> String {
        if self.use_triconnected {
            format!(
                "data/ordinary/tri/{}/gra{}_{}.g6",
                get_type_string(self.even_edges),
                self.num_vertices,
                self.num_loops
            )
        } else {
            format!(
                "data/ordinary/{}/gra{}_{}.g6",
                get_type_string(self.even_edges),
                self.num_vertices,
                self.num_loops
            )
        }
    }

    pub fn get_pruned_basis_file_path(&self) -> String {
        if self.use_triconnected {
            format!(
                "data/ordinary/tri/{}/gra_pruned{}_{}.g6",
                get_type_string(self.even_edges),
                self.num_vertices,
                self.num_loops
            )
        } else {
            format!(
                "data/ordinary/{}/gra_pruned{}_{}.g6",
                get_type_string(self.even_edges),
                self.num_vertices,
                self.num_loops
            )
        }
    }

    pub fn get_basis_ref_file_path(&self) -> String {
        if self.use_triconnected {
            format!(
                "data/ordinary/tri/ref/{}/gra_tri{}_{}.g6",
                get_type_string(self.even_edges),
                self.num_vertices,
                self.num_loops
            )
        } else {
            format!(
                "data/ordinary/ref/{}/gra{}_{}.g6",
                get_type_string(self.even_edges),
                self.num_vertices,
                self.num_loops
            )
        }
    }

    pub fn get_input_file_path(&self) -> String {
        let defect = 2 * self.num_loops as i32 - 2 - self.num_vertices as i32;
        format!("data/graphs{}_{}.g6", self.num_loops, defect)
    }

    pub fn is_valid(&self) -> bool {
        let n_edges = self.num_loops as usize + self.num_vertices as usize - 1;
        (3 * self.num_vertices as usize <= 2 * n_edges)
            && (self.num_vertices > 0)
            && (n_edges <= self.num_vertices as usize * (self.num_vertices as usize - 1) / 2)
    }

    pub fn get_basis_g6(&self) -> Result<Vec<String>, std::io::Error> {
        if !self.is_valid() {
            return Ok(Vec::new());
        }
        load_g6_file(&self.get_basis_file_path())
    }

    pub fn build_basis(&self, ignore_existing_files: bool, compression_level: i32) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_valid() {
            return Ok(());
        }

        let basis_path = self.get_basis_file_path();
        if !ignore_existing_files && std::path::Path::new(&basis_path).exists() {
            println!("Basis file already exists: {}", basis_path);
            return Ok(());
        }

        let infile = self.get_input_file_path();
        let in_g6s = load_g6_file(&infile)?;
        let num_graphs = in_g6s.len();

        if in_g6s.is_empty() {
            eprintln!("No graphs found in input file: {}", infile);
            return Ok(());
        }

        println!("Building basis for {}", basis_path);

        // let bar = get_progress_bar(num_graphs);

        let mut out_g6s: Vec<String> = in_g6s.par_iter()
            .progress_with_style(get_progress_bar_style())
            .filter_map(|g6| {
                let g = Graph::from_g6(g6);
                let keep = (self.use_triconnected && g.is_triconnected()) || 
                   (!self.use_triconnected && g.is_biconnected(None));
                if !keep || g.has_odd_automorphism(self.even_edges) {
                    return None;
                }
                Some(g.to_canon_g6())
            })
            .collect();

        // sort the list of g6 strings to have a canonical order in the basis
        println!("Sorting basis...");
        out_g6s.par_sort();

        println!("Found {} graphs in the basis. Writing to {}...", out_g6s.len(), basis_path);
        save_g6_file(&out_g6s, &basis_path, compression_level)?;

        println!("Done.");

        Ok(())
    }

    pub fn prune_basis(&self, ignore_existing_files: bool, compression_level: i32) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_valid() {
            return Ok(());
        }

        let basis_in = self.get_basis_file_path();
        let basis_out = self.get_pruned_basis_file_path();
        if !ignore_existing_files && std::path::Path::new(&basis_out).exists() {
            println!("Pruned basis file already exists: {}", basis_out);
            return Ok(());
        }
        if !std::path::Path::new(&basis_in).exists() {
            println!("Basis file does not exist for pruning: {}", basis_in);
            return Err( "Basis file does not exist for pruning".into());
        }

        let mut g6s = load_g6_file(&basis_in)?;
        let num_graphs = g6s.len();
        let mut cur_num_graphs = num_graphs;

        println!("Building set of valid graphs for pruning...");
        let start = Instant::now();
        let mut g6_set = g6s.par_iter().cloned().collect::<FxHashSet<String>>();
        println!("Done building set of valid graphs. Time taken: {:.2?}", start.elapsed());
        let mut cur_round =0;

        loop {
            
            println!("Pruning basis for {} in round {}. Original size: {}, Current size: {}...", self.to_string(), cur_round, num_graphs, cur_num_graphs);
            // remove all pruneable strings from g6s
            let n_removed = filter_in_place_by_removing(&mut g6_set, 
                |g6, validset| {
                    let g = Graph::from_g6(g6);
                    !g.is_pruneable(validset, self.even_edges)
            });
            println!("Done. Removed {} graphs. New size: {}.", n_removed, g6_set.len());
            if n_removed == 0 {
                break;
            }
            cur_num_graphs = g6_set.len();
            cur_round += 1;
        }
        println!("Pruning complete. Final size: {}. Writing to {}...", g6_set.len(), basis_out);
        let mut out_g6s: Vec<String> = g6_set.into_iter().collect();
        let start = Instant::now();
        out_g6s.par_sort();
        println!("Done sorting pruned basis. Time taken: {:.2?}", start.elapsed());
        save_g6_file(&out_g6s, &basis_out, compression_level)?;
        
        Ok(())
    }

    pub fn to_string(&self) -> String {
        format!(
            "OrdinaryGVS({}, {}, {})",
            self.num_vertices,
            self.num_loops,
            get_type_string(self.even_edges)
        )
    }

    pub fn test_basis_vs_ref(&self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_valid() {
            println!("Basis check void for {}", self.to_string());
            return Ok(());
        }
        test_basis_vs_reference(
            &self.get_basis_file_path(),
            &self.get_basis_ref_file_path(),
            self.even_edges,
            true,
            true
        )
    }
}

pub struct OrdinaryContract {
    pub domain: OrdinaryGVS,
    pub target: OrdinaryGVS,
    pub num_loops: u8,
    pub num_vertices: u8,
    pub even_edges: bool,
    pub use_triconnected: bool,
}

impl OrdinaryContract {
    pub fn new(num_vertices: u8, num_loops: u8, even_edges: bool, use_triconnected: bool) -> Self {
        OrdinaryContract {
            domain: OrdinaryGVS::new(num_vertices, num_loops, even_edges, use_triconnected),
            target: OrdinaryGVS::new(num_vertices - 1, num_loops, even_edges, use_triconnected),
            num_loops,
            num_vertices,
            even_edges,
            use_triconnected,
        }
    }

    pub fn get_matrix_file_path(&self) -> String {
        format!(
            "data/ordinary/{}{}/contractD{}_{}.txt",
            if self.use_triconnected { "tri/" } else { "" },
            get_type_string(self.even_edges),
            self.num_vertices,
            self.num_loops
        )
    }

    pub fn get_ref_matrix_file_path(&self) -> String {
        if self.use_triconnected {
            format!(
                "data/ordinary/tri/ref/{}/contractD_tri{}_{}.txt",
                get_type_string(self.even_edges),
                self.num_vertices,
                self.num_loops
            )
        } else {
            format!(
                "data/ordinary/ref/{}/contractD{}_{}.txt",
                get_type_string(self.even_edges),
                self.num_vertices,
                self.num_loops
            )
        }
    }

    pub fn is_valid(&self) -> bool {
        self.domain.is_valid()
            && self.target.is_valid()
            && self.domain.num_vertices == self.num_vertices
            && self.target.num_vertices == self.num_vertices - 1
            && self.domain.num_loops == self.num_loops
            && self.target.num_loops == self.num_loops
    }

    pub fn to_string(&self) -> String {
        format!(
            "OrdinaryContract({}, {}, {})",
            self.num_vertices,
            self.num_loops,
            get_type_string(self.even_edges)
        )
    }

    pub fn build_matrix(&self, ignore_existing_files: bool, compression_level: i32) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_valid() {
            return Ok(());
        }

        let matrix_path = self.get_matrix_file_path();
        // println!("{}", get_type_string(self.even_edges));
        // println!("{}", self.even_edges);
        println!("The matrix path is {}", matrix_path);
        println!("The input basis file is {}", self.domain.get_basis_file_path());
        println!("The output basis file is {}", self.target.get_basis_file_path());
        if !ignore_existing_files && std::path::Path::new(&matrix_path).exists() {
            println!("Matrix file already exists: {}", matrix_path);
            return Ok(());
        }

        let in_basis = self.domain.get_basis_g6()?;
        let out_basis = self.target.get_basis_g6()?;
        
        println!("Building output basis index map...");
        let out_basis_map = make_basis_dict(&out_basis);
        
        let num_rows = in_basis.len();
        let num_cols = out_basis.len();

        println!("Number of input basis elements: {}", num_rows);
        println!("Number of output basis elements: {}", num_cols);
        println!("Contracting....");

        // let bar = get_progress_bar(num_rows);

        let matrix_rows: Vec<FxHashMap<usize, i32>> = in_basis.par_iter()
            .progress_with_style(get_progress_bar_style())
            .map(|g6| {
                let mut local: FxHashMap<usize, i32> = FxHashMap::default();
                let g = Graph::from_g6(g6);
                let contractions = g.get_contractions_with_sign(self.even_edges);

                for (g1, sign) in contractions {
                    let (g1s, sign2) = g1.to_canon_g6_sgn(self.even_edges);
                    let val = sign * sign2;

                    if let Some(&col) = out_basis_map.get(&g1s) {
                        *local.entry(col).or_insert(0) += val;
                    }
                }

                local
            })
            .collect();

        // let mut matrix: FxHashMap<(usize, usize), i32> = FxHashMap::with_capacity_and_hasher(
        //     matrix_rows.iter().map(|m| m.len()).sum(),
        //     Default::default(),
        // );

        // for row_map in matrix_rows {
        //     for (k, v) in row_map {
        //         // no += needed; rows are disjoint
        //         matrix.insert(k, v);
        //     }
        // }
        // bar.finish();

        println!("Saving matrix to file: {}", matrix_path);
        save_matrix_to_sms_file(&matrix_rows, num_cols, &matrix_path, compression_level)?;
        println!("Done.");
        
        Ok(())
    }

    pub fn test_matrix_vs_ref(&self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_valid() {
            println!("Matrix check void for {}", self.to_string());
            return Ok(());
        }
        test_matrix_vs_reference(
            &self.get_matrix_file_path(),
            &self.get_ref_matrix_file_path(),
            &self.domain.get_basis_file_path(),
            &self.target.get_basis_file_path(),
            &self.domain.get_basis_ref_file_path(),
            &self.target.get_basis_ref_file_path(),
            self.even_edges,
        )
    }
    
}



pub fn test_matrix_vs_reference(
    mat_file: &str,
    ref_file: &str,
    domain_basis_file: &str,
    target_basis_file: &str,
    domain_ref_file: &str,
    target_ref_file: &str,
    even_edges: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Checking matrix correctness {} ...", mat_file);
    
    let (ref_matrix, nrows_ref, ncols_ref) = load_matrix_from_sms_file(ref_file)?;
    let (matrix, nrows, ncols) = load_matrix_from_sms_file(mat_file)?;
    
    if nrows != nrows_ref || ncols != ncols_ref {
        println!(
            "Matrix dimensions are different: {}x{} vs {}x{}",
            nrows, ncols, nrows_ref, ncols_ref
        );
        return Err("Matrix dimensions do not match".into());
    }
    
    if matrix.len() != ref_matrix.len() {
        println!(
            "Matrix number of entries are different: {} vs {}",
            matrix.len(),
            ref_matrix.len()
        );
        return Err("Matrix number of entries do not match".into());
    }
    
    // before comparing entries, we have to account for possibly different basis orderings.
    // load the domain and tagert basis and the reference basis. canonize the reference basis and find the permutation
    // then correct the matrix indices (rows and columns) accorind to the row- and column permutations
    // get the domain and target basis
    let in_basis = load_g6_file(domain_basis_file)?;
    let out_basis = load_g6_file(target_basis_file)?;
    let in_basis_map = make_basis_dict(&in_basis);
    let out_basis_map = make_basis_dict(&out_basis);
    
    let mut in_basis_ref = load_g6_file(domain_ref_file)?;
    let mut out_basis_ref = load_g6_file(target_ref_file)?;
    
    let mut in_basis_ref_sgn = vec![0i32; in_basis_ref.len()];
    for i in 0..in_basis_ref.len() {
        let g = Graph::from_g6(&in_basis_ref[i]);
        let (g1s, sgn) = g.to_canon_g6_sgn(even_edges);
        in_basis_ref[i] = g1s;
        in_basis_ref_sgn[i] = sgn;

        // sanity checks
        if g.has_odd_automorphism(even_edges) {
            println!("Reference graph has odd automorphism: {}", g.to_g6());
            return Err("Reference graph has odd automorphism".into());
        }
    }
    
    let mut out_basis_ref_sgn = vec![0i32; out_basis_ref.len()];
    for i in 0..out_basis_ref.len() {
        let g = Graph::from_g6(&out_basis_ref[i]);
        let (g1s, sgn) = g.to_canon_g6_sgn(even_edges);
        out_basis_ref[i] = g1s;
        out_basis_ref_sgn[i] = sgn;
        if g.has_odd_automorphism(even_edges) {
            println!("Reference graph has odd automorphism: {}", g.to_g6());
            return Err("Reference graph has odd automorphism".into());
        }
    }
    
    let mut in_perm = vec![0usize; in_basis_ref.len()];
    for i in 0..in_basis_ref.len() {
        if let Some(&idx) = in_basis_map.get(&in_basis_ref[i]) {
            in_perm[i] = idx;
        } else {
            println!("Error: {} not found in domain basis", in_basis_ref[i]);
            return Err(format!("Reference graph {} not found in domain basis", in_basis_ref[i]).into());
        }
    }
    
    let mut out_perm = vec![0usize; out_basis_ref.len()];
    for i in 0..out_basis_ref.len() {
        if let Some(&idx) = out_basis_map.get(&out_basis_ref[i]) {
            out_perm[i] = idx;
        } else {
            println!("Error: {} not found in target basis", out_basis_ref[i]);
            return Err(format!("Reference graph {} not found in target basis", out_basis_ref[i]).into());
        }
    }
    
    let mut matrix2: std::collections::BTreeMap<(usize, usize), i32> = std::collections::BTreeMap::new();
    for ((row, col), value) in ref_matrix.iter() {
        let new_row = in_perm[*row];
        let new_col = out_perm[*col];
        let corrected_value = value * in_basis_ref_sgn[*row] * out_basis_ref_sgn[*col];
        matrix2.insert((new_row, new_col), corrected_value);
    }
    
    for ((key_row, key_col), value) in matrix.iter() {
        if let Some(&ref_value) = matrix2.get(&(*key_row, *key_col)) {
            if ref_value != *value {
                println!(
                    "Entry {} {} differs: {} vs {}",
                    key_row, key_col, ref_value, value
                );
                println!(
                    "g6 code {} -> {} value: {}",
                    in_basis[*key_row], out_basis[*key_col], value
                );
                return Err(format!("Matrix entry ({}, {}) differs: {} vs {}", key_row, key_col, ref_value, value).into());
            }
        } else {
            println!("Entry {} {} not found in ref matrix", key_row, key_col);
            println!(
                "g6 code {} -> {} value: {}",
                in_basis[*key_row], out_basis[*key_col], value
            );
            return Err(format!("Matrix entry ({}, {}) not found in ref matrix", key_row, key_col).into());
        }
    }
    
    println!("Matrix check completed.");
    Ok(())
}

pub fn test_basis_vs_reference(basis_file: &str, ref_file: &str, even_edges: bool, check_automorphisms: bool, ref_hdr_in_file : bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Checking basis correctness {}...", basis_file);

    // Check both files exist (GC basis might be compressed)
    if !std::path::Path::new(basis_file).exists() && !std::path::Path::new(&(basis_file.to_string() + ZSTD_EXTENSION)).exists() {
        println!("Basis file does not exist: {}", basis_file);
        return Err("Basis file does not exist".into());
    }
    if !std::path::Path::new(ref_file).exists() {
        println!("Reference file does not exist: {}", ref_file);
        return Err("Reference file does not exist".into());
    }
    
    let mut g6s = load_g6_file(basis_file)?;
    let mut ref_g6s = if ref_hdr_in_file {
        load_g6_file(ref_file)?
    } else {
        load_g6_file_nohdr(ref_file)?
    };
    
    // First check: length
    if g6s.len() != ref_g6s.len() {
        println!("Number of graphs differ: {} vs {}", g6s.len(), ref_g6s.len());
        return Err("Number of graphs differ".into());
    } else {
        println!("Number of graphs match: {}", g6s.len());
    }

    // Re-canonize reference basis
    for i in 0..ref_g6s.len() {
        let g = Graph::from_g6(&ref_g6s[i]);
        ref_g6s[i] = g.to_canon_g6();
        
        if check_automorphisms && g.has_odd_automorphism(even_edges) {
            println!("Reference graph has odd automorphism: {}", g.to_g6());
            return Err("Reference graph has odd automorphism".into());
        }
    }

    // For now: also re-canonize basis to be safe (in case of different canonization methods or bugs in canonization). In the future, we might want to store the basis in canonized form to avoid this step.
    for i in 0..g6s.len() {
        let g = Graph::from_g6(&g6s[i]);
        g6s[i] = g.to_canon_g6();
        
        if check_automorphisms && g.has_odd_automorphism(even_edges) {
            println!("Basis graph has odd automorphism: {}", g.to_g6());
            return Err("Basis graph has odd automorphism".into());
        }
    }
    
    // Check whether entries are the same
    ref_g6s.par_sort();
    let g6slen = g6s.len();
    let g6s_set: std::collections::HashSet<_> = g6s.into_iter().collect();
    let ref_g6s_set: std::collections::HashSet<_> = ref_g6s.iter().cloned().collect();
    
    // Check whether graph sets have same size as list (otherwise we have duplicates)
    if g6s_set.len() != g6slen {
        println!("Duplicates found in basis: {} unique vs {} total", g6s_set.len(), g6slen);
    }
    if ref_g6s.len() != ref_g6s_set.len() {
        println!("Duplicates found in reference: {} unique vs {} total", ref_g6s_set.len(), ref_g6s.len());
    }

    let diff: std::collections::HashSet<_> = g6s_set.difference(&ref_g6s_set).cloned().collect();
    if !diff.is_empty() {
        println!("The following {} graphs are in the basis but not in the reference:", diff.len());
        for g6 in &diff {
            println!("{}", g6);
        }
        // return Err("Basis contains graphs not in reference".into());
    } else {
        println!("All graphs in the basis are in the reference");
    }
    
    let diff: std::collections::HashSet<_> = ref_g6s_set.difference(&g6s_set).cloned().collect();
    if !diff.is_empty() {
        println!("The following {} graphs are in the reference but not in the basis:", diff.len());
        for g6 in &diff {
            println!("{}", g6);
        }
        return Err("Reference contains graphs not in basis".into());
    } else {
        println!("All graphs in the reference are in the basis");
    }
    
    Ok(())
}