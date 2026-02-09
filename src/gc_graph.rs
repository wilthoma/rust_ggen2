use std::fs::File;
use std::io::{BufRead, BufReader};
use indicatif::ProgressBar;
use std::io::Write;


fn permutation_sign<T: Ord>(p: &[T]) -> i32 {
    let mut sign = 1;
    for i in 0..p.len() {
        for j in (i + 1)..p.len() {
            if p[i] > p[j] {
                sign *= -1;
            }
        }
    }
    sign
}

fn inverse_permutation(p: &[u8]) -> Vec<u8> {
    let mut inv = vec![0; p.len()];
    for i in 0..p.len() {
        inv[p[i] as usize] = i as u8;
    }
    inv
}

fn print_perm(p: &[u8]) {
    for &val in p {
        print!("{} ", val);
    }
    println!();
}

fn permute_to_left(u: u8, v: u8, n: u8) -> Vec<u8> {
    let mut p = vec![0; n as usize];
    p[0] = u;
    p[1] = v;
    let mut idx = 2;
    for j in 0..n {
        if j == u || j == v {
            continue;
        }
        p[idx] = j;
        idx += 1;
    }
    inverse_permutation(&p)
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
            panic!("Invalid graph6 string");
        }
        
        let n = if first >= 63 && first <= 126 {
            first - 63
        } else {
            panic!("Only supports n ≤ 62")
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

    fn load_from_file(filename: &str) -> Result<Vec<String>, std::io::Error> {
        
        let file = File::open(filename)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Failed to open file for reading"))?;
        let mut reader = BufReader::new(file);
        
        let mut first_line = String::new();
        reader.read_line(&mut first_line)?;
        let num_graphs: usize = first_line.trim().parse()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Invalid number in first line"))?;
        
        let mut g6_list = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.is_empty() {
                g6_list.push(line);
            }
        }
        
        if g6_list.len() != num_graphs {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Number of graphs does not match"));
        }
        
        Ok(g6_list)
    }

    fn load_from_file_nohdr(filename: &str) -> Result<Vec<String>, std::io::Error> {
        
        let file = File::open(filename)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Failed to open file for reading"))?;
        let reader = BufReader::new(file);
        
        let g6_list = reader.lines()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|line| !line.is_empty())
            .collect();
        
        Ok(g6_list)
    }


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
        // Placeholder implementation
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
        Graph::load_from_file(&self.get_basis_file_path())
    }

    pub fn build_basis(&self, ignore_existing_files: bool) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_valid() {
            return Ok(());
        }

        let basis_path = self.get_basis_file_path();
        if !ignore_existing_files && std::path::Path::new(&basis_path).exists() {
            println!("Basis file already exists: {}", basis_path);
            return Ok(());
        }

        let infile = self.get_input_file_path();
        let in_g6s = Graph::load_from_file_nohdr(&infile)?;
        let num_graphs = in_g6s.len();

        if in_g6s.is_empty() {
            eprintln!("No graphs found in input file: {}", infile);
            return Ok(());
        }

        println!("Building basis for {}", basis_path);

        let bar = ProgressBar::new(num_graphs as u64);
        bar.set_style(indicatif::ProgressStyle::default_bar()
            .template("[{bar:50.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"));

        let mut out_g6s = Vec::new();
        for g6 in in_g6s {
            let g = Graph::from_g6(&g6);
            if (self.use_triconnected && g.is_triconnected()) || 
               (!self.use_triconnected && g.is_biconnected(None)) {
                if !g.has_odd_automorphism(self.even_edges) {
                    out_g6s.push(g.to_g6());
                }
            }
            bar.inc(1);
        }
        bar.finish();

        println!("Found {} graphs in the basis.", out_g6s.len());
        println!("Done.");

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
        )?;
        Ok(())
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
            "data/ordinary/{}contractD{}_{}.txt",
            if self.use_triconnected { "tri/" } else { "" },
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

    pub fn build_matrix(&self, ignore_existing_files: bool) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_valid() {
            return Ok(());
        }

        let matrix_path = self.get_matrix_file_path();
        if !ignore_existing_files && std::path::Path::new(&matrix_path).exists() {
            println!("Matrix file already exists: {}", matrix_path);
            return Ok(());
        }

        let in_basis = self.domain.get_basis_g6()?;
        let out_basis = self.target.get_basis_g6()?;
        
        let mut in_basis_map = std::collections::HashMap::new();
        for (i, g6) in in_basis.iter().enumerate() {
            in_basis_map.insert(g6.clone(), i);
        }
        
        let mut out_basis_map = std::collections::HashMap::new();
        for (i, g6) in out_basis.iter().enumerate() {
            out_basis_map.insert(g6.clone(), i);
        }
        
        let mut matrix: std::collections::BTreeMap<(usize, usize), i32> = std::collections::BTreeMap::new();
        let num_rows = in_basis.len();
        let num_cols = out_basis.len();

        println!("Number of input basis elements: {}", num_rows);
        println!("Number of output basis elements: {}", num_cols);
        println!("Contracting....");

        let bar = ProgressBar::new(num_rows as u64);
        bar.set_style(indicatif::ProgressStyle::default_bar()
            .template("[{bar:50.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"));

        for g6 in in_basis {
            let g = Graph::from_g6(&g6);
            let contractions = g.get_contractions_with_sign(self.even_edges);
            
            for (g1, sign) in contractions {
                let g1s = g1.to_g6();
                let sign2 = 1i32;
                let val = sign * sign2;
                
                if let Some(&col) = out_basis_map.get(&g1s) {
                    let row = in_basis_map[&g6];
                    *matrix.entry((row, col)).or_insert(0) += val;
                }
            }
            bar.inc(1);
        }
        bar.finish();

        println!("Saving matrix to file: {}", matrix_path);
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
        )?;
        Ok(())
    }
    
}


fn ensure_folder_of_filename_exists(filename: &str) -> std::io::Result<()> {
    if let Some(pos) = filename.rfind(|c| c == '/' || c == '\\') {
        let folder = &filename[..pos];
        if !std::path::Path::new(folder).exists() {
            std::fs::create_dir_all(folder)?;
        }
    }
    Ok(())
}

fn load_matrix_from_sms_file(filename: &str) -> std::io::Result<(std::collections::BTreeMap<(usize, usize), i32>, usize, usize)> {
    
    let file = std::fs::File::open(filename)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to open file for reading {}", filename)))?;
    let reader = std::io::BufReader::new(file);
    let mut lines = reader.lines();
    
    let first_line = lines.next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Empty file"))?
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Failed to read first line"))?;
    
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let nrows: usize = parts[0].parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Invalid nrows"))?;
    let ncols: usize = parts[1].parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Invalid ncols"))?;
    
    let mut matrix = std::collections::BTreeMap::new();
    
    for line in lines {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            continue;
        }
        
        let row: usize = parts[0].parse()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Invalid row"))?;
        let col: usize = parts[1].parse()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Invalid col"))?;
        let val: i32 = parts[2].parse()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Invalid value"))?;
        
        if row == 0 && col == 0 && val == 0 {
            break;
        }
        if row == 0 || col == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, 
                format!("Invalid row or column index (namely 0) in SMS file: {}", filename)));
        }
        
        matrix.insert((row - 1, col - 1), val);
    }
    
    Ok((matrix, nrows, ncols))
}

fn save_matrix_to_sms_file(matrix: &std::collections::BTreeMap<(usize, usize), i32>, nrows: usize, ncols: usize, filename: &str) -> std::io::Result<()> {
    ensure_folder_of_filename_exists(filename)?;
    
    let mut file = std::fs::File::create(filename)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Failed to open file for writing"))?;
    
    writeln!(file, "{} {} {}", nrows, ncols, matrix.len())?;
    
    for ((row, col), value) in matrix.iter() {
        if *value == 0 {
            continue;
        }
        writeln!(file, "{} {} {}", row + 1, col + 1, value)?;
    }
    
    writeln!(file, "0 0 0")?;
    Ok(())
}

fn make_basis_dict(basis: &[String]) -> std::collections::HashMap<String, usize> {
    basis.iter().enumerate()
        .map(|(i, g6)| (g6.clone(), i))
        .collect()
}

fn test_matrix_vs_reference(
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
        return Ok(());
    }
    
    if matrix.len() != ref_matrix.len() {
        println!(
            "Matrix number of entries are different: {} vs {}",
            matrix.len(),
            ref_matrix.len()
        );
    }
    
    let in_basis = Graph::load_from_file(domain_basis_file)?;
    let out_basis = Graph::load_from_file(target_basis_file)?;
    let in_basis_map = make_basis_dict(&in_basis);
    let out_basis_map = make_basis_dict(&out_basis);
    
    let mut in_basis_ref = Graph::load_from_file(domain_ref_file)?;
    let mut out_basis_ref = Graph::load_from_file(target_ref_file)?;
    
    let mut in_basis_ref_sgn = vec![0i32; in_basis_ref.len()];
    for i in 0..in_basis_ref.len() {
        let g = Graph::from_g6(&in_basis_ref[i]);
        if g.has_odd_automorphism(even_edges) {
            println!("Reference graph has odd automorphism: {}", g.to_g6());
        }
        in_basis_ref_sgn[i] = 1;
    }
    
    let mut out_basis_ref_sgn = vec![0i32; out_basis_ref.len()];
    for i in 0..out_basis_ref.len() {
        let g = Graph::from_g6(&out_basis_ref[i]);
        if g.has_odd_automorphism(even_edges) {
            println!("Reference graph has odd automorphism: {}", g.to_g6());
        }
        out_basis_ref_sgn[i] = 1;
    }
    
    let mut in_perm = vec![0usize; in_basis_ref.len()];
    for i in 0..in_basis_ref.len() {
        if let Some(&idx) = in_basis_map.get(&in_basis_ref[i]) {
            in_perm[i] = idx;
        } else {
            println!("Error: {} not found in domain basis", in_basis_ref[i]);
        }
    }
    
    let mut out_perm = vec![0usize; out_basis_ref.len()];
    for i in 0..out_basis_ref.len() {
        if let Some(&idx) = out_basis_map.get(&out_basis_ref[i]) {
            out_perm[i] = idx;
        } else {
            println!("Error: {} not found in target basis", out_basis_ref[i]);
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
            }
        } else {
            println!("Entry {} {} not found in ref matrix", key_row, key_col);
            println!(
                "g6 code {} -> {} value: {}",
                in_basis[*key_row], out_basis[*key_col], value
            );
        }
    }
    
    println!("Matrix check completed.");
    Ok(())
}

fn test_basis_vs_reference(basis_file: &str, ref_file: &str, even_edges: bool, check_automorphisms: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Checking basis correctness {}...", basis_file);
    
    let g6s = Graph::load_from_file(basis_file)?;
    let mut ref_g6s = Graph::load_from_file(ref_file)?;
    
    // Re-canonize reference basis
    for i in 0..ref_g6s.len() {
        let g = Graph::from_g6(&ref_g6s[i]);
        // Note: to_canon_g6() not implemented, using to_g6() instead
        ref_g6s[i] = g.to_g6();
        
        if check_automorphisms && g.has_odd_automorphism(even_edges) {
            println!("Reference graph has odd automorphism: {}", g.to_g6());
        }
    }
    
    // Check whether entries are the same
    ref_g6s.sort();
    let g6s_set: std::collections::HashSet<_> = g6s.into_iter().collect();
    let ref_g6s_set: std::collections::HashSet<_> = ref_g6s.iter().cloned().collect();
    
    let diff: std::collections::HashSet<_> = g6s_set.difference(&ref_g6s_set).cloned().collect();
    if !diff.is_empty() {
        println!("The following graphs are in the basis but not in the reference:");
        for g6 in &diff {
            println!("{}", g6);
        }
    } else {
        println!("All graphs in the basis are in the reference");
    }
    
    let diff: std::collections::HashSet<_> = ref_g6s_set.difference(&g6s_set).cloned().collect();
    if !diff.is_empty() {
        println!("The following graphs are in the reference but not in the basis:");
        for g6 in &diff {
            println!("{}", g6);
        }
    } else {
        println!("All graphs in the reference are in the basis");
    }
    
    Ok(())
}