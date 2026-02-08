use std::fs::File;
use std::io::{BufRead, BufReader};


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

struct Graph {
    num_vertices: u8,
    edges: Vec<Edge>,
}

impl Graph {
    fn new(n: u8) -> Self {
        Graph {
            num_vertices: n,
            edges: Vec::new(),
        }
    }

    fn with_edges(n: u8, edges: Vec<Edge>) -> Self {
        Graph {
            num_vertices: n,
            edges,
        }
    }

    fn add_edge(&mut self, u: u8, v: u8, data: i32) {
        let (u, v) = if u < v { (u, v) } else { (v, u) };
        self.edges.push(Edge::new(u, v, data));
    }

    fn to_g6(&self) -> String {
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

    fn from_g6(g6: &str) -> Self {
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

    fn get_neighbors(&self, v: u8) -> std::collections::HashSet<u8> {
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
            let mut g1 = self.clone();
            g1.relabel(&pp);
            g1.number_edges();
            let prev_size = g1.edges.len();
            g1 = &g1.contract_edge(0);
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
}
