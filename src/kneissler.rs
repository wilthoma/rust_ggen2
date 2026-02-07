use itertools::Itertools;

use crate::graphs::*;

pub fn barrel_graph(k: usize, p: &[usize]) -> Graph {
    // assumes Graph::new(n: usize) and Graph::add_edge(u: usize, v: usize)
    let k = k as u8;
    let mut g = Graph::new(2 * k);

    // generate rims of barrel
    for j in 0..k {
        g.add_edge(j, (j + 1) % k);
        g.add_edge(k + j, k + (j + 1) % k);
    }

    // generate spokes
    g.add_edge(k - 1, 2 * k - 1);
    for (i, &j) in p.iter().enumerate() {
        g.add_edge(i as u8, k + j as u8);
    }

    g
}


pub fn all_barrel_graphs(k: usize) -> impl Iterator<Item = Graph> {
    (0..k - 1)
        .permutations(k - 1)
        .map(move |p| barrel_graph(k, &p))
}

// pub fn tbarrel_graph(k: usize, p: &[usize]) -> Graph {
//     let mut g = barrel_graph(k, p);
//     g.merge_vertices(&[k - 2, k - 1]);
//     g.relabel((0..g.order()).collect::<Vec<_>>(), true);
//     g
// }
pub fn tbarrel_graph(k: usize, p: &[usize]) -> Graph {
    let k = k as u8;
    let mut g = Graph::new(2 * k - 1);
    // one rim of length k
    for j in 0..k {
        g.add_edge(j, (j + 1) % k);
    }
    // ... the other rim of length k - 1
    for j in 0..k - 1 {
        g.add_edge(k + j, k + (j + 1) % (k - 1));
    }
    g.add_edge(k - 1, 2 * k - 2);
    
    for (i, &j) in p.iter().enumerate() {
        g.add_edge(i as u8, k + j as u8);
    }
    g
}

pub fn all_tbarrel_graphs(k: usize) -> impl Iterator<Item = Graph> {
    (0..k - 1)
        .permutations(k - 1)
        .map(move |p| tbarrel_graph(k, &p))
}

pub fn xtbarrel_graph(k: usize, p: &[usize]) -> Graph {
    let k = k as u8;
    let mut g = Graph::new(2 * k - 1);
    for j in 0..k - 1 {
        g.add_edge(j, (j + 1) % (k - 1));
        g.add_edge(k + j, k + (j + 1) % (k - 1));
    }
    g.add_edge(k - 1, 2 * k - 2);
    g.add_edge(k - 1, k - 2);
    for (i, &j) in p.iter().enumerate() {
        if j as u8 + 2 < k  {
            g.add_edge(i as u8, k + j as u8);
        } else {
            g.add_edge(i as u8, k - 1);
        }
    }
    g
}

pub fn all_xtbarrel_graphs(k: usize) -> impl Iterator<Item = Graph> {
    (0..k - 1)
        .permutations(k - 1)
        .map(move |p| xtbarrel_graph(k, &p))
}

pub fn triangle_graph(k: usize, p: &[usize]) -> Graph {
    let k = k as u8;
    let mut g = Graph::new(2 * k);
    for j in 0..k {
        g.add_edge(j, (j + 1) % k);
    }
    for j in 0..k - 1 {
        g.add_edge(k + 1 + j, k + 1 + (j + 1) % (k - 1));
    }
    g.add_edge(k - 1, k);
    g.add_edge(k, 2 * k - 1);
    for (i, &j) in p.iter().enumerate() {
        if j  as u8 +2  < k {
            g.add_edge(i as u8, k + 1 + j as u8);
        } else {
            g.add_edge(i as u8, k);
        }
    }
    g
}

pub fn all_triangle_graphs(k: usize) -> impl Iterator<Item = Graph> {
    (0..k - 1)
        .permutations(k - 1)
        .map(move |p| triangle_graph(k, &p))
}

pub fn hgraph(k: usize, p: &[usize]) -> Graph {
    let k = k as u8;
    let mut g = Graph::new(2 * k);
    for j in 0..k - 1 {
        g.add_edge(j, (j + 1) % (k - 1));
        g.add_edge(k + 1 + j, k + 1 + (j + 1) % (k - 1));
    }
    g.add_edge(k - 2, k - 1);
    g.add_edge(k - 1, 2 * k - 1);
    g.add_edge(k - 1, k);
    for (i, &j) in p.iter().enumerate() {
        if i  as u8 + 2 < k {
            g.add_edge(i as u8, k + j as u8);
        } else if j > 0 {
            g.add_edge(k, k + j as u8);
        }
    }
    g
}

pub fn all_hgraph_graphs(k: usize) -> impl Iterator<Item = Graph> {
    (0..k - 1)
        .permutations(k - 1)
        .filter(move |p| p[k - 2] > 0)
        .map(move |p| hgraph(k, &p))
}

pub fn kneissler_filename(nloops: usize, ntype: usize) -> String {
    format!("data/kneissler_{}_{}.g6", nloops, ntype)
}

pub fn compute_all_kneissler_graphs(nloops : usize, ntype: usize) {
    let fname = kneissler_filename(nloops, ntype);
    let k = nloops - 1;
    if ntype == 0 {
        let gs = all_barrel_graphs(k).map(|g| g.to_g6()).collect::<Vec<_>>();
        let gs2 = canonicalize_and_dedup_g6(&gs, "labelg").unwrap().into_iter().collect::<Vec<_>>();
        Graph::save_to_file(&gs2, &fname).unwrap();
    } else if ntype == 1 {
        let gs = all_tbarrel_graphs(k)
            .chain(all_xtbarrel_graphs(k))
            .map(|g| g.to_g6()).collect::<Vec<_>>();
        let gs2 = canonicalize_and_dedup_g6(&gs, "labelg").unwrap().into_iter().collect::<Vec<_>>();
        Graph::save_to_file(&gs2, &fname).unwrap();
    } else if ntype == 2 {
        let gs = all_barrel_graphs(k)
            .chain(all_triangle_graphs(k))
            .chain(all_hgraph_graphs(k))
            .map(|g| g.to_g6()).collect::<Vec<_>>();
        let gs2 = canonicalize_and_dedup_g6(&gs, "labelg").unwrap().into_iter().collect::<Vec<_>>();
        Graph::save_to_file(&gs2, &fname).unwrap();
    } else if ntype == 3 {
        // load the type 0 and type 2 graphs from file
        let fname0 = kneissler_filename(nloops, 0);
        let fname2 = kneissler_filename(nloops, 2);
        let gs0 : Vec<String> = Graph::load_from_file(&fname0).unwrap();
        let mut gs2 : Vec<String> = Graph::load_from_file(&fname2).unwrap();
        gs2.retain(|g| !gs0.contains(g));
        Graph::save_to_file(&gs2, &fname).unwrap();
    } else {
        panic!("Unknown graph type");
    }
    
}