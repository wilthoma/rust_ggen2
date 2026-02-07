use std::os::raw::c_uint;

unsafe extern "C" {
    fn canonicalize_graph(
        n: c_uint,
        edges: *const c_uint,
        num_edges: c_uint,
        out_permutation: *mut c_uint,
    );
}

pub fn mainz() {
    let n = 5;
    let edges: Vec<u32> = vec![
        0, 1,
        0, 2,
        1, 2,
        1, 3,
        3, 4,
    ];

    let mut perm = vec![0u32; n];
    unsafe {
        canonicalize_graph(n as u32, edges.as_ptr(), (edges.len() / 2) as u32, perm.as_mut_ptr());
    }

    println!("Canonical permutation: {:?}", perm);
}
