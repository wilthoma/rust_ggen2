use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Mutex;
use once_cell::sync::Lazy;

// === Constants ===
const MAXN: usize = 128;
const MAXGEN: usize = 1024;

const WORDSIZE: usize = 64;
const MAXM: usize = (MAXN + WORDSIZE - 1) / WORDSIZE;

type Graph = [u64; MAXM];

// === FFI declarations ===
#[repr(C)]
struct OptionStruct {
    getcanon: c_int,
    digraph: c_int,
    defaultptn: c_int,
    userautomproc: Option<extern "C" fn(c_int, *mut c_int, c_int)>,
    writemarkers: c_int,
    cartesian: c_int,
    invarproc: Option<unsafe extern "C" fn()>,
}

#[repr(C)]
struct StatsStruct {
    grpsize1: f64,
    grpsize2: f64,
    numorbits: c_int,
    numgenerators: c_int,
    errstatus: c_int,
}

unsafe extern "C" {
    fn stringtograph(s: *const c_char, g: *mut Graph, m: *mut c_int, n: *mut c_int);
    fn densenauty(
        g: *mut Graph,
        lab: *mut c_int,
        ptn: *mut c_int,
        orbits: *mut c_int,
        options: *mut OptionStruct,
        stats: *mut StatsStruct,
        m: c_int,
        n: c_int,
        canon: *mut Graph,
    );
}

// === Safe global store for generators ===
// static GEN_STORE: Lazy<Mutex<Vec<Vec<i32>>>> = Lazy::new(|| {
//     Mutex::new(Vec::with_capacity(MAXGEN))
// });
static GEN_STORE: Lazy<Mutex<Vec<Vec<i32>>>> = Lazy::new(|| {
    Mutex::new(Vec::new())
});

/// Callback called by nauty for each generator
/// 
// #[no_mangle]
pub extern "C" fn record_generator2(n: c_int, perm: *mut c_int, _nerr: c_int) {
    unsafe {
        let slice = std::slice::from_raw_parts(perm, n as usize);
        GEN_STORE.lock().unwrap().push(slice.to_vec());
    }
}

/// Call this to compute automorphism group and print generators
pub fn compute_automorphisms(g6: &str) {
    let g6_c = CString::new(g6).unwrap();
    let mut g = [0u64; MAXM];
    let mut m = 0;
    let mut n = 0;

    unsafe {
        stringtograph(g6_c.as_ptr(), &mut g, &mut m, &mut n);
    }

    let mut lab = vec![0; n as usize];
    let mut ptn = vec![0; n as usize];
    let mut orbits = vec![0; n as usize];

    let mut options = OptionStruct {
        getcanon: 0,
        digraph: 0,
        defaultptn: 1,
        userautomproc: Some(record_generator2),
        writemarkers: 0,
        cartesian: 0,
        invarproc: None,
    };

    let mut stats = StatsStruct {
        grpsize1: 0.0,
        grpsize2: 0.0,
        numorbits: 0,
        numgenerators: 0,
        errstatus: 0,
    };

    let mut canon = [0u64; MAXM];

    // Clear any previous generators
    GEN_STORE.lock().unwrap().clear();

    unsafe {
        densenauty(
            &mut g,
            lab.as_mut_ptr(),
            ptn.as_mut_ptr(),
            orbits.as_mut_ptr(),
            &mut options,
            &mut stats,
            m,
            n,
            &mut canon,
        );
    }

    println!("Automorphism group size ≈ {} × 10^{}", stats.grpsize1, stats.grpsize2);
    println!("Number of generators: {}", stats.numgenerators);
    println!("Orbits: {:?}", &orbits[..n as usize]);

    for (i, gene) in GEN_STORE.lock().unwrap().iter().enumerate() {
        println!("Generator {}: {:?}", i + 1, gene);
    }
}

/// Retrieve and clear generators for later use
pub fn take_generators() -> Vec<Vec<i32>> {
    let mut store = GEN_STORE.lock().unwrap();
    let result = store.clone();
    store.clear();
    result
}
