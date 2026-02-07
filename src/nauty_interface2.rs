use nauty_Traces_sys::*;
use std::io::{self, Write};
use std::os::raw::c_int;
// use libc;

pub fn mainy() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = optionblk::default();
    options.writeautoms = TRUE;
    let mut stats = statsblk::default();

    loop {
        print!("\nenter n : ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let n = input.trim().parse()?;
        if n > 0 {

            let m = SETWORDSNEEDED(n);

            unsafe {
                nauty_check(WORDSIZE as c_int, m as c_int, n as c_int, NAUTYVERSIONID as c_int);
            }

            let mut lab = vec![0; n];
            let mut ptn = vec![0; n];
            let mut orbits = vec![0; n];

            let mut g = empty_graph(m, n);
            for v in 0..n {
                ADDONEEDGE(&mut g, v, (v + 1) % n, m);
            }

            println!("Generators for Aut(C[{}]):", n);

            unsafe {
                densenauty(
                    g.as_mut_ptr(),
                    lab.as_mut_ptr(),
                    ptn.as_mut_ptr(),
                    orbits.as_mut_ptr(),
                    &mut options,
                    &mut stats,
                    m as c_int,
                    n as c_int,
                    std::ptr::null_mut()
                );
            }

            print!("[");
            for orbit in orbits {
                print!("{} ", orbit)
            }
            println!("]");

            print!("order = ");
            // io::stdout().flush().unwrap();
                // writegroupsize(libc::stderr, stats.grpsize1, stats.grpsize2);
                // writegroupsize(stderr, stats.grpsize1, stats.grpsize2);
            // }
            println!();
        } else {
            break;
        }
    }
    Ok(())
}