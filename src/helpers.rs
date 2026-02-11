// Shared smaller routines used by the other modules, to avoid code duplication.

use core::num;
use std::fs::File;
use std::io::{BufRead, BufReader};
use indicatif::{ProgressBar, ProgressStyle};
use std::io::Write;
use rustc_hash::FxHashMap;
use rayon::prelude::*;


pub fn permutation_sign<T: Ord>(p: &[T]) -> i32 {
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

pub fn inverse_permutation(p: &[u8]) -> Vec<u8> {
    let mut inv = vec![0; p.len()];
    for i in 0..p.len() {
        inv[p[i] as usize] = i as u8;
    }
    inv
}

pub fn print_perm(p: &[u8]) {
    for &val in p {
        print!("{} ", val);
    }
    println!();
}

pub fn permute_to_left(u: u8, v: u8, n: u8) -> Vec<u8> {
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

pub fn load_g6_file(filename: &str) -> std::io::Result<Vec<String>> {
    // check if exists
    if !std::path::Path::new(filename).exists() {
        println!("File does not exist: {}", filename);
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("File not found: {}", filename),
        ));
    }

    let file = std::fs::File::open(filename)?;
    println!("Loading g6 file: {}...", filename);
    let reader = std::io::BufReader::new(file);
    // read first line and trsnform to int
    let mut lines = reader.lines();
    let first_line = lines.next().unwrap()?;
    let num_graphs: usize = first_line.trim().parse().unwrap();
    let mut g6_list = Vec::with_capacity(num_graphs);
    let mut pb: Option<ProgressBar> = None;
    if num_graphs > 1000000 {
        pb = Some(get_progress_bar(num_graphs));
    }
    for line in lines { // .take(num_graphs) {
        let g6 = line?;
        g6_list.push(g6);
        if let Some(ref bar) = pb {
            bar.inc(1);
        }
    }
    if let Some(ref bar) = pb {
        bar.finish();
    }
    if g6_list.len() != num_graphs {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Number of graphs in file does not match the first line",
        ));
    }
    Ok(g6_list)
}

pub fn load_g6_file_nohdr(filename: &str) -> std::io::Result<Vec<String>> {
    let file = std::fs::File::open(filename)?;
    println!("Loading g6 file: {}...", filename);
    let reader = std::io::BufReader::new(file);
    // read first line and trsnform to int
    let lines = reader.lines();
    let mut g6_list = Vec::new();
    for line in lines {
        let g6 = line?;
        g6_list.push(g6);
    }
    Ok(g6_list)
}

pub fn save_g6_file(g6_list: &[String], filename: &str, compression_level: i32) -> std::io::Result<()> {
    println!("Saving g6 file: {}...", filename);
    ensure_folder_of_filename_exists(filename)?;
    let file = std::fs::File::create(filename)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Failed to open file for writing"))?;
    let mut writer = std::io::BufWriter::new(file);
    writeln!(writer, "{}", g6_list.len())?;
    let pb : Option<ProgressBar> = if g6_list.len() > 1000000 {
        Some(get_progress_bar(g6_list.len()))
    } else {
        None
    };
    for g6 in g6_list {
        writeln!(writer, "{}", g6)?;
        if let Some(ref bar) = pb {
            bar.inc(1);
        }
    }
    if let Some(ref bar) = pb {
        bar.finish();
    }
    Ok(())
}

pub fn make_basis_dict(basis: &[String]) -> FxHashMap<String, usize> {
    basis.iter().enumerate()
        .map(|(i, g6)| (g6.clone(), i))
        .collect()
}




pub fn ensure_folder_of_filename_exists(filename: &str) -> std::io::Result<()> {
    if let Some(pos) = filename.rfind(|c| c == '/' || c == '\\') {
        let folder = &filename[..pos];
        if !std::path::Path::new(folder).exists() {
            std::fs::create_dir_all(folder)?;
        }
    }
    Ok(())
}

pub fn load_matrix_from_sms_file(filename: &str) -> std::io::Result<(FxHashMap<(usize, usize), i32>, usize, usize)> {
    
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
    
    let mut matrix = FxHashMap::default();

    let mut pb : Option<ProgressBar> = None;
    let mut cur_row = 0;
    if nrows > 1000000 || ncols > 1000000 {
        pb = Some(get_progress_bar(nrows));
    }
    
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

        if let Some(ref bar) = pb {
            if row != cur_row {
                bar.inc((row - cur_row) as u64);
                cur_row = row;
            }
        }
    }
    
    if let Some(ref bar) = pb {
        bar.finish();
    }

    Ok((matrix, nrows, ncols))
}


pub fn save_matrix_to_sms_file(matrix_rows: &Vec<FxHashMap<usize, i32>>, ncols: usize, filename: &str, compression_level: i32) -> std::io::Result<()> {
    let nrows = matrix_rows.len();
    ensure_folder_of_filename_exists(filename)?;
    println!("Saving matrix to file: {}...", filename);

    let file = std::fs::File::create(filename)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Failed to open file for writing"))?;
    let mut writer = std::io::BufWriter::new(file);

    writeln!(writer, "{} {} {}", nrows, ncols, matrix_rows.iter().map(|m| m.len()).sum::<usize>())?;

    let pb: Option<ProgressBar> = if nrows > 1000000 || ncols > 1000000 {
        Some(get_progress_bar(nrows))
    } else {
        None
    };

    for (row, row_map) in matrix_rows.iter().enumerate() {
        // sort entries by column index
        let mut entries: Vec<(usize, i32)> = row_map.iter().map(|(&col, &value)| (col, value)).collect();
        entries.sort_by_key(|&(col, _)| col);
        for (col, value) in entries {
            if value == 0 {
                continue;
            }
            writeln!(writer, "{} {} {}", row + 1, col + 1, value)?;
        }
        if let Some(ref bar) = pb {
            bar.inc(1);
        }
    }

    if let Some(ref bar) = pb {
        bar.finish();
    }

    writeln!(writer, "0 0 0")?;
    Ok(())
}


pub fn get_progress_bar(total: usize) -> ProgressBar {
let bar = ProgressBar::new(total as u64);
    bar.set_style(
        get_progress_bar_style()
    );
    bar
}

pub fn get_progress_bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) Elapsed: {elapsed_precise} Remaining: {eta_precise}",
    )
    .unwrap()
    .progress_chars("#>-")
}