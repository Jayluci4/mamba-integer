use std::collections::HashMap;
use std::ffi::{CStr};
use std::os::raw::{c_char, c_int};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use fancy_regex::Regex;

const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^
\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
type Pair = (u32, u32);
pub struct Tokenizer {
    pub merges: HashMap<Pair, u32>,
    pub compiled_pattern: Regex,
}
#[no_mangle]
pub extern "C" fn tokenizer_new() -> *mut Tokenizer {
    Box::into_raw(Box::new(Tokenizer {
        merges: HashMap::new(),
        compiled_pattern: Regex::new(GPT4_PATTERN).unwrap(),
    }))
}
#[no_mangle]
pub extern "C" fn tokenizer_free(ptr: *mut Tokenizer) {
    if !ptr.is_null() { unsafe { drop(Box::from_raw(ptr)); } }
}
#[no_mangle]
pub extern "C" fn tokenizer_train(ptr: *mut Tokenizer, input_path: *const c_char, vocab_size: u32) -> c_int {
    let t = unsafe { &mut *ptr };
    let path = unsafe { CStr::from_ptr(input_path).to_str().unwrap() };
    let file = match File::open(path) { Ok(f) => f, Err(_) => return -1 };
    let reader = BufReader::new(file);
    let mut counts: HashMap<Vec<u32>, i32> = HashMap::new();
    for line in reader.lines() {
        if let Ok(text) = line {
            for m in t.compiled_pattern.find_iter(&text) {
                if let Ok(mat) = m {
                    let ids: Vec<u32> = mat.as_str().bytes().map(|b| b as u32).collect();
                    *counts.entry(ids).or_default() += 1;
                }
            }
        }
    }
    let num_merges = vocab_size - 256;
    for i in 0..num_merges {
        let mut pair_counts: HashMap<Pair, i32> = HashMap::new();
        for (ids, count) in &counts {
            for win in ids.windows(2) {
                *pair_counts.entry((win[0], win[1])).or_default() += count;
            }
        }
        let best_pair = pair_counts.into_iter().max_by_key(|&(_, count)| count);
        if let Some((pair, _)) = best_pair {
            let new_id = 256 + i;
            t.merges.insert(pair, new_id);
            let mut next_counts = HashMap::new();
            for (mut ids, count) in counts {
                let mut next_ids = Vec::new();
                let mut j = 0;
                while j < ids.len() {
                    if j + 1 < ids.len() && (ids[j], ids[j+1]) == pair {
                        next_ids.push(new_id); j += 2;
                    } else {
                        next_ids.push(ids[j]); j += 1;
                    }
                }
                next_counts.insert(next_ids, count);
            }
            counts = next_counts;
        } else { break; }
    }
    0
}
#[no_mangle]
pub extern "C" fn tokenizer_encode(ptr: *mut Tokenizer, text: *const c_char, out_len: *mut usize) -> *mut u32 {
    let t = unsafe { &mut *ptr };
    let s = unsafe { CStr::from_ptr(text).to_str().unwrap() };
    let mut all_ids = Vec::new();
    for m in t.compiled_pattern.find_iter(s) {
        if let Ok(mat) = m {
            let mut ids: Vec<u32> = mat.as_str().bytes().map(|b| b as u32).collect();
            while ids.len() >= 2 {
                let mut best_pair: Option<(usize, u32)> = None;
                for i in 0..ids.len()-1 {
                    let p = (ids[i], ids[i+1]);
                    if let Some(&id) = t.merges.get(&p) {
                        if best_pair.is_none() || id < best_pair.unwrap().1 { best_pair = Some((i, id)); }
                    }
                }
                if let Some((i, id)) = best_pair { ids[i] = id; ids.remove(i+1); } else { break; }
            }
            all_ids.extend(ids);
        }
    }
    unsafe { *out_len = all_ids.len(); }
    let res = all_ids.into_boxed_slice();
    let p = res.as_ptr();
    std::mem::forget(res);
    p as *mut u32
}
#[no_mangle]
pub extern "C" fn tokenizer_free_ids(ptr: *mut u32, len: usize) {
    if !ptr.is_null() { unsafe { drop(Vec::from_raw_parts(ptr, len, len)); } }
}
#[no_mangle]
pub extern "C" fn tokenizer_save(ptr: *mut Tokenizer, path: *const c_char) -> c_int {
    let t = unsafe { &mut *ptr };
    let p = unsafe { CStr::from_ptr(path).to_str().unwrap() };
    let mut f = File::create(p).unwrap();
    for (pair, id) in &t.merges { writeln!(f, "{},{} {}", pair.0, pair.1, id).unwrap(); }
    0
}
#[no_mangle]
pub extern "C" fn tokenizer_load(ptr: *mut Tokenizer, path: *const c_char) -> c_int {
    let t = unsafe { &mut *ptr };
    let p = unsafe { CStr::from_ptr(path).to_str().unwrap() };
    let f = File::open(p).unwrap();
    for line in BufReader::new(f).lines() {
        let l = line.unwrap();
        let parts: Vec<&str> = l.split_whitespace().collect();
        let pp: Vec<&str> = parts[0].split(',').collect();
        t.merges.insert((pp[0].parse().unwrap(), pp[1].parse().unwrap()), parts[1].parse().unwrap());
    }
    0
}