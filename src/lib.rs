extern crate protobuf;
extern crate crc;
extern crate memmap;
extern crate tensorflow;
extern crate tch;
extern crate glob;
extern crate rayon;
extern crate lru;

mod from_tf;
pub mod loader;
pub mod iter;
pub mod parser;
