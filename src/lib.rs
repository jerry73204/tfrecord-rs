extern crate protobuf;
extern crate crc;
extern crate memmap;
extern crate tensorflow;
extern crate tch;
extern crate glob;
extern crate rayon;
extern crate lru;
extern crate crossbeam;
extern crate mozjpeg;
#[macro_use] extern crate failure;
#[macro_use] extern crate log;

mod from_tf;
pub mod loader;
pub mod iter;
pub mod parse;
pub mod error;
pub mod utils;
pub mod convert;

use std::any::Any;
use std::collections::HashMap;

pub type FeatureType = Box<dyn Any + Send>;
pub type ExampleType = HashMap<String, FeatureType>;
