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
#[macro_use] extern crate log;

mod from_tf;
pub mod loader;
pub mod iter;
pub mod parser;
pub mod error;
pub mod utils;

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;

pub type FeatureType = Box<dyn Any + Sync + Send>;
pub type ErrorType = Box<dyn Debug + Sync + Send>;
pub type ExampleType = HashMap<String, FeatureType>;

pub type NonSyncFeatureType = Box<dyn Any>;
pub type NonSyncErrorType = Box<dyn Debug>;
pub type NonSyncExampleType = HashMap<String, NonSyncFeatureType>;
