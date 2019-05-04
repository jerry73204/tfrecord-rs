#[macro_use] extern crate maplit;
extern crate libflate;

use std::path::Path;
use std::error::Error;
use std::io::{self, BufReader, BufWriter};
use std::fs::{self, File};
use libflate::zlib;
use tfrecord_rs::loader::{LoaderOptions, LoaderMethod, IndexedLoader, Loader, SeqLoader};
use tfrecord_rs::iter::DsIterator;

#[test]
fn seq_loader_single_file_test() -> Result<(), Box<dyn Error>>
{
    // Prepare file
    let out_path = Path::new("./test_files/seq_loader_single_file_test.tfrecords");
    {
        let in_file = BufReader::new(File::open("./test_files/mnist.train.zlib.tfrecords")?);
        let mut out_file = BufWriter::new(File::create(out_path)?);
        let mut zlib_dec = zlib::Decoder::new(in_file)?;
        io::copy(&mut zlib_dec, &mut out_file)?;
    }

    let loader = SeqLoader::load(out_path)?;
    let record_cnt = loader
        .fold(0, |mut cnt, val| {
            assert!(val.len() == 3178);
            cnt += 1;
            cnt
        });

    assert!(record_cnt == 60000);

    fs::remove_file(out_path)?;
    Ok(())
}

#[test]
fn indexed_loader_record_iter_test() -> Result<(), Box<dyn Error>>
{
    // Prepare file
    let out_path = Path::new("./test_files/indexed_loader_record_iter_test.tfrecords");
    {
        let in_file = BufReader::new(File::open("./test_files/mnist.train.zlib.tfrecords")?);
        let mut out_file = BufWriter::new(File::create(out_path)?);
        let mut zlib_dec = zlib::Decoder::new(in_file)?;
        io::copy(&mut zlib_dec, &mut out_file)?;
    }

    let loader = IndexedLoader::load(out_path)?;
    let record_cnt = loader.into_record_iter()
        .fold(0, |mut cnt, val| {
            assert!(val.len() == 3178);
            cnt += 1;
            cnt
        });

    assert!(record_cnt == 60000);

    fs::remove_file(out_path)?;
    Ok(())
}

#[test]
fn indexed_loader_index_iter_test() -> Result<(), Box<dyn Error>>
{
    // Prepare file
    let out_path = Path::new("./test_files/indexed_loader_index_iter_test.tfrecords");
    {
        let in_file = BufReader::new(File::open("./test_files/mnist.train.zlib.tfrecords")?);
        let mut out_file = BufWriter::new(File::create(out_path)?);
        let mut zlib_dec = zlib::Decoder::new(in_file)?;
        io::copy(&mut zlib_dec, &mut out_file)?;
    }

    let loader = IndexedLoader::load(out_path)?;
    let record_cnt = loader.index_iter()
        .shuffle(10)
        .load_by_tfrecord_index(loader)
        .unwrap_ok()
        .fold(0, |mut cnt, val| {
            assert!(val.len() == 3178);
            cnt += 1;
            cnt
        });

    assert!(record_cnt == 60000);

    fs::remove_file(out_path)?;
    Ok(())
}
