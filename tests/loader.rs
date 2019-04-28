#[macro_use] extern crate maplit;
extern crate libflate;

use std::error;
use std::path::Path;
use std::io;
use std::io::{BufReader, BufWriter};
use std::fs;
use std::fs::File;
use libflate::zlib;
use image::ImageFormat;
use tfrecord_rs::loader::{LoaderOptions, LoaderMethod, IndexedLoader, Loader, SeqLoader};
use tfrecord_rs::iter::DsIterator;

#[test]
fn seq_loader_single_file_test() -> Result<(), Box<error::Error>>
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
fn indexed_loader_record_iter_test() -> Result<(), Box<error::Error>>
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
fn indexed_loader_index_iter_test() -> Result<(), Box<error::Error>>
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

#[test]
fn parse_example_test() -> Result<(), Box<error::Error>>
{
    // Prepare file
    let out_path = Path::new("./test_files/parse_example_test.tfrecords");
    {
        let in_file = BufReader::new(File::open("./test_files/mnist.train.zlib.tfrecords")?);
        let mut out_file = BufWriter::new(File::create(out_path)?);
        let mut zlib_dec = zlib::Decoder::new(in_file)?;
        io::copy(&mut zlib_dec, &mut out_file)?;
    }

    let loader = SeqLoader::load(out_path)?;
    let record_cnt = loader
        .into_tf_example(None)
        .unwrap_ok()
        .fold(0, |mut cnt, val| {
            let mut keys: Vec<_> = val.keys().into_iter().collect();
            keys.sort();
            assert!(keys.len() == 2 && keys[0] == "image_raw" && keys[1] == "label");

            cnt += 1;
            cnt
        });

    assert!(record_cnt == 60000);

    fs::remove_file(out_path)?;
    Ok(())
}


// TODO: We need proper small dataset for this test
// #[test]
// fn decode_image_test() -> Result<(), Box<error::Error>>
// {
//     // Prepare file
//     let out_path = Path::new("./test_files/decode_image_test.tfrecords");
//     {
//         let in_file = BufReader::new(File::open("./test_files/cifar10.test.zlib.tfrecords")?);
//         let mut out_file = BufWriter::new(File::create(out_path)?);
//         let mut zlib_dec = zlib::Decoder::new(in_file)?;
//         io::copy(&mut zlib_dec, &mut out_file)?;
//     }

//     let loader = SeqLoader::load(out_path)?;
//     let record_cnt = loader
//         .into_tf_example(None)
//         .unwrap_ok()
//         .decode_image(hashmap!("frames".to_owned() => None))
//         .unwrap_ok()
//         .fold(0, |mut cnt, val| {
//             if let tfrecord_rs::iter::Feature::ImageList(images) = &val["frames"]
//             {
//                 cnt += 1;
//                 cnt
//             }
//         });

//     assert!(record_cnt == 60000);

//     fs::remove_file(out_path)?;
//     Ok(())
// }

// #[test]
// fn into_torch_tensor_test() -> Result<(), Box<error::Error>>
// {
//     // Prepare file
//     let out_path = Path::new("./test_files/into_torch_tensor_test.tfrecords");
//     let in_file = BufReader::new(File::open("./test_files/mnist.train.zlib.tfrecords")?);
//     let mut out_file = BufWriter::new(File::create(out_path)?);
//     let mut zlib_dec = zlib::Decoder::new(in_file)?;
//     io::copy(&mut zlib_dec, &mut out_file)?;
//     drop(out_file);

//     let loader = SeqLoader::load_ex(out_path)?;
//     let record_cnt = loader
//         .into_tf_example(None)
//         .unwrap_ok()
//         .into_torch_tensor()
//         .unwrap_ok()
//         .fold(0, |mut cnt, val| {
//             let mut keys: Vec<_> = val.keys().into_iter().collect();
//             keys.sort();
//             assert!(keys.len() == 2 && keys[0] == "image_raw" && keys[1] == "label");

//             cnt += 1;
//             cnt
//         });

//     assert!(record_cnt == 60000);

//     fs::remove_file(out_path)?;
//     Ok(())
// }

// #[test]
// fn wtf() -> Result<(), Box<error::Error>>
// {
//     // Prepare file
//     let out_path = Path::new("./test_files/wtf.tfrecords");
//     {
//         let in_file = BufReader::new(File::open("./test_files/mnist.train.zlib.tfrecords")?);
//         let mut out_file = BufWriter::new(File::create(out_path)?);
//         let mut zlib_dec = zlib::Decoder::new(in_file)?;
//         io::copy(&mut zlib_dec, &mut out_file)?;
//     }

//     let loader = IndexedLoader::load(out_path)?;
//     let record_cnt = loader.index_iter()
//         .cycle()
//         .load_by_tfrecord_index(loader)
//         .unwrap_ok()
//         .fold(0, |mut cnt, val| {
//             assert!(val.len() == 3178);
//             cnt += 1;
//             cnt
//         });


//     fs::remove_file(out_path)?;
//     Ok(())
// }
