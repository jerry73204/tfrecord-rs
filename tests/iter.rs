#[macro_use] extern crate maplit;
extern crate libflate;
extern crate par_map;

use std::path::Path;
use std::error::Error;
use std::io::{self, BufReader, BufWriter};
use std::fs::{self, File};
use libflate::zlib;
use ndarray::Array3;
use par_map::ParMap;
use tfrecord_rs::loader::{LoaderOptions, LoaderMethod, IndexedLoader, Loader, SeqLoader};
use tfrecord_rs::iter::DsIterator;
use tfrecord_rs::utils::{bytes_to_example, decode_image_on_example, example_to_torch_tensor};

#[test]
fn parse_example_test() -> Result<(), Box<dyn Error>>
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
        .map(|record| bytes_to_example(&record, None))
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

#[test]
fn decode_image_test() -> Result<(), Box<dyn Error>>
{
    // Prepare file
    // The tfrecord is published DeepMind's GQN dataset
    let path = Path::new("./test_files/rooms_free_camera_with_object_rotations.tfrecords");

    let loader = SeqLoader::load(path)?;
    let record_cnt = loader
        .map(|record| bytes_to_example(&record, None))
        .unwrap_result()
        .map(|example| decode_image_on_example(example, Some(hashmap!("frames" => None))))
        .unwrap_result()
        .fold(0, |mut cnt, example| {
            let keys: Vec<_> = example.keys().into_iter().collect();
            assert!(keys.len() == 2);
            example["frames"].downcast_ref::<Vec<Array3<u8>>>().unwrap();

            cnt += 1;
            cnt
        });
    assert!(record_cnt == 100);
    Ok(())
}

#[test]
fn parallel_decode_image_test() -> Result<(), Box<dyn Error>>
{
    // Prepare file
    // The tfrecord is published DeepMind's GQN dataset
    let path = Path::new("./test_files/rooms_free_camera_with_object_rotations.tfrecords");

    let loader = SeqLoader::load(path)?;
    let record_cnt = loader
        .map(|record| bytes_to_example(&record, None))
        .unwrap_result()
        .par_map(|example| decode_image_on_example(example, Some(hashmap!("frames" => None))))
        .unwrap_result()
        .fold(0, |mut cnt, example| {
            let keys: Vec<_> = example.keys().into_iter().collect();
            assert!(keys.len() == 2);
            example["frames"].downcast_ref::<Vec<Array3<u8>>>().unwrap();

            cnt += 1;
            cnt
        });
    assert!(record_cnt == 100);
    Ok(())
}

#[test]
fn torch_tensor_test() -> Result<(), Box<dyn Error>>
{
    // Prepare file
    let path = Path::new("./test_files/rooms_free_camera_with_object_rotations.tfrecords");
    let loader = SeqLoader::load(path)?;
    let record_cnt = loader
        .map(|record| bytes_to_example(&record, None))
        .unwrap_result()
        .map(|example| decode_image_on_example(example, Some(hashmap!("frames" => None))))
        .unwrap_result()
        .map(|example| example_to_torch_tensor(example, None, tch::Device::Cpu))
        .unwrap_result()
        .fold(0, |mut cnt, example| {
            let mut keys: Vec<_> = example.keys().into_iter().collect();
            keys.sort();
            assert!(keys.len() == 2 && keys[0] == "cameras" && keys[1] == "frames");

            example["cameras"].downcast_ref::<tch::Tensor>().unwrap();
            example["frames"].downcast_ref::<Vec<tch::Tensor>>().unwrap();

            cnt += 1;
            cnt
        });
    assert!(record_cnt == 100);
    Ok(())
}

#[test]
fn parse_event_test() -> Result<(), Box<dyn Error>>
{
    // Prepare file
    let path = Path::new("/path/to/file.tfevents.*");
    let loader = SeqLoader::load(path)?;
    let record_cnt = loader
        .map(|record| tfrecord_rs::parser::parse_event(&record))
        .unwrap_result()
        .fold(0, |mut cnt, event| {
            cnt += 1;
            cnt
        });

    dbg!(record_cnt);
    Ok(())
}
