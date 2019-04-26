# tfrecord-rs

This repository provides TFRecord file reader implemented in Rust. **The project is under development.** Use it with caution.

## Get Started

```ini
[dependencies]
tfrecord_rs = { git = "https://github.com/jerry73204/tfrecord-rs.git" }
```

## Get Started

`SeqLoader::load()` accepts path to .tfrecords file and returns an raw record iterator. We call `into_tf_example()` to convert raw records (in type `Vec<u8>`) to structed examples.

```rust
use tfrecord_rs::loader::SeqLoader;
use tfrecord_rs::iter::DsIterator;
use tfrecord_rs::iter::Feature;

fn main()
{
    let loader = SeqLoader::load("/path/to/your.tfrecords")?;
    let data: Vec<_> = loader
        .into_tf_example(None)
        .unwrap_ok()
        .map(|example| {
            match &example["feature_name"]
            {
                Feature::F32List(floats) => {
                    // ...
                }
                Feature::I64List(integers) => {
                    // ...
                }
                _ => {
                    // ...
                }
            }
        })
        .collect();
}
```

The loader also accepts a path to directory containing .tfrecords files, path objects, and set of paths.

```rust
SeqLoader::load("/path/to/dir")?;
SeqLoader::load(Path::new("/path/to/your.tfrecords"))?;
SeqLoader::load(vec!["file1.tfrecords", "file1.tfrecords"])?;

```

If you need random access capability, the `IndexedLoader` builds the record indexes for each .tfrecord file, and call `index_iter()` to produce indexes for whatever manipulations. The later `load_by_tfrecord_index()` consumes indexes and looks up the records.


```rust
use tfrecord_rs::loader::SeqLoader;
use tfrecord_rs::iter::DsIterator;
use tfrecord_rs::iter::Feature;

fn main()
{
    let loader = IndexedLoader::load("/tfrecord/dir")?;
    let record_cnt = loader
        .index_iter()
        .load_by_tfrecord_index(loader)
        .unwrap_ok()
        .into_tf_example(None)
        .unwrap_ok()
        .map(|example| {
            match &example["feature_name"]
            {
                Feature::F32List(floats) => {
                    // ...
                }
                Feature::I64List(integers) => {
                    // ...
                }
                _ => {
                    // ...
                }
            }
        })
        .collect();
}
```

## TODOs

- Random access loader (done)
- Example iterator (done)
- Integration with tch-rs (partial)
- Integration with tensorflow (not yet)
- Push to crates.io or some (not yet)
