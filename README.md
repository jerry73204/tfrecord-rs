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
// TODO The example comes soon.
```

The loader also accepts a path to directory containing .tfrecords files, path objects, and set of paths.

```rust
SeqLoader::load("/path/to/dir")?;
SeqLoader::load(Path::new("/path/to/your.tfrecords"))?;
SeqLoader::load(vec!["file1.tfrecords", "file1.tfrecords"])?;
```

If you need random access capability, the `IndexedLoader` builds the record indexes for each .tfrecord file, and call `index_iter()` to produce indexes for whatever manipulations. The later `load_by_tfrecord_index()` consumes indexes and looks up the records.


```rust
// TODO The example comes soon.
```

## TODOs

- Random access loader (done)
- Integration with tch-rs (done)
- Integration with tensorflow (not yet)
- Publish to crates.io (not yet)
- Documention (not yet)
