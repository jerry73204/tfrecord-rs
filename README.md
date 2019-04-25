# tfrecord-rs

This repository provides TFRecord file reader implemented in Rust. *The project is under development.* Use it with caution.

## Get Started

```ini
[dependencies]
tfrecord_rs = { git = "https://github.com/jerry73204/tfrecord-rs.git" }
```

## Example

*The example is outdated!*

```rust
use tfrecord_rs::FeatureValue;
use tfrecord_rs::loader::SequentialRecordLoader;

fn example_loader() -> Result<(), Box<error::Error>>
{
    // The loader accepts a string, path or objects with std::io::Read trait
    let loader = SequentialRecordLoader::from("/path/to/your.tfrecord");

    for record in loader
    {
        // record is a Vec<u8> binary string.
        // We convert it to an example here.
        let example = tfrecord_rs::parse_single_example(&record)?;

        // example is a HashMap
        // The value can be either FeatureValue::{Bytes(val),Float32(val),Int64(val)}
        for (name, value) in example
        {
            if let FeatureValue::Float32(floats) = value
            {
                // Write your code here. For example, convert it to a tensor.
                println!("{:?}", floats);
            }
            else
            {
                panic!("Unexpected record type");
            }
        }
    }

    Ok(())
}
```

## TODOs

- Random access loader (almost done)
- Example iterator (done)
- Integration with tch-rs (partial)
- Integration with tensorflow (not done)
