use failure::Fail;

#[derive(Debug, Fail)]
#[fail(display = "Parse error: {}", desc)]
pub struct ParseError {
    pub desc: String
}

#[derive(Debug, Fail)]
#[fail(display = "Item named \"{}\" not found", name)]
pub struct ItemNotFoundError {
    pub name: String
}

#[derive(Debug, Fail)]
#[fail(display = "Unsupported image format")]
pub struct UnsuportedImageFormatError;

#[derive(Debug, Fail)]
#[fail(display = "Unsupported value type on key {}", key_name)]
pub struct UnsuportedValueTypeError {
    pub key_name: String,
}

#[derive(Debug, Fail)]
#[fail(display = "Value type is inconsistent on key {}", key_name)]
pub struct InconsistentValueTypeError {
    pub key_name: String,
}

#[derive(Debug, Fail)]
#[fail(display = "Checksum mismatched, expect {} but found {}", expect, found)]
pub struct ChecksumMismatchError {
    pub expect: String,
    pub found: String,
}

#[derive(Debug, Fail)]
#[fail(display = "Invalid record index")]
pub struct InvalidRecordIndexError;

#[derive(Debug, Fail)]
#[fail(display = "Corrupted record")]
pub struct CorruptedRecordError;
