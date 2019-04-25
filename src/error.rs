use std::{io, fmt, error};

// fn

pub fn make_checksum_error(expect_cksum: u32, true_cksum: u32) -> io::Error
{
    io::Error::new(io::ErrorKind::Other, format!("Checksum mismatch: expect checksum {}, but get {}", expect_cksum, true_cksum))
}

pub fn make_truncated_error() -> io::Error
{
    io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated record")
}

pub fn make_corrupted_error() -> io::Error
{
    io::Error::new(io::ErrorKind::Other, format!("TFRecord data is malformed"))
}

pub fn make_loading_error() -> io::Error
{
    io::Error::new(io::ErrorKind::Other, format!("Failed to load record data. Is the record index corrupted or the file is unreadable?"))
}

// struct

#[derive(Debug, Clone)]
pub struct ParseError
{
    desc: String,
}

// impl

impl ParseError
{
    pub fn new(desc: &str) -> ParseError
    {
        ParseError {
            desc: format!("Parsing error: {}", desc)
        }
    }
}

impl error::Error for ParseError
{
    fn description(&self) -> &str {
        &self.desc
    }
}

impl fmt::Display for ParseError
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}", self.desc)
    }
}
