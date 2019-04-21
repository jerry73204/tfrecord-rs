use std::io;
use std::path;
use std::fs;
use std::error;
use crc::crc32;
use byteorder::{ReadBytesExt, LittleEndian};

fn checksum(buf: &[u8]) -> u32 {
    let cksum = crc32::checksum_castagnoli(buf);
    ((cksum >> 15) | (cksum << 17)).wrapping_add(0xa282ead8u32)
}

fn try_read_len<R>(reader: &mut R, check_integrity: bool) -> Result<Option<u64>, Box<error::Error>> where
    R: io::Read
{
    let mut len_buf = [0u8; 8];

    match reader.read(&mut len_buf)
    {
        Ok(0) => Ok(None),
        Ok(n) if n == len_buf.len() => {
            let len = (&len_buf[..]).read_u64::<LittleEndian>()?;

            if check_integrity
            {
                let answer_cksum = reader.read_u32::<LittleEndian>()?;
                if answer_cksum == checksum(&len_buf)
                {
                    Ok(Some(len))
                }
                else
                {
                    Err(Box::new(make_corrupted_error()))
                }
            }
            else
            {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                Ok(Some(len))
            }
        }
        Ok(_) => Err(Box::new(make_truncated_error())),
        Err(e) => Err(Box::new(e)),
    }
}

fn try_read_record<R>(reader: &mut R, len: usize, check_integrity: bool) -> Result<Vec<u8>, Box<error::Error>> where
    R: io::Read
{
    let mut buf = Vec::<u8>::new();
    buf.resize(len, 0);
    reader.read_exact(&mut buf)?;
    let answer_cksum = reader.read_u32::<LittleEndian>()?;

    if check_integrity && answer_cksum != checksum(&buf.as_slice())
    {
        return Err(Box::new(make_corrupted_error()));
    }

    Ok(buf)
}

fn make_corrupted_error() -> io::Error
{
    io::Error::new(io::ErrorKind::Other, "Corrupted record")
}

fn make_truncated_error() -> io::Error
{
    io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated record")
}


pub struct SequentialRecordLoader<R: io::Read>
{
    check_integrity: bool,
    pub reader: R,
}

impl<R> SequentialRecordLoader<R> where
    R: io::Read
{

}

impl<R> Iterator for SequentialRecordLoader<R> where
    R: io::Read
{
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item>
    {
        if let Some(len) = try_read_len(&mut self.reader, self.check_integrity).unwrap()
        {
            let record = try_read_record(&mut self.reader, len as usize, self.check_integrity).unwrap();
            Some(record)
        }
        else
        {
            None
        }
    }
}

impl<R> From<(R, bool)> for SequentialRecordLoader<R> where
    R: io::Read
{
    fn from(reader_option: (R, bool)) -> SequentialRecordLoader<R>
    {
        let (reader, check_integrity) = reader_option;
        SequentialRecordLoader {
            check_integrity,
            reader,
        }
    }
}

impl<R> From<R> for SequentialRecordLoader<R> where
    R: io::Read
{
    fn from(reader: R) -> SequentialRecordLoader<R>
    {
        SequentialRecordLoader::from((reader, true))
    }
}

impl From<&str> for SequentialRecordLoader<fs::File>
{
    fn from(path: &str) -> SequentialRecordLoader<fs::File>
    {
        let reader = fs::File::open(path).unwrap();
        SequentialRecordLoader::from(reader)
    }
}

impl From<(&str, bool)> for SequentialRecordLoader<fs::File>
{
    fn from(read_option: (&str, bool)) -> SequentialRecordLoader<fs::File>
    {
        let (path, check_integrity) = read_option;
        let reader = fs::File::open(path).unwrap();
        SequentialRecordLoader::from((reader, check_integrity))
    }
}

impl From<&path::Path> for SequentialRecordLoader<fs::File>
{
    fn from(path: &path::Path) -> SequentialRecordLoader<fs::File>
    {
        let reader = fs::File::open(path).unwrap();
        SequentialRecordLoader::from(reader)
    }
}

impl From<(&path::Path, bool)> for SequentialRecordLoader<fs::File>
{
    fn from(read_option: (&path::Path, bool)) -> SequentialRecordLoader<fs::File>
    {
        let (path, check_integrity) = read_option;
        let reader = fs::File::open(path).unwrap();
        SequentialRecordLoader::from((reader, check_integrity))
    }
}
