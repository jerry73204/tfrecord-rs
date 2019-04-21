use std::io;
use std::path;
use std::fs;
use std::error;
use std::ops;
use crc::crc32;
use byteorder::{ReadBytesExt, LittleEndian};

// Helper functions

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

pub fn build_index_from_reader<R>(reader: &mut R, check_integrity: bool) -> Result<Vec<(usize, usize)>, Box<error::Error>> where
    R: io::Read + io::Seek
{
    let mut index: Vec<(usize, usize)> = Vec::new();

    loop
    {
        match try_read_len(reader, check_integrity)?
        {
            None => break,
            Some(len) => {
                let offset = reader.seek(io::SeekFrom::Current(0))? as usize;
                if check_integrity
                {
                    let mut buf = Vec::<u8>::new();
                    buf.resize(len as usize, 0);
                    reader.read_exact(&mut buf)?;
                    let answer_cksum = reader.read_u32::<LittleEndian>()?;

                    if check_integrity && answer_cksum != checksum(&buf)
                    {
                        return Err(Box::new(make_corrupted_error()));
                    }
                }
                else
                {
                    reader.seek(io::SeekFrom::Current(len as i64 + 4))?;
                }

                index.push((offset, len as usize));
            }
        }
    }

    Ok(index)
}

pub fn build_index_from_buffer(buf: &[u8], check_integrity: bool) -> Result<Vec<(usize, usize)>, Box<error::Error>>
{
    let mut index: Vec<(usize, usize)> = Vec::new();
    let mut offset = 0usize;
    let limit = buf.len();
    let len_size = 8;
    let cksum_size = 4;

    while offset < limit
    {
        let mut len_buf = &buf[offset..(offset + len_size)];
        let len = len_buf.read_u64::<LittleEndian>()? as usize;
        offset += len_size;

        if check_integrity
        {
            let mut cksum_buf = &buf[offset..(offset + cksum_size)];
            let len_cksum = cksum_buf.read_u32::<LittleEndian>()?;
            let answer_cksum = checksum(len_buf);
            if answer_cksum != len_cksum
            {
                return Err(Box::new(make_corrupted_error()));
            }
        }
        offset += cksum_size;

        let saved_offset = offset;

        if check_integrity
        {
            let mut record_buf = &buf[offset..(offset + len)];
            let answer_cksum = record_buf.read_u32::<LittleEndian>()?;

            let mut cksum_buf = &buf[(offset + len)..(offset + len + cksum_size)];
            let record_cksum = cksum_buf.read_u32::<LittleEndian>()?;

            if answer_cksum != record_cksum
            {
                return Err(Box::new(make_corrupted_error()));
            }
        }
        offset += len + cksum_size;

        index.push((saved_offset, len))
    }

    Ok(index)
}

fn make_corrupted_error() -> io::Error
{
    io::Error::new(io::ErrorKind::Other, "Corrupted record")
}

fn make_truncated_error() -> io::Error
{
    io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated record")
}

// Sequential record loader

pub struct SequentialRecordLoader<R: io::Read>
{
    check_integrity: bool,
    pub reader: R,
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


impl From<path::PathBuf> for SequentialRecordLoader<fs::File>
{
    fn from(path: path::PathBuf) -> SequentialRecordLoader<fs::File>
    {
        let reader = fs::File::open(path).unwrap();
        SequentialRecordLoader::from(reader)
    }
}

impl From<(path::PathBuf, bool)> for SequentialRecordLoader<fs::File>
{
    fn from(read_option: (path::PathBuf, bool)) -> SequentialRecordLoader<fs::File>
    {
        let (path, check_integrity) = read_option;
        let reader = fs::File::open(path).unwrap();
        SequentialRecordLoader::from((reader, check_integrity))
    }
}


// Random access loader

pub struct RandomAccessRecordLoader<R: io::Read>
{
    record_index: Vec<(usize, usize)>,
    pub reader: R,
}

impl<R> RandomAccessRecordLoader<R> where
    R: io::Read + io::Seek,
{
    pub fn num_records(&self) -> usize
    {
        self.record_index.len()
    }

    pub fn load_one(&mut self, index: usize) -> Vec<u8>
    {
        let (offset, len) = self.record_index[index];
        self.read_record(offset, len).unwrap()
    }

    pub fn load_many<T>(&mut self, range: T) -> Vec<Vec<u8>> where
        T: ops::RangeBounds<usize>
    {
        let mut result = Vec::<Vec<u8>>::new();

        let start: usize = match range.start_bound() {
            ops::Bound::Included(n) => *n,
            ops::Bound::Excluded(n) => *n + 1,
            ops::Bound::Unbounded => 0,
        };

        let end: usize = match range.end_bound() {
            ops::Bound::Included(n) => *n + 1,
            ops::Bound::Excluded(n) => *n,
            ops::Bound::Unbounded => self.record_index.len(),
        };

        for ind in start..end
        {
            let (offset, len) = self.record_index[ind];
            let record = self.read_record(offset, len).unwrap();
            result.push(record);
        }

        result
    }

    fn read_record(&mut self, offset: usize, len: usize) -> Result<Vec<u8>, Box<error::Error>>
    {
        self.reader.seek(io::SeekFrom::Start(offset as u64))?;
        let mut buf = Vec::<u8>::new();
        buf.resize(len, 0);
        self.reader.read_exact(&mut buf)?;

        Ok(buf)
    }
}

impl<R> From<(R, bool)> for RandomAccessRecordLoader<R> where
    R: io::Read + io::Seek
{
    fn from(reader_option: (R, bool)) -> RandomAccessRecordLoader<R>
    {
        let (mut reader, check_integrity) = reader_option;
        let record_index = build_index_from_reader(&mut reader, check_integrity).unwrap();

        RandomAccessRecordLoader {
            record_index,
            reader,
        }
    }
}

impl<R> From<R> for RandomAccessRecordLoader<R> where
    R: io::Read + io::Seek
{
    fn from(reader: R) -> RandomAccessRecordLoader<R>
    {
        RandomAccessRecordLoader::from((reader, true))
    }
}

impl From<&path::Path> for RandomAccessRecordLoader<fs::File>
{
    fn from(path: &path::Path) -> RandomAccessRecordLoader<fs::File>
    {
        let reader = fs::File::open(path).unwrap();
        RandomAccessRecordLoader::from(reader)
    }
}

impl From<(&path::Path, bool)> for RandomAccessRecordLoader<fs::File>
{
    fn from(read_option: (&path::Path, bool)) -> RandomAccessRecordLoader<fs::File>
    {
        let (path, check_integrity) = read_option;
        let reader = fs::File::open(path).unwrap();
        RandomAccessRecordLoader::from((reader, check_integrity))
    }
}

impl From<path::PathBuf> for RandomAccessRecordLoader<fs::File>
{
    fn from(path: path::PathBuf) -> RandomAccessRecordLoader<fs::File>
    {
        let reader = fs::File::open(path).unwrap();
        RandomAccessRecordLoader::from(reader)
    }
}

impl From<(path::PathBuf, bool)> for RandomAccessRecordLoader<fs::File>
{
    fn from(read_option: (path::PathBuf, bool)) -> RandomAccessRecordLoader<fs::File>
    {
        let (path, check_integrity) = read_option;
        let reader = fs::File::open(path).unwrap();
        RandomAccessRecordLoader::from((reader, check_integrity))
    }
}

// Memory-mapped loader

pub struct MmapRecordLoader
{
    record_index: Vec<(usize, usize)>,
    mmap: memmap::Mmap,
}

impl MmapRecordLoader
{
    pub fn num_records(&self) -> usize
    {
        self.record_index.len()
    }

    pub fn load_one(&mut self, index: usize) -> &[u8]
    {
        let (offset, len) = self.record_index[index];
        &self.mmap[offset..(offset + len)]
    }

    pub fn load_many<T>(&mut self, range: T) -> Vec<&[u8]> where
        T: ops::RangeBounds<usize>
    {
        let mut result = Vec::<&[u8]>::new();

        let start: usize = match range.start_bound() {
            ops::Bound::Included(n) => *n,
            ops::Bound::Excluded(n) => *n + 1,
            ops::Bound::Unbounded => 0,
        };

        let end: usize = match range.end_bound() {
            ops::Bound::Included(n) => *n + 1,
            ops::Bound::Excluded(n) => *n,
            ops::Bound::Unbounded => self.record_index.len(),
        };

        for ind in start..end
        {
            let (offset, len) = self.record_index[ind];
            let record = &self.mmap[offset..(offset + len)];
            result.push(record);
        }

        result
    }
}

impl From<memmap::Mmap> for MmapRecordLoader
{
    fn from(mmap: memmap::Mmap) -> MmapRecordLoader
    {
        MmapRecordLoader::from((mmap, true))
    }
}

impl From<(memmap::Mmap, bool)> for MmapRecordLoader
{
    fn from(read_option: (memmap::Mmap, bool)) -> MmapRecordLoader
    {
        let (mmap, check_integrity) = read_option;
        let record_index = build_index_from_buffer(&mmap, check_integrity).unwrap();

        MmapRecordLoader {
            record_index,
            mmap,
        }
    }
}

impl From<fs::File> for MmapRecordLoader
{
    fn from(file :fs::File) -> MmapRecordLoader
    {
        MmapRecordLoader::from((file, true))
    }
}

impl From<(fs::File, bool)> for MmapRecordLoader
{
    fn from(read_option: (fs::File, bool)) -> MmapRecordLoader
    {
        let (file, check_integrity) = read_option;
        let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };
        MmapRecordLoader::from((mmap, check_integrity))
    }
}

impl From<&path::Path> for MmapRecordLoader
{
    fn from(path: &path::Path) -> MmapRecordLoader
    {
        let file = fs::File::open(path).unwrap();
        MmapRecordLoader::from(file)
    }
}

impl From<(&path::Path, bool)> for MmapRecordLoader
{
    fn from(read_option: (&path::Path, bool)) -> MmapRecordLoader
    {
        let (path, check_integrity) = read_option;
        let file = fs::File::open(path).unwrap();
        MmapRecordLoader::from((file, check_integrity))
    }
}

impl From<path::PathBuf> for MmapRecordLoader
{
    fn from(path: path::PathBuf) -> MmapRecordLoader
    {
        let file = fs::File::open(path).unwrap();
        MmapRecordLoader::from(file)
    }
}

impl From<(path::PathBuf, bool)> for MmapRecordLoader
{
    fn from(read_option: (path::PathBuf, bool)) -> MmapRecordLoader
    {
        let (path, check_integrity) = read_option;
        let file = fs::File::open(path).unwrap();
        MmapRecordLoader::from((file, check_integrity))
    }
}
