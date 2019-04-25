use std::io;
use std::io::Seek;
use std::io::Read;
use std::iter;
use std::marker;
use std::slice;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::fs;
use std::fs::File;
use std::error;
use std::ops;
use crc::crc32;
use byteorder::{ReadBytesExt, LittleEndian};
use rayon::prelude::*;
use crate::iter::DsIterator;
use crate::error::{make_checksum_error,make_truncated_error};

// Helper functions

fn checksum(buf: &[u8]) -> u32 {
    let cksum = crc32::checksum_castagnoli(buf);
    ((cksum >> 15) | (cksum << 17)).wrapping_add(0xa282ead8u32)
}

fn try_read_len<R>(reader: &mut R, check_integrity: bool) -> Result<Option<u64>, io::Error> where
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
                let len_cksum = checksum(&len_buf);
                if answer_cksum == len_cksum
                {
                    Ok(Some(len))
                }
                else
                {
                    Err(make_checksum_error(answer_cksum, len_cksum))
                }
            }
            else
            {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                Ok(Some(len))
            }
        }
        Ok(_) => Err(make_truncated_error()),
        Err(e) => Err(e),
    }
}

fn try_read_record<R>(reader: &mut R, len: usize, check_integrity: bool) -> Result<Vec<u8>, io::Error> where
    R: io::Read
{
    let mut buf = Vec::<u8>::new();
    buf.resize(len, 0);
    reader.read_exact(&mut buf)?;
    let answer_cksum = reader.read_u32::<LittleEndian>()?;

    if check_integrity
    {
        let record_cksum = checksum(&buf);
        if answer_cksum != record_cksum
        {
            return Err(make_checksum_error(answer_cksum, record_cksum));
        }
    }

    Ok(buf)
}

pub fn build_indexes_from_paths(
    paths: Vec<PathBuf>,
    check_integrity: bool,
    parallel: bool,
) -> Result<Vec<(Arc<PathBuf>, Vec<(usize, usize)>)>, io::Error>
{
    let load_file = |path| -> Result<_, io::Error> {
        let mut file = File::open(&path)?;
        let index = build_index_from_reader(&mut file, check_integrity)?;
        Ok((Arc::new(path), index))
    };

    let indexes = if parallel
    {
        let indexes_result: Result<Vec<_>, io::Error> = paths.into_par_iter()
            .map(load_file)
            .collect();

        let mut indexes = indexes_result?;
        indexes.par_sort_unstable_by(|(left_path, _), (right_path, _)| {
            left_path.cmp(right_path)
        });
        indexes
    }
    else
    {
        let indexes_result: Result<Vec<_>, io::Error> = paths.into_iter()
            .map(load_file)
            .collect();
        let mut indexes = indexes_result?;
        indexes.sort_unstable_by(|(left_path, _), (right_path, _)| {
            left_path.cmp(right_path)
        });
        indexes
    };

    Ok(indexes)
}

pub fn build_index_from_reader<R>(reader: &mut R, check_integrity: bool) -> Result<Vec<(usize, usize)>, io::Error> where
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
                    let record_cksum = checksum(&buf);

                    if answer_cksum != record_cksum
                    {
                        return Err(make_checksum_error(answer_cksum, record_cksum));
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

pub fn build_index_from_buffer(buf: &[u8], check_integrity: bool) -> Result<Vec<(usize, usize)>, io::Error>
{
    let mut index: Vec<(usize, usize)> = Vec::new();
    let mut offset = 0usize;
    let limit = buf.len();
    let len_size = 8;
    let cksum_size = 4;

    while offset < limit
    {
        let len_buf = &buf[offset..(offset + len_size)];
        let len = (&len_buf[..]).read_u64::<LittleEndian>()? as usize;
        offset += len_size;

        if check_integrity
        {
            let cksum_buf = &buf[offset..(offset + cksum_size)];
            let answer_cksum = (&cksum_buf[..]).read_u32::<LittleEndian>()?;
            let len_cksum = checksum(len_buf);

            if answer_cksum != len_cksum
            {
                return Err(make_checksum_error(answer_cksum, len_cksum));
            }
        }
        offset += cksum_size;

        let saved_offset = offset;

        if check_integrity
        {
            let record_buf = &buf[offset..(offset + len)];
            let record_cksum = checksum(record_buf);

            let cksum_buf = &buf[(offset + len)..(offset + len + cksum_size)];
            let answer_cksum = (&cksum_buf[..]).read_u32::<LittleEndian>()?;

            if answer_cksum != record_cksum
            {
                return Err(make_checksum_error(answer_cksum, record_cksum));
            }
        }
        offset += len + cksum_size;

        index.push((saved_offset, len))
    }

    Ok(index)
}

// traits

pub trait Loader<A, L, E>: Sized where
    E: error::Error,
{
    fn load(arg: A) -> Result<L, E>
    {
        Self::load_ex(arg, Default::default())
    }
    fn load_ex(_: A, _: LoaderOptions) -> Result<L, E>;
}

// Type aliases

pub type RecordIndex = (usize, usize);

// structs

#[derive(Clone, Debug)]
pub enum LoaderMethod
{
    Mmap, File,
}

#[derive(Clone)]
pub struct LoaderOptions
{
    pub check_integrity: bool,
    pub auto_close: bool,
    pub parallel: bool,
    // pub num_workers: Option<u64>,
    pub open_limit: Option<usize>,
    pub method: LoaderMethod,
}

enum FileList
{
    MmapLru(lru::LruCache<Arc<PathBuf>, memmap::Mmap>),
    MmapMap(HashMap<Arc<PathBuf>, memmap::Mmap>),
    MmapOnDemand,
    FileLru(lru::LruCache<Arc<PathBuf>, File>),
    FileMap(HashMap<Arc<PathBuf>, File>),
    FileOnDemand,
}

pub struct TFRecordLoader
{
    indexes: Vec<RecordIndex>,
    record_indexes: Vec<(Arc<PathBuf>, Vec<RecordIndex>)>,
    file_list: FileList,
}

pub struct IndexIter
{
    cursor: usize,
    indexes: Vec<RecordIndex>,
}

pub struct RecordIter
{
    loader: TFRecordLoader,
    cursor: usize,
}


// impls

impl Default for LoaderOptions
{
    fn default() -> Self
    {
        LoaderOptions {
            check_integrity: true,
            auto_close: true,
            parallel: true,
            // num_workers: None,
            open_limit: None,
            method: LoaderMethod::Mmap,
        }
    }
}


impl TFRecordLoader
{
    pub fn index_iter(&self) -> IndexIter
    {
        IndexIter {
            cursor: 0,
            indexes: self.indexes.clone()
        }
    }

    pub fn into_record_iter(self) -> RecordIter
    {
        RecordIter {
            cursor: 0,
            loader: self,
        }
    }

    pub fn get_indexes(&self) -> &[RecordIndex]
    {
        &self.indexes
    }

    pub fn fetch(&mut self, index: RecordIndex) -> Option<Vec<u8>>
    {
        let (outer_ind, inner_ind) = index;
        let (path_rc, file_index) = self.record_indexes.get(outer_ind)?;
        let (offset_ref, len_ref) = file_index.get(inner_ind)?;
        let path = Arc::try_unwrap(path_rc.clone()).unwrap();
        let offset = *offset_ref;
        let len = *len_ref;

        let read_from_file = |mut file: &File, off: usize, len: usize| -> Result<Vec<u8>, io::Error> {
            let mut buf = vec![0u8; len];
            file.seek(io::SeekFrom::Start(off as u64))?;
            file.read_exact(&mut buf)?;
            Ok(buf)
        };

        match self.file_list {
            FileList::FileOnDemand => {
                let file = File::open(path).unwrap();
                let buf = read_from_file(&file, offset, len).unwrap();
                Some(buf)
            }
            FileList::FileMap(ref mut file_map) => {
                let buf = match file_map.get(&path) {
                    Some(file) => {
                        read_from_file(file, offset, len).unwrap()
                    },
                    None => {
                        let file = File::open(path).unwrap();
                        let buf = read_from_file(&file, offset, len).unwrap();
                        file_map.insert(path_rc.clone(), file);
                        buf
                    }
                };

                Some(buf)
            }
            FileList::FileLru(ref mut file_lru) => {
                let buf = match file_lru.get(path_rc) {
                    Some(file) => {
                        read_from_file(&file, offset, len).unwrap()
                    }
                    None => {
                        let file = File::open(path).unwrap();
                        let buf = read_from_file(&file, offset, len).unwrap();
                        file_lru.put(path_rc.clone(), file);
                        buf
                    }
                };

                Some(buf)
            }
            FileList::MmapOnDemand => {
                let file = File::open(path).unwrap();
                let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };
                let buf = mmap.get(offset..(offset + len)).unwrap();
                Some(buf.to_vec())
            }
            FileList::MmapMap(ref mut mmap_map) => {
                let buf = match mmap_map.get(&path) {
                    Some(mmap) => {
                        mmap.get(offset..(offset + len)).unwrap().to_vec()
                    },
                    None => {
                        let file = File::open(path).unwrap();
                        let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };
                        let buf = mmap.get(offset..(offset + len)).unwrap().to_vec();
                        mmap_map.insert(path_rc.clone(), mmap);
                        buf
                    }
                };

                Some(buf)
            }
            FileList::MmapLru(ref mut mmap_lru) => {
                let buf = match mmap_lru.get(path_rc) {
                    Some(mmap) => {
                        mmap.get(offset..(offset + len)).unwrap().to_vec()
                    }
                    None => {
                        let file = File::open(path).unwrap();
                        let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };
                        let buf = mmap.get(offset..(offset + len)).unwrap().to_vec();
                        mmap_lru.put(path_rc.clone(), mmap);
                        buf
                    }
                };

                Some(buf)
            }
        }
    }
}


impl Loader<&str, TFRecordLoader, io::Error> for TFRecordLoader
{
    fn load_ex(path_str: &str, options: LoaderOptions) -> Result<TFRecordLoader, io::Error>
    {
        TFRecordLoader::load_ex(path_str.to_owned(), options)
    }
}

impl Loader<String, TFRecordLoader, io::Error> for TFRecordLoader
{
    fn load_ex(path_str: String, options: LoaderOptions) -> Result<TFRecordLoader, io::Error>
    {
        let meta = fs::metadata(&path_str).unwrap();
        if meta.is_dir()
        {
            let paths: Vec<_> = fs::read_dir(&path_str)
                .unwrap()
                .filter_map(|entry_ret| {
                    let entry = entry_ret.unwrap();
                    let meta = entry.metadata().unwrap();
                    let fname = entry.file_name().into_string().unwrap();

                    if meta.is_file() && fname.ends_with(".tfrecord")
                    {
                        Some(entry.path())
                    }
                    else
                    {
                        None
                    }

                }).collect();
            TFRecordLoader::load_ex(paths, options)
        }
        else if meta.is_file()
        {
            let path = PathBuf::from(path_str);
            let path_list = vec![path];
            TFRecordLoader::load_ex(path_list, options)
        }
        else
        {
            panic!("{} is not a file or directory", path_str);
        }
    }
}

impl Loader<&[&Path], TFRecordLoader, io::Error> for TFRecordLoader
{
    fn load_ex(paths: &[&Path], options: LoaderOptions) -> Result<TFRecordLoader, io::Error>
    {
        let cloned_paths: Vec<_> = paths.into_iter().map(|p| p.to_path_buf()).collect();
        TFRecordLoader::load_ex(cloned_paths, options)
    }
}

impl Loader<Vec<PathBuf>, TFRecordLoader, io::Error> for TFRecordLoader
{
    fn load_ex(paths: Vec<PathBuf>, options: LoaderOptions) -> Result<TFRecordLoader, io::Error>
    {
        let record_indexes = build_indexes_from_paths(paths, options.check_integrity, options.parallel).unwrap();

        let file_list = match options.method {
            LoaderMethod::Mmap => {
                match options.open_limit {
                    None => FileList::MmapMap(HashMap::new()),
                    Some(0) => FileList::MmapOnDemand,
                    Some(limit) => FileList::MmapLru(lru::LruCache::new(limit)),
                }
            }
            LoaderMethod::File => {
                match options.open_limit {
                    None => FileList::FileMap(HashMap::new()),
                    Some(0) => FileList::FileOnDemand,
                    Some(limit) => FileList::FileLru(lru::LruCache::new(limit)),
                }
            }
        };

        let mut indexes = Vec::<RecordIndex>::new();

        for (outer_ind, (_, file_index)) in record_indexes.iter().enumerate()
        {
            for inner_ind in 0..(file_index.len())
            {
                indexes.push((outer_ind, inner_ind));
            }
        }

        Ok(
            TFRecordLoader {
                record_indexes,
                indexes,
                file_list,
            }
        )
    }
}

impl Iterator for IndexIter
{
    type Item = RecordIndex;

    fn next(&mut self) -> Option<Self::Item>
    {
        if self.cursor < self.indexes.len()
        {
            let ret = self.indexes[self.cursor];
            self.cursor += 1;
            Some(ret)
        }
        else
        {
            None
        }
    }
}

impl DsIterator for IndexIter {}

impl Iterator for RecordIter
{
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item>
    {
        let indexes = self.loader.get_indexes();
        if self.cursor < indexes.len()
        {
            let index = indexes[self.cursor];
            let record = self.loader.fetch(index).unwrap();
            self.cursor += 1;
            Some(record)
        }
        else
        {
            None
        }
    }
}

impl DsIterator for RecordIter {}
