use std::io::{self, Seek, Read, BufReader};
use std::vec;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::Arc;
use std::fs;
use std::fs::File;
use crc::crc32;
use byteorder::{ReadBytesExt, LittleEndian};
use rayon::prelude::*;
use failure::Fallible;
use crate::error::ChecksumMismatchError;

fn checksum(buf: &[u8]) -> u32 {
    let cksum = crc32::checksum_castagnoli(buf);
    ((cksum >> 15) | (cksum << 17)).wrapping_add(0xa282ead8u32)
}

fn try_read_len<R>(reader: &mut R, check_integrity: bool) -> Fallible<Option<u64>> where
    R: io::Read {
    let mut len_buf = [0u8; 8];

    match reader.read(&mut len_buf) {
        Ok(0) => Ok(None),
        Ok(n) if n == len_buf.len() => {
            let len = (&len_buf[..]).read_u64::<LittleEndian>()?;
            debug!("Get record length {}", len);

            if check_integrity {
                let expect_cksum = reader.read_u32::<LittleEndian>()?;
                let found_cksum = checksum(&len_buf);
                if expect_cksum == found_cksum {
                    Ok(Some(len))
                }
                else {
                    Err(ChecksumMismatchError {
                        expect: format!("{:#010x}", expect_cksum),
                        found: format!("{:#010x}", found_cksum),
                    }.into())
                }
            }
            else {

                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                Ok(Some(len))
            }
        }
        Ok(_) => Err(io::Error::new(io::ErrorKind::UnexpectedEof, "File truncated").into()),
        Err(err) => Err(err.into()),
    }
}

fn try_read_record<R>(reader: &mut R, len: usize, check_integrity: bool) -> Fallible<Vec<u8>> where
    R: io::Read + io::Seek {

    let mut buf = Vec::<u8>::new();
    buf.resize(len, 0);
    reader.read_exact(&mut buf)?;
    let expect_cksum = reader.read_u32::<LittleEndian>()?;

    if check_integrity {

        let found_cksum = checksum(&buf);
        if expect_cksum != found_cksum {

            return Err(ChecksumMismatchError {
                expect: format!("{:#010x}", expect_cksum),
                found: format!("{:#010x}", found_cksum),
            }.into());
        }
    }

    Ok(buf)
}

pub fn build_indexes_from_paths(
    paths: Vec<PathBuf>,
    check_integrity: bool,
    parallel: bool,
) -> Fallible<Vec<(Arc<PathBuf>, Vec<(usize, usize)>)>> {

    let load_file = |path: PathBuf| -> Fallible<_> {
        debug!("Loading index on \"{}\"", path.display());
        let mut file = BufReader::new(File::open(&path)?);
        let index = build_index_from_reader(&mut file, check_integrity)?;
        Ok((Arc::new(path), index))
    };

    let indexes = if parallel {
        let indexes_result: Fallible<Vec<_>> = paths.into_par_iter()
            .map(load_file)
            .collect();

        let mut indexes = indexes_result?;
        indexes.par_sort_unstable_by(|(left_path, _), (right_path, _)| {
            left_path.cmp(right_path)
        });
        indexes
    }
    else {

        let indexes_result: Fallible<Vec<_>> = paths.into_iter()
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

pub fn build_index_from_reader<R>(reader: &mut R, check_integrity: bool) -> Fallible<Vec<(usize, usize)>> where
    R: io::Read + io::Seek {

    let mut index: Vec<(usize, usize)> = Vec::new();

    loop {
        match try_read_len(reader, check_integrity)? {
            None => break,
            Some(len) => {
                let offset = reader.seek(io::SeekFrom::Current(0))? as usize;
                if check_integrity {

                    let mut buf = Vec::<u8>::new();
                    buf.resize(len as usize, 0);
                    reader.read_exact(&mut buf)?;
                    let expect_cksum = reader.read_u32::<LittleEndian>()?;
                    let found_cksum = checksum(&buf);

                    if expect_cksum != found_cksum {
                        return Err(ChecksumMismatchError {
                            expect: format!("{:#010x}", expect_cksum),
                            found: format!("{:#010x}", found_cksum),
                        }.into());
                    }
                }
                else {

                    reader.seek(io::SeekFrom::Current(len as i64 + 4))?;
                }

                index.push((offset, len as usize));
            }
        }
    }

    Ok(index)
}

pub fn build_index_from_buffer(buf: &[u8], check_integrity: bool) -> Fallible<Vec<(usize, usize)>> {

    let mut index: Vec<(usize, usize)> = Vec::new();
    let mut offset = 0usize;
    let limit = buf.len();
    let len_size = 8;
    let cksum_size = 4;

    while offset < limit {

        let len_buf = &buf[offset..(offset + len_size)];
        let len = (&len_buf[..]).read_u64::<LittleEndian>()? as usize;
        offset += len_size;

        if check_integrity {

            let cksum_buf = &buf[offset..(offset + cksum_size)];
            let expect_cksum = (&cksum_buf[..]).read_u32::<LittleEndian>()?;
            let found_cksum = checksum(len_buf);

            if expect_cksum != found_cksum {

                return Err(ChecksumMismatchError {
                    expect: format!("{:#010x}", expect_cksum),
                    found: format!("{:#010x}", found_cksum),
                }.into());
            }
        }
        offset += cksum_size;

        let saved_offset = offset;

        if check_integrity {

            let record_buf = &buf[offset..(offset + len)];
            let found_cksum = checksum(record_buf);

            let cksum_buf = &buf[(offset + len)..(offset + len + cksum_size)];
            let expect_cksum = (&cksum_buf[..]).read_u32::<LittleEndian>()?;

            if expect_cksum != found_cksum {
                return Err(ChecksumMismatchError {
                    expect: format!("{:#010x}", expect_cksum),
                    found: format!("{:#010x}", found_cksum),
                }.into());
            }
        }
        offset += len + cksum_size;

        index.push((saved_offset, len))
    }

    Ok(index)
}

// traits

pub trait Loader<A, L>: Sized {

    fn load(arg: A) -> Fallible<L> {

        Self::load_ex(arg, Default::default())
    }
    fn load_ex(_: A, _: LoaderOptions) -> Fallible<L>;
}

// Type aliases

pub type RecordIndex = (usize, usize);

// structs

#[derive(Clone, Debug)]
pub enum LoaderMethod {

    Mmap, File,
}

#[derive(Clone)]
pub struct LoaderOptions {

    pub check_integrity: bool,
    pub auto_close: bool,
    pub parallel: bool,
    // pub num_workers: Option<u64>,
    pub open_limit: Option<usize>,
    pub method: LoaderMethod,
}

enum FileList {

    MmapLru(lru::LruCache<Arc<PathBuf>, memmap::Mmap>),
    MmapMap(HashMap<Arc<PathBuf>, memmap::Mmap>),
    MmapOnDemand,
    FileLru(lru::LruCache<Arc<PathBuf>, BufReader<File>>),
    FileMap(HashMap<Arc<PathBuf>, BufReader<File>>),
    FileOnDemand,
}

pub struct SeqLoader {

    options: LoaderOptions,
    paths_iter: vec::IntoIter<PathBuf>,
    file_opt: Option<fs::File>,
}

pub struct IndexedLoader {

    indexes: Vec<RecordIndex>,
    record_indexes: Vec<(Arc<PathBuf>, Vec<RecordIndex>)>,
    file_list: FileList,
}

#[derive(Clone)]
pub struct IndexIter {

    cursor: usize,
    indexes: Vec<RecordIndex>,
}

#[derive(Clone)]
pub struct IndexRecordIter {

    loader_rc: Arc<IndexedLoader>,
    cursor: usize,
}


// impls

impl Default for LoaderOptions {

    fn default() -> Self {

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


impl IndexedLoader {
    pub fn index_iter(&self) -> IndexIter {

        IndexIter {
            cursor: 0,
            indexes: self.indexes.clone()
        }
    }

    pub fn into_record_iter(self) -> IndexRecordIter {
        IndexRecordIter {
            cursor: 0,
            loader_rc: Arc::new(self),
        }
    }

    pub fn get_indexes(&self) -> &[RecordIndex] {
        &self.indexes
    }

    pub fn fetch(&mut self, index: RecordIndex) -> Option<Vec<u8>> {

        let (outer_ind, inner_ind) = index;
        let (path_rc, file_index) = self.record_indexes.get_mut(outer_ind)?;
        let (offset_ref, len_ref) = file_index.get(inner_ind)?;
        // let mut path = Arc::get_mut(path_rc).unwrap();
        let offset = *offset_ref;
        let len = *len_ref;

        let read_from_file = |file: &mut BufReader<File>, off: usize, len: usize| -> Fallible<Vec<u8>> {
            let mut buf = vec![0u8; len];
            file.seek(io::SeekFrom::Start(off as u64))?;
            file.read_exact(&mut buf)?;
            Ok(buf)
        };

        match self.file_list {
            FileList::FileOnDemand => {
                let path = Arc::get_mut(path_rc).unwrap();
                let mut file = BufReader::new(File::open(path).unwrap());
                let buf = read_from_file(&mut file, offset, len).unwrap();
                Some(buf)
            }
            FileList::FileMap(ref mut file_map) => {
                let buf = match file_map.get_mut(path_rc) {

                    Some(ref mut file) => {
                        read_from_file(file, offset, len).unwrap()
                    },
                    None => {
                        let path = Arc::get_mut(path_rc).unwrap();
                        let mut file = BufReader::new(File::open(path).unwrap());
                        let buf = read_from_file(&mut file, offset, len).unwrap();
                        file_map.insert(path_rc.clone(), file);
                        buf
                    }
                };

                Some(buf)
            }
            FileList::FileLru(ref mut file_lru) => {
                let buf = match file_lru.get_mut(path_rc) {
                    Some(file) => {
                        read_from_file(file, offset, len).unwrap()
                    }
                    None => {
                        let path = Arc::get_mut(path_rc).unwrap();
                        let mut file = BufReader::new(File::open(path).unwrap());
                        let buf = read_from_file(&mut file, offset, len).unwrap();
                        file_lru.put(path_rc.clone(), file);
                        buf
                    }
                };

                Some(buf)
            }
            FileList::MmapOnDemand => {
                let path = Arc::get_mut(path_rc).unwrap();
                let file = File::open(path).unwrap();
                let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };
                let buf = mmap.get(offset..(offset + len)).unwrap();
                Some(buf.to_vec())
            }
            FileList::MmapMap(ref mut mmap_map) => {
                let buf = match mmap_map.get(path_rc) {
                    Some(mmap) => {
                        mmap.get(offset..(offset + len)).unwrap().to_vec()
                    },
                    None => {
                        let path = Arc::get_mut(path_rc).unwrap();
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
                        let path = Arc::get_mut(path_rc).unwrap();
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


impl Loader<&str, IndexedLoader> for IndexedLoader {

    fn load_ex(path_str: &str, options: LoaderOptions) -> Fallible<IndexedLoader> {

        IndexedLoader::load_ex(path_str.to_owned(), options)
    }
}

impl Loader<String, IndexedLoader> for IndexedLoader {

    fn load_ex(path_str: String, options: LoaderOptions) -> Fallible<IndexedLoader> {

        let path = PathBuf::from(path_str);
        IndexedLoader::load_ex(path, options)
    }
}

impl Loader<Vec<&str>, IndexedLoader> for IndexedLoader {

    fn load_ex(path_strs: Vec<&str>, options: LoaderOptions) -> Fallible<IndexedLoader> {

        let paths: Vec<_> = path_strs.into_iter()
            .map(|orig| PathBuf::from(orig))
            .collect();
        IndexedLoader::load_ex(paths, options)
    }
}

impl Loader<Vec<String>, IndexedLoader> for IndexedLoader {

    fn load_ex(path_strs: Vec<String>, options: LoaderOptions) -> Fallible<IndexedLoader> {

        let paths: Vec<_> = path_strs.into_iter()
            .map(|orig| PathBuf::from(orig))
            .collect();
        IndexedLoader::load_ex(paths, options)
    }
}

impl Loader<&[&Path], IndexedLoader> for IndexedLoader {

    fn load_ex(paths: &[&Path], options: LoaderOptions) -> Fallible<IndexedLoader> {

        let cloned_paths: Vec<_> = paths.into_iter().map(|p| p.to_path_buf()).collect();
        IndexedLoader::load_ex(cloned_paths, options)
    }
}

impl Loader<Vec<&Path>, IndexedLoader> for IndexedLoader {

    fn load_ex(paths: Vec<&Path>, options: LoaderOptions) -> Fallible<IndexedLoader> {

        let cloned_paths: Vec<_> = paths.into_iter().map(|p| p.to_path_buf()).collect();
        IndexedLoader::load_ex(cloned_paths, options)
    }
}

impl Loader<&Path, IndexedLoader> for IndexedLoader {

    fn load_ex(path: &Path, options: LoaderOptions) -> Fallible<IndexedLoader> {

        IndexedLoader::load_ex(path.to_owned(), options)
    }
}

impl Loader<PathBuf, IndexedLoader> for IndexedLoader {

    fn load_ex(path: PathBuf, options: LoaderOptions) -> Fallible<IndexedLoader> {

        let meta = fs::metadata(&path).unwrap();
        if meta.is_dir() {

            let paths: Vec<_> = fs::read_dir(&path)
                .unwrap()
                .filter_map(|entry_ret| {
                    let entry = entry_ret.unwrap();
                    let meta = entry.metadata().unwrap();
                    // let fname = entry.file_name().into_string().unwrap();

                    if meta.is_file() {
                        Some(entry.path())
                    }
                    else {
                        None
                    }

                }).collect();
            IndexedLoader::load_ex(paths, options)
        }
        else if meta.is_file() {
            let path_list = vec![path];
            IndexedLoader::load_ex(path_list, options)
        }
        else {
            Err(
                io::Error::new(
                    io::ErrorKind::NotFound,
                    path.display().to_string(),
                ).into()
            )
        }
    }
}

impl Loader<Vec<PathBuf>, IndexedLoader> for IndexedLoader {

    fn load_ex(paths: Vec<PathBuf>, options: LoaderOptions) -> Fallible<IndexedLoader> {

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

        for (outer_ind, (_, file_index)) in record_indexes.iter().enumerate() {

            for inner_ind in 0..(file_index.len()) {

                indexes.push((outer_ind, inner_ind));
            }
        }

        Ok(
            IndexedLoader {
                record_indexes,
                indexes,
                file_list,
            }
        )
    }
}


impl Loader<&str, SeqLoader> for SeqLoader {

    fn load_ex(path_str: &str, options: LoaderOptions) -> Fallible<SeqLoader> {

        SeqLoader::load_ex(path_str.to_owned(), options)
    }
}

impl Loader<String, SeqLoader> for SeqLoader {

    fn load_ex(path_str: String, options: LoaderOptions) -> Fallible<SeqLoader> {

        let path = PathBuf::from(path_str);
        SeqLoader::load_ex(path, options)
    }
}

impl Loader<Vec<&str>, SeqLoader> for SeqLoader {

    fn load_ex(path_strs: Vec<&str>, options: LoaderOptions) -> Fallible<SeqLoader> {

        let paths: Vec<_> = path_strs.into_iter()
            .map(|orig| PathBuf::from(orig))
            .collect();
        SeqLoader::load_ex(paths, options)
    }
}

impl Loader<Vec<String>, SeqLoader> for SeqLoader {

    fn load_ex(path_strs: Vec<String>, options: LoaderOptions) -> Fallible<SeqLoader> {

        let paths: Vec<_> = path_strs.into_iter()
            .map(|orig| PathBuf::from(orig))
            .collect();
        SeqLoader::load_ex(paths, options)
    }
}

impl Loader<&[&Path], SeqLoader> for SeqLoader {

    fn load_ex(paths: &[&Path], options: LoaderOptions) -> Fallible<SeqLoader> {

        let cloned_paths: Vec<_> = paths.into_iter().map(|p| p.to_path_buf()).collect();
        SeqLoader::load_ex(cloned_paths, options)
    }
}

impl Loader<Vec<&Path>, SeqLoader> for SeqLoader {

    fn load_ex(paths: Vec<&Path>, options: LoaderOptions) -> Fallible<SeqLoader> {

        let cloned_paths: Vec<_> = paths.into_iter().map(|p| p.to_path_buf()).collect();
        SeqLoader::load_ex(cloned_paths, options)
    }
}

impl Loader<&Path, SeqLoader> for SeqLoader {

    fn load_ex(path: &Path, options: LoaderOptions) -> Fallible<SeqLoader> {

        SeqLoader::load_ex(path.to_owned(), options)
    }
}

impl Loader<PathBuf, SeqLoader> for SeqLoader {

    fn load_ex(path: PathBuf, options: LoaderOptions) -> Fallible<SeqLoader> {

        let meta = fs::metadata(&path).unwrap();
        if meta.is_dir() {

            let paths: Vec<_> = fs::read_dir(&path)
                .unwrap()
                .filter_map(|entry_ret| {
                    let entry = entry_ret.unwrap();
                    let meta = entry.metadata().unwrap();

                    if meta.is_file() {
                        Some(entry.path())
                    }
                    else {
                        None
                    }

                }).collect();
            SeqLoader::load_ex(paths, options)
        }
        else if meta.is_file() {
            let path_list = vec![path];
            SeqLoader::load_ex(path_list, options)
        }
        else {
            Err(
                io::Error::new(
                    io::ErrorKind::NotFound,
                    path.display().to_string(),
                ).into()
            )
        }
    }
}

impl Loader<Vec<PathBuf>, SeqLoader> for SeqLoader {

    fn load_ex(paths: Vec<PathBuf>, options: LoaderOptions) -> Fallible<SeqLoader> {

        let paths_iter = paths.into_iter();
        let file_opt = None;
        Ok(
            SeqLoader {
                options,
                paths_iter,
                file_opt,
            }
        )
    }
}

impl Iterator for SeqLoader {

    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        let (mut file, len) = loop {
            match &mut self.file_opt {
                None => {
                    match self.paths_iter.next() {
                        None => {
                            debug!("SeqLoader finished");
                            return None;
                        }
                        Some(path) => {
                            debug!("Opening \"{}\"", path.display());
                            self.file_opt = Some(fs::File::open(path).unwrap());
                            continue
                        }
                    }
                }
                Some(ref mut file) => {
                    match try_read_len(file, self.options.check_integrity).unwrap() {
                        Some(len) => break (file, len),
                        None => {
                            debug!("Reach EOF and close file");
                            self.file_opt = None;
                            continue
                        }
                    }
                }
            }
        };

        let record = try_read_record(
            &mut file,
            len as usize,
            self.options.check_integrity
        ).unwrap();
        Some(record)
    }
}

impl Iterator for IndexIter {
    type Item = RecordIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.indexes.len() {
            let ret = self.indexes[self.cursor];
            self.cursor += 1;
            Some(ret)
        }
        else {

            None
        }
    }
}

impl Iterator for IndexRecordIter {

    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        let loader = Arc::get_mut(&mut self.loader_rc).unwrap();
        let indexes = loader.get_indexes();
        if self.cursor < indexes.len() {

            let index = indexes[self.cursor];
            let record = loader.fetch(index).unwrap();
            self.cursor += 1;
            Some(record)
        }
        else {

            None
        }
    }
}
