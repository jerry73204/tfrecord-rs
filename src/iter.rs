use std::{io, ops, error, hash, cmp, marker, collections::hash_map};
use std::collections::vec_deque::VecDeque;
use std::collections::{HashMap, HashSet};
use tch;
use image::{ImageFormat, ImageDecoder};
use image::png::PNGDecoder;
use image::jpeg::JPEGDecoder;
use image::gif::Decoder as GIFDecoder;
use image::webp::WebpDecoder;
use image::pnm::PNMDecoder;
use image::tiff::TIFFDecoder;
use image::tga::TGADecoder;
use image::bmp::BMPDecoder;
use image::ico::ICODecoder;
use rand::prelude::*;
use ndarray::{ArrayBase, Array2, Array3};

// use tensorflow as tf;
use crate::parser;
use crate::loader;
use crate::error::{ParseError, make_load_index_error};

// Trait defiintions

pub trait DsIterator: Iterator
{
    fn into_tf_example(self, names: Option<HashSet<String>>) -> IntoTfExample<Self> where
        Self: Sized
    {
        IntoTfExample {
            iter: self,
            names,
        }
    }

    fn decode_image(self, formats: HashMap<String, Option<ImageFormat>>) -> DecodeImage<Self> where
        Self: Sized
    {
        DecodeImage {
            iter: self,
            formats,
        }
    }

    fn into_torch_tensor(self, names: Option<HashSet<String>>) -> IntoTorchTensor<Self> where
        Self: Sized
    {
        IntoTorchTensor {
            iter: self,
            names,
        }
    }

    fn into_tf_tensor(self) -> IntoTfTensor<Self> where
        Self: Sized
    {
        IntoTfTensor {
            iter: self,
        }
    }

    fn filter_hashmap_entry<K>(self, keys: HashSet<K>) -> FilterHashMapEntry<Self, K> where Self: Sized {
        FilterHashMapEntry {
            iter: self,
            keys,
        }
    }

    fn unwrap_result<V, E>(self) -> UnwrapResult<Self, V, E> where
        Self: Sized
    {
        UnwrapResult {
            iter: self,
            dummy_value: marker::PhantomData,
            dummy_error: marker::PhantomData,
        }
    }

    fn unwrap_ok<V, E>(self) -> UnwrapOk<Self, V, E> where
        Self: Sized
    {
        UnwrapOk {
            iter: self,
            dummy_value: marker::PhantomData,
            dummy_error: marker::PhantomData,
        }
    }

    fn shuffle(self, buf_size: usize) -> Shuffle<Self> where
        Self: Sized + Iterator,
    {
        let buffer = VecDeque::with_capacity(buf_size);
        let rng = rand::thread_rng();

        Shuffle {
            iter: self,
            buffer,
            rng,
        }
    }

    // TODO
    // fn prefetch(self, buf_size: usize) -> Prefetch<Self> where
    //     Self: Sized + Iterator,
    // {
    //     let buffer = Vec::with_capacity(buf_size);

    //     Prefetch {
    //         iter: self,
    //         buffer,
    //     }
    // }

    fn load_by_tfrecord_index(self, loader: loader::IndexedLoader) -> LoadByTfRecordIndex<Self> where
        Self: Sized
    {
        LoadByTfRecordIndex {
            iter: self,
            loader,
        }
    }
}

// Struct definitions

pub type FeatureDict = HashMap<String, Feature>;

pub enum Feature
{
    BytesList(Vec<Vec<u8>>),
    F32List(Vec<f32>),
    I64List(Vec<i64>),
    // BytesSeqList(Vec<Vec<Vec<u8>>>),
    // F32SeqList(Vec<Vec<f32>>),
    // I64SeqList(Vec<Vec<i64>>),
    Array2U8(Array2<u8>),
    Array2F32(Array2<f32>),
    Array2F64(Array2<f64>),
    Array3U8(Array3<u8>),
    Array3F32(Array3<f32>),
    Array3F64(Array3<f64>),
    Array2U8List(Vec<Array2<u8>>),
    Array2F32List(Vec<Array2<f32>>),
    Array2F64List(Vec<Array2<f64>>),
    Array3U8List(Vec<Array3<u8>>),
    Array3F32List(Vec<Array3<f32>>),
    Array3F64List(Vec<Array3<f64>>),
    TorchTensor(tch::Tensor),
    TorchTensorList(Vec<tch::Tensor>),
}

pub enum FeatureShape<'a>
{
    Fixed(Vec<i64>),
    FixedRef(&'a [i64]),
}

#[derive(Clone)]
pub struct IntoTfExample<I>
{
    names: Option<HashSet<String>>,
    iter: I,
}

#[derive(Clone)]
pub struct IntoTorchTensor<I>
{
    iter: I,
    names: Option<HashSet<String>>,
}

#[derive(Clone)]
pub struct IntoTfTensor<I>
{
    iter: I,
}

#[derive(Clone)]
pub struct FilterHashMapEntry<I, K>
{
    keys: HashSet<K>,
    iter: I,
}

#[derive(Clone)]
pub struct UnwrapResult<I, V, E>
{
    iter: I,
    dummy_value: marker::PhantomData<V>,
    dummy_error: marker::PhantomData<E>,
}

#[derive(Clone)]
pub struct UnwrapOk<I, V, E>
{
    iter: I,
    dummy_value: marker::PhantomData<V>,
    dummy_error: marker::PhantomData<E>,
}

#[derive(Clone)]
pub struct DecodeImage<I>
{
    formats: HashMap<String, Option<ImageFormat>>,
    iter: I,
}

#[derive(Clone)]
pub struct Shuffle<I: Iterator>
{
    iter: I,
    buffer: VecDeque<I::Item>,
    rng: rand::rngs::ThreadRng,
}

// #[derive(Clone)]
// pub struct Prefetch<I: Iterator>
// {
//     iter: I,
//     buffer: Vec<I::Item>,
// }

pub struct LoadByTfRecordIndex<I>
{
    iter: I,
    loader: loader::IndexedLoader,
}

// impl

impl<T> DsIterator for T where
    T: Iterator,
{}

impl<I> Iterator for IntoTfExample<I> where
    I: Iterator<Item=Vec<u8>>,
{
    type Item = Result<FeatureDict, io::Error>;

    fn next(&mut self) -> Option<Self::Item>
    {
        match self.iter.next()
        {
            None => None,
            Some(buf) => {
                match self.names
                {
                    None => {
                        match parser::parse_single_example(&buf)
                        {
                            Err(e) => Some(Err(e)),
                            Ok(mut example) => {
                                let mut converted_example = HashMap::new();
                                for (name, value) in example.drain()
                                {
                                    let new_value = match value
                                    {
                                        parser::FeatureList::Bytes(val) =>
                                            Feature::BytesList(val),
                                        parser::FeatureList::F32(val) =>
                                            Feature::F32List(val),
                                        parser::FeatureList::I64(val) =>
                                            Feature::I64List(val),
                                    };
                                    converted_example.insert(name, new_value);
                                }
                                Some(Ok(converted_example))
                            }
                        }
                    }
                    Some(ref feature_set) => {
                        match parser::parse_single_example(&buf)
                        {
                            Err(e) => Some(Err(e)),
                            Ok(mut example) => {
                                let mut filtered_example = HashMap::new();

                                for name in feature_set
                                {
                                    if let Some(value) = example.remove(name)
                                    {
                                        let owned_name = name.to_owned();
                                        let new_value = match value
                                        {
                                            parser::FeatureList::Bytes(val) =>
                                                Feature::BytesList(val),
                                            parser::FeatureList::F32(val) =>
                                                Feature::F32List(val),
                                            parser::FeatureList::I64(val) =>
                                                Feature::I64List(val),
                                        };
                                        filtered_example.insert(owned_name, new_value);
                                    }
                                }
                                Some(Ok(filtered_example))
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<I> Iterator for IntoTorchTensor<I> where
    I: Iterator<Item=FeatureDict>
{
    type Item = Result<FeatureDict, Box<error::Error>>;

    fn next(&mut self) -> Option<Self::Item>
    {
        match self.iter.next()
        {
            None => None,
            Some(mut example) => {
                let entries: Vec<_> = match &self.names
                {
                    Some(names) => {
                        let mut entries = Vec::new();

                        for key in names
                        {
                            let entry = match example.remove_entry(key)
                            {
                                Some(entry) => entry,
                                None => {
                                    let err = ParseError::new(&format!("Name \"{}\" is not found in example", key));
                                    return Some(Err(Box::new(err)));
                                }
                            };
                            let (new_key, new_val) = entry;
                            entries.push((new_key, new_val));
                        }
                        entries
                    }
                    None => example.into_iter().collect(),
                };

                let new_example: HashMap<_, _> = entries.into_iter()
                    .map(|(key, feature)| {
                        match feature
                        {
                            Feature::BytesList(list) => {
                                let tensor_list: Vec<_> = list.into_iter()
                                    .map(|bytes| {
                                        tch::Tensor::of_slice(&bytes)
                                    }).collect();
                                (key, Feature::TorchTensorList(tensor_list))
                            }
                            Feature::F32List(val) => {
                                (key, Feature::TorchTensor(tch::Tensor::of_slice(&val)))
                            }
                            Feature::I64List(val) => {
                                (key, Feature::TorchTensor(tch::Tensor::of_slice(&val)))
                            }
                            Feature::Array2U8(array) => {
                                let shape: Vec<_> = array.shape()
                                    .iter()
                                    .map(|dim| *dim as i64)
                                    .collect();
                                let tensor = tch::Tensor::of_slice(array.as_slice().unwrap());
                                tensor.view(&shape);
                                (key, Feature::TorchTensor(tensor))
                            }
                            Feature::Array2F32(array) => {
                                let shape: Vec<_> = array.shape()
                                    .iter()
                                    .map(|dim| *dim as i64)
                                    .collect();
                                let tensor = tch::Tensor::of_slice(array.as_slice().unwrap());
                                tensor.view(&shape);
                                (key, Feature::TorchTensor(tensor))
                            }
                            Feature::Array2F64(array) => {
                                let shape: Vec<_> = array.shape()
                                    .iter()
                                    .map(|dim| *dim as i64)
                                    .collect();
                                let tensor = tch::Tensor::of_slice(array.as_slice().unwrap());
                                tensor.view(&shape);
                                (key, Feature::TorchTensor(tensor))
                            }
                            Feature::Array3U8(array) => {
                                let shape: Vec<_> = array.shape()
                                    .iter()
                                    .map(|dim| *dim as i64)
                                    .collect();
                                let tensor = tch::Tensor::of_slice(array.as_slice().unwrap());
                                tensor.view(&shape);
                                (key, Feature::TorchTensor(tensor))
                            }
                            Feature::Array3F32(array) => {
                                let shape: Vec<_> = array.shape()
                                    .iter()
                                    .map(|dim| *dim as i64)
                                    .collect();
                                let tensor = tch::Tensor::of_slice(array.as_slice().unwrap());
                                tensor.view(&shape);
                                (key, Feature::TorchTensor(tensor))
                            }
                            Feature::Array3F64(array) => {
                                let shape: Vec<_> = array.shape()
                                    .iter()
                                    .map(|dim| *dim as i64)
                                    .collect();
                                let tensor = tch::Tensor::of_slice(array.as_slice().unwrap());
                                tensor.view(&shape);
                                (key, Feature::TorchTensor(tensor))
                            }
                            Feature::Array2U8List(list) => {
                                let tensor_list: Vec<_> = list.into_iter()
                                    .map(|array| {
                                        let shape: Vec<_> = array.shape()
                                            .iter()
                                            .map(|dim| *dim as i64)
                                            .collect();

                                        let tensor = tch::Tensor::of_slice(array.as_slice().unwrap());
                                        tensor.view(&shape);
                                        tensor
                                    })
                                    .collect();
                                (key, Feature::TorchTensorList(tensor_list))
                            }
                            Feature::Array2F32List(list) => {
                                let tensor_list: Vec<_> = list.into_iter()
                                    .map(|array| {
                                        let shape: Vec<_> = array.shape()
                                            .iter()
                                            .map(|dim| *dim as i64)
                                            .collect();

                                        let tensor = tch::Tensor::of_slice(array.as_slice().unwrap());
                                        tensor.view(&shape);
                                        tensor
                                    })
                                    .collect();
                                (key, Feature::TorchTensorList(tensor_list))
                            }
                            Feature::Array2F64List(list) => {
                                let tensor_list: Vec<_> = list.into_iter()
                                    .map(|array| {
                                        let shape: Vec<_> = array.shape()
                                            .iter()
                                            .map(|dim| *dim as i64)
                                            .collect();

                                        let tensor = tch::Tensor::of_slice(array.as_slice().unwrap());
                                        tensor.view(&shape);
                                        tensor
                                    })
                                    .collect();
                                (key, Feature::TorchTensorList(tensor_list))
                            }
                            Feature::Array3U8List(list) => {
                                let tensor_list: Vec<_> = list.into_iter()
                                    .map(|array| {
                                        let shape: Vec<_> = array.shape()
                                            .iter()
                                            .map(|dim| *dim as i64)
                                            .collect();

                                        let tensor = tch::Tensor::of_slice(array.as_slice().unwrap());
                                        tensor.view(&shape);
                                        tensor
                                    })
                                    .collect();
                                (key, Feature::TorchTensorList(tensor_list))
                            }
                            Feature::Array3F32List(list) => {
                                let tensor_list: Vec<_> = list.into_iter()
                                    .map(|array| {
                                        let shape: Vec<_> = array.shape()
                                            .iter()
                                            .map(|dim| *dim as i64)
                                            .collect();

                                        let tensor = tch::Tensor::of_slice(array.as_slice().unwrap());
                                        tensor.view(&shape);
                                        tensor
                                    })
                                    .collect();
                                (key, Feature::TorchTensorList(tensor_list))
                            }
                            Feature::Array3F64List(list) => {
                                let tensor_list: Vec<_> = list.into_iter()
                                    .map(|array| {
                                        let shape: Vec<_> = array.shape()
                                            .iter()
                                            .map(|dim| *dim as i64)
                                            .collect();

                                        let tensor = tch::Tensor::of_slice(array.as_slice().unwrap());
                                        tensor.view(&shape);
                                        tensor
                                    })
                                    .collect();
                                (key, Feature::TorchTensorList(tensor_list))
                            }
                            Feature::TorchTensor(tensor) => (key, Feature::TorchTensor(tensor)),
                            Feature::TorchTensorList(list) => (key, Feature::TorchTensorList(list)),
                        }
                    })
                    .collect();


                Some(Ok(new_example))
            }
        }
    }
}

// TODO: implementation

// impl<'a, I> Iterator for IntoTfTensor<'a, I> where
//     I: Iterator<Item=FeatureDict>
// {
//     type Item = Result<HashMap<String, T>, ParseError>;

//     fn next(&mut self) -> Option<Self::Item>
//     {
//         match self.iter.next()
//         {
//             None => None,
//             Some(example) => {
//                 let tensor_map_result: Result<_, _> = example.into_iter()
//                     .map(|(name, value)| {
//                         let result = match value
//                         {
//                             // Feature::BytesList(val) => {
//                             //     Err(ParseError::new("Cannot convert BytesList to tch::Tensor. Consider using decode_image()"))
//                             // }
//                             Feature::F32List(val) => {
//                                 Ok(tch::Tensor::of_slice(&val))
//                             }
//                             Feature::I64List(val) => {
//                                 Ok(tch::Tensor::of_slice(&val))
//                             }
//                             // Feature::BytesSeqList(val) => {
//                             //     Err(ParseError::new("Cannot convert BytesList to tch::Tensor. Consider using decode_image()"))
//                             // }
//                             // Feature::F32SeqList(val) => {
//                             //     Ok(tch::Tensor::of_slice(&val))
//                             // }
//                             // Feature::I64SeqList(val) => {
//                             //     Ok(tch::Tensor::of_slice(&val))
//                             // }
//                             _ => Err(ParseError::new("Cannot convert BytesList to tch::Tensor. Consider using decode_image()")),
//                         };

//                         (name, result)
//                     })
//                     .fold(Ok(HashMap::new()), |overall_result, (name, result)| {
//                         match overall_result
//                         {
//                             Err(e) => Err(e),
//                             Ok(mut acc) => {
//                                 match result
//                                 {
//                                     Err(e) => Err(e),
//                                     Ok(tensor) => {
//                                         acc.insert(name, tensor);
//                                         Ok(acc)
//                                     }
//                                 }
//                             }
//                         }
//                     });

//                 Some(tensor_map_result)
//             }
//         }
//     }
// }


impl<I, K, V> Iterator for FilterHashMapEntry<I, K> where
    I: Iterator<Item=HashMap<K, V>>,
    K: hash::Hash + cmp::Eq
{
    type Item = HashMap<K, V>;

    fn next(&mut self) -> Option<Self::Item>
    {
        match self.iter.next()
        {
            None => None,
            Some(mut index) => {
                let new_index: HashMap<K, V> = self.keys.iter().filter_map(|query_key| {
                    match index.remove_entry(&query_key)
                    {
                        Some((key, value)) => Some((key, value)),
                        None => None,
                    }
                }).collect();
                Some(new_index)
            }
        }
    }
}

impl<I, V, E> Iterator for UnwrapOk<I, V, E> where
    I: Iterator<Item=Result<V, E>>,
{
    type Item = V;

    fn next(&mut self) -> Option<Self::Item>
    {
        match self.iter.next()
        {
            None => None,
            Some(result) => Some(result.ok().unwrap())
        }
    }
}

impl<I, V, E> Iterator for UnwrapResult<I, V, E> where
    I: Iterator<Item=Result<V, E>>,
    E: std::fmt::Debug,
{
    type Item = V;

    fn next(&mut self) -> Option<Self::Item>
    {
        match self.iter.next()
        {
            None => None,
            Some(result) => Some(result.unwrap())
        }
    }
}

impl<I> Iterator for LoadByTfRecordIndex<I> where
    I: Iterator<Item=loader::RecordIndex>,
{
    type Item = Result<Vec<u8>, io::Error>;

    fn next(&mut self) -> Option<Self::Item>
    {
        match self.iter.next()
        {
            None => None,
            Some(index) => match self.loader.fetch(index) {
                Some(record) => Some(Ok(record)),
                None => Some(Err(make_load_index_error())),
            }
        }
    }
}

impl<I> Iterator for DecodeImage<I> where
    I: Iterator<Item=FeatureDict>,
{
    type Item = Result<FeatureDict, Box<error::Error>>;

    fn next(&mut self) -> Option<Self::Item>
    {
        match self.iter.next()
        {
            None => None,
            Some(mut example) => {
                for (query_name, format_opt) in &self.formats
                {

                    let mut entry = match example.entry(query_name.to_owned())
                    {
                        hash_map::Entry::Vacant(entry) => {
                            let name = entry.key();
                            let err = ParseError::new(&format!("Name \"{}\" is not found in example", name));
                            return Some(Err(Box::new(err)));
                        }
                        hash_map::Entry::Occupied(entry) => entry
                    };
                    let (format, bytes_list) = match entry.get_mut() {
                        Feature::BytesList(bytes_list) => {
                            if bytes_list.is_empty()
                            {
                                let name = entry.key();
                                let err = ParseError::new(&format!("Cannot decode empty bytes list feature with name \"{}\"", name));
                                return Some(Err(Box::new(err)));
                            }

                            match format_opt
                            {
                                Some(format) => (format.to_owned(), bytes_list),
                                None => {
                                    match image::guess_format(&bytes_list[0])
                                    {
                                        Ok(format) => (format, bytes_list),
                                        Err(e) => return Some(Err(Box::new(e))),
                                    }
                                }
                            }
                        }
                        _ => {
                            let name = entry.key();
                            let err = ParseError::new(&format!("Cannot decode non bytes list feature with name \"{}\"", name));
                            return Some(Err(Box::new(err)));
                        }
                    };

                    let mut images = Vec::new();
                    match format
                    {
                        ImageFormat::PNG => {
                            for bytes in bytes_list
                            {
                                match PNGDecoder::new(bytes.as_slice())
                                {
                                    Ok(decoder) => {
                                        let dimensions = decoder.dimensions();
                                        match decoder.read_image()
                                        {
                                            Ok(image) => images.push((image, dimensions)),
                                            Err(e) => return Some(Err(Box::new(e))),
                                        }
                                    },
                                    Err(e) => return Some(Err(Box::new(e))),
                                }
                            }
                        }
                        ImageFormat::JPEG => {
                            for bytes in bytes_list
                            {
                                match JPEGDecoder::new(bytes.as_slice())
                                {
                                    Ok(decoder) => {
                                        let dimensions = decoder.dimensions();
                                        match decoder.read_image()
                                        {
                                            Ok(image) => images.push((image, dimensions)),
                                            Err(e) => return Some(Err(Box::new(e))),
                                        }
                                    },
                                    Err(e) => return Some(Err(Box::new(e))),
                                }
                            }
                        }
                        ImageFormat::GIF => {
                            for bytes in bytes_list
                            {
                                match GIFDecoder::new(bytes.as_slice())
                                {
                                    Ok(decoder) => {
                                        let dimensions = decoder.dimensions();
                                        match decoder.read_image()
                                        {
                                            Ok(image) => images.push((image, dimensions)),
                                            Err(e) => return Some(Err(Box::new(e))),
                                        }
                                    },
                                    Err(e) => return Some(Err(Box::new(e))),
                                }
                            }
                        }
                        ImageFormat::WEBP => {
                            for bytes in bytes_list
                            {
                                match WebpDecoder::new(bytes.as_slice())
                                {
                                    Ok(decoder) => {
                                        let dimensions = decoder.dimensions();
                                        match decoder.read_image()
                                        {
                                            Ok(image) => images.push((image, dimensions)),
                                            Err(e) => return Some(Err(Box::new(e))),
                                        }
                                    },
                                    Err(e) => return Some(Err(Box::new(e))),
                                }
                            }
                        }
                        ImageFormat::PNM => {
                            for bytes in bytes_list
                            {
                                match PNMDecoder::new(bytes.as_slice())
                                {
                                    Ok(decoder) => {
                                        let dimensions = decoder.dimensions();
                                        match decoder.read_image()
                                        {
                                            Ok(image) => images.push((image, dimensions)),
                                            Err(e) => return Some(Err(Box::new(e))),
                                        }
                                    },
                                    Err(e) => return Some(Err(Box::new(e))),
                                }
                            }
                        }
                        ImageFormat::TIFF => {
                            for bytes in bytes_list
                            {
                                match TIFFDecoder::new(io::Cursor::new(&bytes))
                                {
                                    Ok(decoder) => {
                                        let dimensions = decoder.dimensions();
                                        match decoder.read_image()
                                        {
                                            Ok(image) => images.push((image, dimensions)),
                                            Err(e) => return Some(Err(Box::new(e))),
                                        }
                                    },
                                    Err(e) => return Some(Err(Box::new(e))),
                                }
                            }
                        }
                        ImageFormat::TGA => {
                            for bytes in bytes_list
                            {
                                match TGADecoder::new(io::Cursor::new(&bytes))
                                {
                                    Ok(decoder) => {
                                        let dimensions = decoder.dimensions();
                                        match decoder.read_image()
                                        {
                                            Ok(image) => images.push((image, dimensions)),
                                            Err(e) => return Some(Err(Box::new(e))),
                                        }
                                    },
                                    Err(e) => return Some(Err(Box::new(e))),
                                }
                            }
                        }
                        ImageFormat::BMP => {
                            for bytes in bytes_list
                            {
                                match BMPDecoder::new(io::Cursor::new(&bytes))
                                {
                                    Ok(decoder) => {
                                        let dimensions = decoder.dimensions();
                                        match decoder.read_image()
                                        {
                                            Ok(image) => images.push((image, dimensions)),
                                            Err(e) => return Some(Err(Box::new(e))),
                                        }
                                    },
                                    Err(e) => return Some(Err(Box::new(e))),
                                }
                            }
                        }
                        ImageFormat::ICO => {
                            for bytes in bytes_list
                            {
                                match ICODecoder::new(io::Cursor::new(&bytes))
                                {
                                    Ok(decoder) => {
                                        let dimensions = decoder.dimensions();
                                        match decoder.read_image()
                                        {
                                            Ok(image) => images.push((image, dimensions)),
                                            Err(e) => return Some(Err(Box::new(e))),
                                        }
                                    },
                                    Err(e) => return Some(Err(Box::new(e))),
                                }
                            }
                        }
                        _ => {
                            let name = entry.key();
                            let err = ParseError::new(&format!("Image format is not supported for feature with name \"{}\"", name));
                            return Some(Err(Box::new(err)));
                        }
                    };


                    let mut arrays = Vec::new();
                    for (image, dims) in images
                    {
                        let (width, height) = dims;
                        let shape = (width as usize, height as usize, 3);

                        match Array3::from_shape_vec(shape, image)
                        {
                            Ok(array) => arrays.push(array),
                            Err(err) => return Some(Err(Box::new(err))),
                        }
                    }
                    entry.insert(Feature::Array3U8List(arrays));
                }

                Some(Ok(example))
            }
        }
    }
}

impl<I> Iterator for Shuffle<I> where
    I: Iterator,
{
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item>
    {
        let capacity = self.buffer.capacity();
        if capacity > 0
        {
            while self.buffer.len() < capacity
            {
                match self.iter.next()
                {
                    None => break,
                    Some(item) => {
                        self.buffer.push_front(item);
                        let buf_len = self.buffer.len();
                        let swap_ind = self.rng.gen_range(0, buf_len);
                        self.buffer.swap(0, swap_ind);
                    }
                }
            }

            self.buffer.pop_back()
        }
        else
        {
            self.iter.next()
        }
    }
}

// TODO
// impl<I> Iterator for Prefetch<I> where
//     I: Iterator,
// {
//     type Item = I::Item;
//     fn next(&mut self) -> Option<Self::Item>
//     {
//     }
// }
