use std::io;
use std::cmp::Eq;
use std::hash::Hash;
use std::marker::PhantomData;
use std::collections::vec_deque::VecDeque;
use std::collections::{HashMap, HashSet};
use std::any::Any;
use std::fmt::{Debug, Display};
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
use ndarray::{ArrayBase, Array2, Array3, Array4};
// use tensorflow as tf;
use crate::parser;
use crate::loader;
use crate::error::{ParseError, make_load_index_error};

// Trait defiintions

pub trait DsIterator: Iterator
{
    fn to_tf_example(self, names_opt: Option<HashSet<&str>>) -> ToTfExample<Self, &str> where
        Self: Sized {
        ToTfExample {
            iter: self,
            names_opt,
        }
    }

    fn decode_image<S>(self, formats_opt: Option<HashMap<S, Option<ImageFormat>>>) -> DecodeImage<Self, S> where
        Self: Sized {
        DecodeImage {
            iter: self,
            formats_opt,
        }
    }

    fn to_torch_tensor(self, names_opt: Option<HashSet<&str>>) -> ToTorchTensor<Self, &str> where
        Self: Sized, {
        ToTorchTensor {
            iter: self,
            names_opt,
            dummy_name: PhantomData,
        }
    }

    fn to_tf_tensor(self) -> ToTfTensor<Self> where
        Self: Sized
    {
        ToTfTensor {
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
            dummy_value: PhantomData,
            dummy_error: PhantomData,
        }
    }

    fn unwrap_ok<V, E>(self) -> UnwrapOk<Self, V, E> where
        Self: Sized
    {
        UnwrapOk {
            iter: self,
            dummy_value: PhantomData,
            dummy_error: PhantomData,
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

#[derive(Clone)]
pub struct ToTfExample<I, S>
{
    names_opt: Option<HashSet<S>>,
    iter: I,
}

pub struct ToTorchTensor<I, S>
{
    iter: I,
    names_opt: Option<HashSet<S>>,
    dummy_name: PhantomData<S>,
}

#[derive(Clone)]
pub struct ToTfTensor<I>
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
    dummy_value: PhantomData<V>,
    dummy_error: PhantomData<E>,
}

#[derive(Clone)]
pub struct UnwrapOk<I, V, E>
{
    iter: I,
    dummy_value: PhantomData<V>,
    dummy_error: PhantomData<E>,
}

#[derive(Clone)]
pub struct DecodeImage<I, S>
{
    formats_opt: Option<HashMap<S, Option<ImageFormat>>>,
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

impl<I, S> Iterator for ToTfExample<I, S> where
        I: Iterator<Item=Vec<u8>>,
        S: AsRef<str> + Hash + Eq + Display, {

    type Item = Result<HashMap<String, Box<dyn Any>>, Box<Debug>>;

    fn next(&mut self) -> Option<Self::Item>
    {
        let buf = match self.iter.next() {
            None => return None,
            Some(buf) => buf,
        };

        let mut example = match parser::parse_single_example(&buf) {
            Err(e) => return Some(Err(Box::new(e))),
            Ok(example) => example,
        };

        let entries: Vec<_> = match self.names_opt {
            Some(ref names) => {
                let mut entries = Vec::new();
                for name in names {
                    let entry = match example.remove_entry(name.as_ref()) {
                        Some(entry) => entry,
                        None => {
                            let err = ParseError::new(&format!("Name \"{}\" is not found in example", name));
                            return Some(Err(Box::new(err)));
                        }
                    };
                    entries.push(entry);
                }

                entries
            }
            None => example.into_iter().collect()
        };

        let mut result = HashMap::new();

        for (name, value) in entries {
            let parsed_value: Box<dyn Any> = match value {
                parser::FeatureList::Bytes(val) => Box::new(val),
                parser::FeatureList::F32(val) => Box::new(val),
                parser::FeatureList::I64(val) => Box::new(val),
            };

            result.insert(name, parsed_value);
        }

        Some(Ok(result))
    }
}

macro_rules! try_convert_array_to_torch (
    ( $value_ref:ident, $dtype:ty ) => (
        match $value_ref.downcast_ref::<$dtype>() {
            None => {}
            Some(val) => {
                let dims = val.shape().into_iter().map(|v| *v as i64).collect::<Vec<_>>();
                let tensor = tch::Tensor::of_slice(val.as_slice().unwrap());
                tensor.view(&dims);
                return Ok(Box::new(tensor));
            }
        }
    )
);

macro_rules! try_convert_array_vec_to_torch (
    ( $value_ref:ident, $dtype:ty ) => (
        match $value_ref.downcast_ref::<Vec<$dtype>>() {
            None => {}
            Some(list) => {
                let tensor_list = list.into_iter()
                    .map(|val| {
                        let dims = val.shape().into_iter().map(|v| *v as i64).collect::<Vec<_>>();
                        let tensor = tch::Tensor::of_slice(val.as_slice().unwrap());
                        tensor.view(&dims);
                        tensor
                    })
                    .collect::<Vec<_>>();
                return Ok(Box::new(tensor_list));
            }
        }
    )
);


macro_rules! try_convert_vec_to_torch (
    ( $value_ref:ident, $dtype:ty ) => (
        match $value_ref.downcast_ref::<Vec<$dtype>>() {
            None => {}
            Some(val) => {
                return Ok(Box::new(tch::Tensor::of_slice(val)));
            }
        }
    )
);

macro_rules! try_convert_vec_vec_to_torch (
    ( $value_ref:ident, $dtype:ty ) => (
        match $value_ref.downcast_ref::<Vec<Vec<$dtype>>>() {
            None => {}
            Some(list) => {
                let tensor_list = list.into_iter()
                    .map(|val| tch::Tensor::of_slice(val))
                    .collect::<Vec<_>>();
                return Ok(Box::new(tensor_list));
            }
        }
    )
);

impl<I, S> ToTorchTensor<I, S> where
    S: AsRef<str> {

    fn try_convert_to_tensor(name: &str, value_ref: Box<dyn Any>) -> Result<Box<dyn Any>, Box<dyn Debug>> {

        // TODO: optimize type matching
        try_convert_vec_to_torch!(value_ref, u8);
        try_convert_vec_to_torch!(value_ref, f32);
        try_convert_vec_to_torch!(value_ref, f64);
        try_convert_vec_to_torch!(value_ref, i32);
        try_convert_vec_to_torch!(value_ref, i64);

        try_convert_vec_vec_to_torch!(value_ref, u8);
        try_convert_vec_vec_to_torch!(value_ref, f32);
        try_convert_vec_vec_to_torch!(value_ref, f64);
        try_convert_vec_vec_to_torch!(value_ref, i32);
        try_convert_vec_vec_to_torch!(value_ref, i64);

        try_convert_array_to_torch!(value_ref, Array2<u8>);
        try_convert_array_to_torch!(value_ref, Array2<f32>);
        try_convert_array_to_torch!(value_ref, Array2<f64>);
        try_convert_array_to_torch!(value_ref, Array2<i32>);
        try_convert_array_to_torch!(value_ref, Array2<i64>);

        try_convert_array_to_torch!(value_ref, Array3<u8>);
        try_convert_array_to_torch!(value_ref, Array3<f32>);
        try_convert_array_to_torch!(value_ref, Array3<f64>);
        try_convert_array_to_torch!(value_ref, Array3<i32>);
        try_convert_array_to_torch!(value_ref, Array3<i64>);

        try_convert_array_to_torch!(value_ref, Array4<u8>);
        try_convert_array_to_torch!(value_ref, Array4<f32>);
        try_convert_array_to_torch!(value_ref, Array4<f64>);
        try_convert_array_to_torch!(value_ref, Array4<i32>);
        try_convert_array_to_torch!(value_ref, Array4<i64>);

        try_convert_array_vec_to_torch!(value_ref, Array2<u8>);
        try_convert_array_vec_to_torch!(value_ref, Array2<f32>);
        try_convert_array_vec_to_torch!(value_ref, Array2<f64>);
        try_convert_array_vec_to_torch!(value_ref, Array2<i32>);
        try_convert_array_vec_to_torch!(value_ref, Array2<i64>);

        try_convert_array_vec_to_torch!(value_ref, Array3<u8>);
        try_convert_array_vec_to_torch!(value_ref, Array3<f32>);
        try_convert_array_vec_to_torch!(value_ref, Array3<f64>);
        try_convert_array_vec_to_torch!(value_ref, Array3<i32>);
        try_convert_array_vec_to_torch!(value_ref, Array3<i64>);

        try_convert_array_vec_to_torch!(value_ref, Array4<u8>);
        try_convert_array_vec_to_torch!(value_ref, Array4<f32>);
        try_convert_array_vec_to_torch!(value_ref, Array4<f64>);
        try_convert_array_vec_to_torch!(value_ref, Array4<i32>);
        try_convert_array_vec_to_torch!(value_ref, Array4<i64>);

        let err = ParseError::new(&format!("The type of feature with name \"{}\" is not supported to convert to Torch Tensor", name));
        Err(Box::new(err))
    }
}

impl<I, S> Iterator for ToTorchTensor<I, S> where
    I: Iterator<Item=HashMap<String, Box<dyn Any>>>,
    S: AsRef<str> + Hash + Eq + Display {
    type Item = Result<HashMap<String, Box<dyn Any>>, Box<Debug>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut example = match self.iter.next() {
            None => return None,
            Some(example) => example,
        };

        let mut result = HashMap::<String, Box<dyn Any>>::new();

        let entries: Vec<_> = match &self.names_opt {
            None => example.into_iter().collect(),
            Some(names) => {
                let mut entries = Vec::new();
                for name in names {
                    let entry = match example.remove_entry(name.as_ref()) {
                        Some(entry) => entry,
                        None => {
                            let err = ParseError::new(&format!("Name \"{}\" is not found in example", name));
                            return Some(Err(Box::new(err)));
                        }
                    };
                    entries.push(entry);
                }

                for (name, val) in example.drain() {
                    result.insert(name, val);
                }

                entries
            }
        };

        for (name, feature_ref) in entries {
            let ret = match Self::try_convert_to_tensor(&name, feature_ref) {
                Err(err) => return Some(Err(err)),
                Ok(ret) => ret,
            };
            result.insert(name, ret);
        }

        Some(Ok(result))
    }
}

// TODO: implementation

// impl<'a, I> Iterator for IntoTfTensor<'a, I> where
//     I: Iterator<Item=HashMap<&'a str, Box<dyn Any>>>
// {
//     type Item = Result<HashMap<&'a str, T>, ParseError>;

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
    K: Hash + Eq
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
    E: Debug,
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

impl<I, S> DecodeImage<I, S> {

    fn try_decode_image(bytes: &[u8], format_opt: Option<ImageFormat>) -> Result<Array3<u8>, Box<Debug>> {
        let format = match format_opt {
            Some(format) => format,
            None => {
                match image::guess_format(bytes) {
                    Ok(format) => format,
                    Err(err) => return Err(Box::new(err)),
                }
            }
        };

        let (image, (width, height)) = match format {
            ImageFormat::PNG => {
                match PNGDecoder::new(bytes) {
                    Ok(decoder) => {
                        let dimensions = decoder.dimensions();
                        match decoder.read_image() {
                            Ok(image) => (image, dimensions),
                            Err(err) => return Err(Box::new(err)),
                        }
                    },
                    Err(err) => return Err(Box::new(err)),
                }
            }
            ImageFormat::JPEG => {
                match JPEGDecoder::new(bytes) {
                    Ok(decoder) => {
                        let dimensions = decoder.dimensions();
                        match decoder.read_image() {
                            Ok(image) => (image, dimensions),
                            Err(err) => return Err(Box::new(err)),
                        }
                    },
                    Err(err) => return Err(Box::new(err)),
                }
            }
            ImageFormat::GIF => {
                match GIFDecoder::new(bytes) {
                    Ok(decoder) => {
                        let dimensions = decoder.dimensions();
                        match decoder.read_image() {
                            Ok(image) => (image, dimensions),
                            Err(err) => return Err(Box::new(err)),
                        }
                    },
                    Err(err) => return Err(Box::new(err)),
                }
            }
            ImageFormat::WEBP => {
                match WebpDecoder::new(bytes) {
                    Ok(decoder) => {
                        let dimensions = decoder.dimensions();
                        match decoder.read_image() {
                            Ok(image) => (image, dimensions),
                            Err(err) => return Err(Box::new(err)),
                        }
                    },
                    Err(err) => return Err(Box::new(err)),
                }
            }
            ImageFormat::PNM => {
                match PNMDecoder::new(bytes) {
                    Ok(decoder) => {
                        let dimensions = decoder.dimensions();
                        match decoder.read_image() {
                            Ok(image) => (image, dimensions),
                            Err(err) => return Err(Box::new(err)),
                        }
                    },
                    Err(err) => return Err(Box::new(err)),
                }
            }
            ImageFormat::TIFF => {
                match TIFFDecoder::new(io::Cursor::new(bytes)) {
                    Ok(decoder) => {
                        let dimensions = decoder.dimensions();
                        match decoder.read_image() {
                            Ok(image) => (image, dimensions),
                            Err(err) => return Err(Box::new(err)),
                        }
                    },
                    Err(err) => return Err(Box::new(err)),
                }
            }
            ImageFormat::TGA => {
                match TGADecoder::new(io::Cursor::new(bytes)) {
                    Ok(decoder) => {
                        let dimensions = decoder.dimensions();
                        match decoder.read_image() {
                            Ok(image) => (image, dimensions),
                            Err(err) => return Err(Box::new(err)),
                        }
                    },
                    Err(err) => return Err(Box::new(err)),
                }
            }
            ImageFormat::BMP => {
                match BMPDecoder::new(io::Cursor::new(bytes)) {
                    Ok(decoder) => {
                        let dimensions = decoder.dimensions();
                        match decoder.read_image() {
                            Ok(image) => (image, dimensions),
                            Err(err) => return Err(Box::new(err)),
                        }
                    },
                    Err(err) => return Err(Box::new(err)),
                }
            }
            ImageFormat::ICO => {
                match ICODecoder::new(io::Cursor::new(bytes)) {
                    Ok(decoder) => {
                        let dimensions = decoder.dimensions();
                        match decoder.read_image() {
                            Ok(image) => (image, dimensions),
                            Err(err) => return Err(Box::new(err)),
                        }
                    },
                    Err(err) => return Err(Box::new(err)),
                }
            }
            _ => {
                let err = ParseError::new(&format!("Image format is not supported"));
                return Err(Box::new(err));
            }
        };


        let array = match ArrayBase::from_shape_vec((width as usize, height as usize, 3), image) {
            Err(err) => return Err(Box::new(err)),
            Ok(array) => array,
        };
        Ok(array)
    }
}

impl<I, S> Iterator for DecodeImage<I, S> where
    I: Iterator<Item=HashMap<String, Box<dyn Any>>>,
    S: AsRef<str> + Hash + Eq + Display {

    type Item = Result<HashMap<String, Box<dyn Any>>, Box<Debug>>;

    fn next(&mut self) -> Option<Self::Item>
    {
        let mut example = match self.iter.next() {
            None => return None,
            Some(example) => example,
        };

        let mut result = HashMap::<String, Box<Any>>::new();

        let entries = match &self.formats_opt {
            Some(formats) => {
                let mut entries = Vec::new();
                for (select_name, format_opt) in formats {
                    let (name, value_ref) = match example.remove_entry(select_name.as_ref()) {
                        Some(entry) => entry,
                        None => {
                            let err = ParseError::new(&format!("Name \"{}\" is not found in example", select_name));
                            return Some(Err(Box::new(err)));
                        }
                    };

                    for (name, val) in example.drain() {
                        result.insert(name, val);
                    }

                    entries.push((name, value_ref, format_opt.to_owned()));
                }
                entries
            }
            None => example.into_iter().map(|(name, val)| (name, val, None)).collect(),
        };

        for (name, value_ref, format_opt) in entries
        {
            if let Some(bytes) = value_ref.downcast_ref::<Vec<u8>>() {
                let image = match Self::try_decode_image(bytes, format_opt) {
                    Err(err) => return Some(Err(err)),
                    Ok(image) => image,
                };
                result.insert(name, Box::new(image));
            }
            else if let Some(bytes_list) = value_ref.downcast_ref::<Vec<Vec<u8>>>() {
                if bytes_list.is_empty() {
                    let err = ParseError::new(&format!("Cannot decode empty bytes list with name \"{}\"", name));
                    return Some(Err(Box::new(err)));
                }

                let mut images = Vec::new();
                for bytes in bytes_list {
                    let image = match Self::try_decode_image(bytes, format_opt) {
                        Err(err) => return Some(Err(err)),
                        Ok(image) => image,
                    };
                    images.push(image);
                }
                result.insert(name, Box::new(images));
            }
            else {
                let err = ParseError::new(&format!("Cannot decode non-bytes list feature with name \"{}\"", name));
                return Some(Err(Box::new(err)));
            }
        }

        Some(Ok(result))
    }
}

impl<I> Iterator for Shuffle<I> where
    I: Iterator,
{
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item>
    {
        let capacity = self.buffer.capacity();
        if capacity > 0 {
            while self.buffer.len() < capacity {
                match self.iter.next() {
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
        else {
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
