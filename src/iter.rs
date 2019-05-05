use std::io::{self, Cursor};
use std::thread::{self, JoinHandle};
use std::cmp::Eq;
use std::hash::Hash;
use std::marker::PhantomData;
use std::collections::vec_deque::VecDeque;
use std::collections::{HashMap, HashSet};
use std::mem::transmute;
use std::panic::catch_unwind;
use std::any::Any;
use std::fmt::{Debug, Display};
use rayon::prelude::*;
use tch;
use image::{ImageFormat, ImageDecoder};
use image::png::PNGDecoder;
use image::gif::Decoder as GIFDecoder;
use image::webp::WebpDecoder;
use image::pnm::PNMDecoder;
use image::tiff::TIFFDecoder;
use image::tga::TGADecoder;
use image::bmp::BMPDecoder;
use image::ico::ICODecoder;
use rand::prelude::*;
use ndarray::{ArrayBase, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4};
use crossbeam::channel::Receiver;
// use tensorflow as tf;
use crate::parser;
use crate::loader;
use crate::error::{ParseError, make_load_index_error};

// type defs

type FeatureType = Box<dyn Any + Sync + Send>;
type ErrorType = Box<dyn Debug + Sync + Send>;
type ExampleType = HashMap<String, FeatureType>;

// functoins

fn filter_entries<S, V>(
    mut map: HashMap<String, V>,
    names_opt: &Option<HashSet<S>>
) -> Result<(HashMap<String, V>, Vec<(String, V)>), ParseError> where
    S: AsRef<str>{

    let mut new_map = HashMap::new();

    let entries: Vec<_> = match names_opt {
        Some(ref names) => {
            let mut entries = Vec::new();
            for name in names {
                let entry = match map.remove_entry(name.as_ref()) {
                    Some(entry) => entry,
                    None => {
                        let err = ParseError::new(&format!("Feature with name \"{}\" is not found", name.as_ref()));
                        return Err(err);
                    }
                };
                entries.push(entry);
            }

            new_map = map;
            entries
        }
        None => map.into_iter().collect()
    };

    Ok((new_map, entries))
}


fn try_decode_image(bytes: &[u8], format_opt: Option<ImageFormat>) -> Result<Array3<u8>, ErrorType> {
    let format = match format_opt {
        Some(format) => format,
        None => {
            match image::guess_format(bytes) {
                Ok(format) => format,
                Err(err) => return Err(Box::new(err)),
            }
        }
    };

    let (image, (width, height, channels)) = match format {
        ImageFormat::PNG => {
            match PNGDecoder::new(bytes) {
                Ok(decoder) => {
                    let (width, height) = decoder.dimensions();
                    match decoder.read_image() {
                        Ok(image) => (image, (width as usize, height as usize, 3)),
                        Err(err) => return Err(Box::new(err)),
                    }
                },
                Err(err) => return Err(Box::new(err)),
            }
        }
        ImageFormat::JPEG => {
            match  decode_jpeg(&bytes) {
                Err(err) => return Err(Box::new(err)),
                Ok((image, dims)) => (image, dims),
            }
        }
        ImageFormat::GIF => {
            match GIFDecoder::new(bytes) {
                Ok(decoder) => {
                    let (width, height) = decoder.dimensions();
                    match decoder.read_image() {
                        Ok(image) => (image, (width as usize, height as usize, 3)),
                        Err(err) => return Err(Box::new(err)),
                    }
                },
                Err(err) => return Err(Box::new(err)),
            }
        }
        ImageFormat::WEBP => {
            match WebpDecoder::new(bytes) {
                Ok(decoder) => {
                    let (width, height) = decoder.dimensions();
                    match decoder.read_image() {
                        Ok(image) => (image, (width as usize, height as usize, 3)),
                        Err(err) => return Err(Box::new(err)),
                    }
                },
                Err(err) => return Err(Box::new(err)),
            }
        }
        ImageFormat::PNM => {
            match PNMDecoder::new(bytes) {
                Ok(decoder) => {
                    let (width, height) = decoder.dimensions();
                    match decoder.read_image() {
                        Ok(image) => (image, (width as usize, height as usize, 3)),
                        Err(err) => return Err(Box::new(err)),
                    }
                },
                Err(err) => return Err(Box::new(err)),
            }
        }
        ImageFormat::TIFF => {
            match TIFFDecoder::new(Cursor::new(bytes)) {
                Ok(decoder) => {
                    let (width, height) = decoder.dimensions();
                    match decoder.read_image() {
                        Ok(image) => (image, (width as usize, height as usize, 3)),
                        Err(err) => return Err(Box::new(err)),
                    }
                },
                Err(err) => return Err(Box::new(err)),
            }
        }
        ImageFormat::TGA => {
            match TGADecoder::new(Cursor::new(bytes)) {
                Ok(decoder) => {
                    let (width, height) = decoder.dimensions();
                    match decoder.read_image() {
                        Ok(image) => (image, (width as usize, height as usize, 3)),
                        Err(err) => return Err(Box::new(err)),
                    }
                },
                Err(err) => return Err(Box::new(err)),
            }
        }
        ImageFormat::BMP => {
            match BMPDecoder::new(Cursor::new(bytes)) {
                Ok(decoder) => {
                    let (width, height) = decoder.dimensions();
                    match decoder.read_image() {
                        Ok(image) => (image, (width as usize, height as usize, 3)),
                        Err(err) => return Err(Box::new(err)),
                    }
                },
                Err(err) => return Err(Box::new(err)),
            }
        }
        ImageFormat::ICO => {
            match ICODecoder::new(Cursor::new(bytes)) {
                Ok(decoder) => {
                    let (width, height) = decoder.dimensions();
                    match decoder.read_image() {
                        Ok(image) => (image, (width as usize, height as usize, 3)),
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


    let array = match ArrayBase::from_shape_vec((width, height, channels), image) {
        Err(err) => return Err(Box::new(err)),
        Ok(array) => array,
    };
    Ok(array)
}

fn decode_jpeg(data: &[u8]) -> Result<(Vec<u8>, (usize, usize, usize)), io::Error> {
    catch_unwind(|| -> Result<_, io::Error> {
        let decompress = mozjpeg::Decompress::with_markers(mozjpeg::ALL_MARKERS)
            .from_mem(data)?;
        let (width, height) = decompress.size();

        match decompress.image()? {
            mozjpeg::Format::RGB(mut dec) => {
                let mut pixels = dec.read_scanlines::<(u8, u8, u8)>().unwrap();
                let bytes = unsafe {
                    pixels.set_len(pixels.len() * 3);
                    transmute::<Vec<(u8, u8, u8)>, Vec<u8>>(pixels)
                };
                assert!(bytes.len() == width * height * 3);
                Ok((bytes, (width, height, 3)))
            }
            mozjpeg::Format::Gray(mut dec) => {
                let pixels = dec.read_scanlines::<u8>().unwrap();
                Ok((pixels, (width, height, 1)))
            }
            mozjpeg::Format::CMYK(mut dec) => {
                let pixels = dec.read_scanlines::<(u8, u8, u8, u8)>().unwrap();
                let bytes = pixels.into_iter()
                    .flat_map(|(c, m, y, k)| {
                        let r = ((255_f64 - c as f64) * (255_f64 - k as f64) / 255_f64) as u8;
                        let g = ((255_f64 - m as f64) * (255_f64 - k as f64) / 255_f64) as u8;
                        let b = ((255_f64 - y as f64) * (255_f64 - k as f64) / 255_f64) as u8;
                        vec![r, g, b]
                    })
                    .collect::<Vec<u8>>();
                Ok((bytes, (width, height, 3)))
            }
        }
    }).unwrap()
}

fn decode_image_on_example<S>(
    mut example: ExampleType,
    formats_opt: &Option<HashMap<S, Option<ImageFormat>>>,
) -> Result<ExampleType, ErrorType> where
    S: AsRef<str> + Hash + Eq + Display {

    let mut result = ExampleType::new();
    let entries = match formats_opt {
        Some(formats) => {
            let mut entries = Vec::new();
            for (select_name, format_opt) in formats {
                let (name, value_ref) = match example.remove_entry(select_name.as_ref()) {
                    Some(entry) => entry,
                    None => {
                        let err = ParseError::new(&format!("Name \"{}\" is not found in example", select_name));
                        return Err(Box::new(err));
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

    for (name, value_ref, format_opt) in entries {
        if let Some(bytes) = value_ref.downcast_ref::<Vec<u8>>() {
            let image = match try_decode_image(bytes, format_opt) {
                Err(err) => return Err(err),
                Ok(image) => image,
            };
            result.insert(name, Box::new(image));
        }
        else if let Some(bytes_list) = value_ref.downcast_ref::<Vec<Vec<u8>>>() {
            if bytes_list.is_empty() {
                let err = ParseError::new(&format!("Cannot decode empty bytes list with name \"{}\"", name));
                return Err(Box::new(err));
            }

            let mut images = Vec::new();
            for bytes in bytes_list {
                let image = match try_decode_image(bytes, format_opt) {
                    Err(err) => return Err(err),
                    Ok(image) => image,
                };
                images.push(image);
            }
            result.insert(name, Box::new(images));
        }
        else {
            let err = ParseError::new(&format!("Cannot decode non-bytes list feature with name \"{}\"", name));
            return Err(Box::new(err));
        }
    }

    Ok(result)
}


// Trait defiintions

pub trait DsIterator: Iterator + Sized {

    fn to_tf_example(self, names_opt: Option<HashSet<&str>>) -> ToTfExample<Self, &str> {
        ToTfExample {
            iter: self,
            names_opt,
        }
    }

    fn decode_image<S>(self, formats_opt: Option<HashMap<S, Option<ImageFormat>>>) -> DecodeImage<Self, S> {
        DecodeImage {
            iter: self,
            formats_opt,
        }
    }

    fn par_decode_image<S>(
        self, formats_opt: Option<HashMap<S, Option<ImageFormat>>>,
        buf_size: usize,
    ) -> ParallelDecodeImage where
        Self: 'static + Iterator<Item=ExampleType> + Send,
        S: 'static + AsRef<str> + Hash + Eq + Display + Sync + Send {

        let (sender, receiver) = crossbeam::channel::bounded(buf_size);

        let worker = thread::spawn(move || {
            debug!("Consumer thread started for decode_image()");

            let iter = self.par_bridge()
                .map(move |example| decode_image_on_example(example, &formats_opt));

            iter.for_each(|val| {
                debug!("{} elements buffered in parallel image decoding queue (producer)", sender.len());
                sender.send(Some(val)).unwrap();
            });
            sender.send(None).unwrap();

            debug!("Consumer thread ended for decode_image()");
        });

        ParallelDecodeImage {
            worker_opt: Some(worker),
            receiver,
        }
    }

    fn to_torch_tensor(self, names_opt: Option<HashSet<&str>>) -> ToTorchTensor<Self, &str> {
        ToTorchTensor {
            iter: self,
            names_opt,
            dummy_name: PhantomData,
        }
    }

    fn to_tf_tensor(self) -> ToTfTensor<Self> where
        Self: Sized {

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
        Self: Sized {

        UnwrapResult {
            iter: self,
            dummy_value: PhantomData,
            dummy_error: PhantomData,
        }
    }

    fn unwrap_ok<V, E>(self) -> UnwrapOk<Self, V, E> where
        Self: Sized {

        UnwrapOk {
            iter: self,
            dummy_value: PhantomData,
            dummy_error: PhantomData,
        }
    }

    fn shuffle(self, buf_size: usize) -> Shuffle<Self, StdRng> {

        let buffer = VecDeque::with_capacity(buf_size);
        let rng = StdRng::from_entropy();

        Shuffle {
            iter: self,
            buffer,
            rng,
        }
    }

    fn prefetch(mut self, buf_size: usize) -> Prefetch<Self> where
        Self: 'static + Sync + Send,
        Self::Item: 'static + Sync + Send, {

        let (sender, receiver) = crossbeam::channel::bounded(buf_size);

        let worker = thread::spawn(move || {
            debug!("Producer thread started for prefetch()");
            loop {
                debug!("{} elements buffered in prefetch queue (sender)", sender.len());
                match self.next() {
                    None => {
                        sender.send(None).unwrap();
                        debug!("Producer thread ended for prefetch()");
                        return;
                    }
                    Some(val) => {
                        sender.send(Some(val)).unwrap();
                    }
                }
            }
        });

        Prefetch {
            worker_opt: Some(worker),
            receiver,
        }
    }

    fn load_by_tfrecord_index(self, loader: loader::IndexedLoader) -> LoadByTfRecordIndex<Self> {

        LoadByTfRecordIndex {
            iter: self,
            loader,
        }
    }
}

// Struct definitions

#[derive(Clone)]
pub struct ToTfExample<I, S> {

    names_opt: Option<HashSet<S>>,
    iter: I,
}

pub struct ToTorchTensor<I, S> {

    iter: I,
    names_opt: Option<HashSet<S>>,
    dummy_name: PhantomData<S>,
}

#[derive(Clone)]
pub struct ToTfTensor<I> {

    iter: I,
}

#[derive(Clone)]
pub struct FilterHashMapEntry<I, K> {

    keys: HashSet<K>,
    iter: I,
}

#[derive(Clone)]
pub struct UnwrapResult<I, V, E> {

    iter: I,
    dummy_value: PhantomData<V>,
    dummy_error: PhantomData<E>,
}

#[derive(Clone)]
pub struct UnwrapOk<I, V, E> {

    iter: I,
    dummy_value: PhantomData<V>,
    dummy_error: PhantomData<E>,
}

#[derive(Clone)]
pub struct DecodeImage<I, S> {

    formats_opt: Option<HashMap<S, Option<ImageFormat>>>,
    iter: I,
}

pub struct ParallelDecodeImage {
    receiver: Receiver<Option<Result<ExampleType, ErrorType>>>,
    worker_opt: Option<JoinHandle<()>>,
}

#[derive(Clone)]
pub struct Shuffle<I: Iterator, R: rand::Rng> {
    iter: I,
    buffer: VecDeque<I::Item>,
    rng: R,
}

pub struct Prefetch<I: Iterator> {

    receiver: Receiver<Option<I::Item>>,
    worker_opt: Option<JoinHandle<()>>,
}

pub struct LoadByTfRecordIndex<I> {

    iter: I,
    loader: loader::IndexedLoader,
}

// impl

impl<T> DsIterator for T where
    T: Iterator, {
}

impl<I, S> Iterator for ToTfExample<I, S> where
        I: Iterator<Item=Vec<u8>>,
        S: AsRef<str> + Hash + Eq + Display, {

    type Item = Result<ExampleType, ErrorType>;

    fn next(&mut self) -> Option<Self::Item> {

        let buf = match self.iter.next() {
            None => return None,
            Some(buf) => buf,
        };

        let example = match parser::parse_single_example(&buf) {
            Err(e) => return Some(Err(Box::new(e))),
            Ok(example) => example,
        };

        let (_, entries) = match filter_entries(example, &self.names_opt) {
            Ok(ret) => ret,
            Err(err) => return Some(Err(Box::new(err))),
        };

        let mut result = HashMap::new();
        for (name, value) in entries {
            let parsed_value: FeatureType = match value {
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

    fn try_convert_to_tensor(name: &str, value_ref: FeatureType) -> Result<Box<dyn Any>, ErrorType> {

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

        try_convert_array_to_torch!(value_ref, Array1<u8>);
        try_convert_array_to_torch!(value_ref, Array1<f32>);
        try_convert_array_to_torch!(value_ref, Array1<f64>);
        try_convert_array_to_torch!(value_ref, Array1<i32>);
        try_convert_array_to_torch!(value_ref, Array1<i64>);

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

        try_convert_array_vec_to_torch!(value_ref, Array1<u8>);
        try_convert_array_vec_to_torch!(value_ref, Array1<f32>);
        try_convert_array_vec_to_torch!(value_ref, Array1<f64>);
        try_convert_array_vec_to_torch!(value_ref, Array1<i32>);
        try_convert_array_vec_to_torch!(value_ref, Array1<i64>);

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

        try_convert_array_to_torch!(value_ref, ArrayView1<u8>);
        try_convert_array_to_torch!(value_ref, ArrayView1<f32>);
        try_convert_array_to_torch!(value_ref, ArrayView1<f64>);
        try_convert_array_to_torch!(value_ref, ArrayView1<i32>);
        try_convert_array_to_torch!(value_ref, ArrayView1<i64>);

        try_convert_array_to_torch!(value_ref, ArrayView2<u8>);
        try_convert_array_to_torch!(value_ref, ArrayView2<f32>);
        try_convert_array_to_torch!(value_ref, ArrayView2<f64>);
        try_convert_array_to_torch!(value_ref, ArrayView2<i32>);
        try_convert_array_to_torch!(value_ref, ArrayView2<i64>);

        try_convert_array_to_torch!(value_ref, ArrayView3<u8>);
        try_convert_array_to_torch!(value_ref, ArrayView3<f32>);
        try_convert_array_to_torch!(value_ref, ArrayView3<f64>);
        try_convert_array_to_torch!(value_ref, ArrayView3<i32>);
        try_convert_array_to_torch!(value_ref, ArrayView3<i64>);

        try_convert_array_to_torch!(value_ref, ArrayView4<u8>);
        try_convert_array_to_torch!(value_ref, ArrayView4<f32>);
        try_convert_array_to_torch!(value_ref, ArrayView4<f64>);
        try_convert_array_to_torch!(value_ref, ArrayView4<i32>);
        try_convert_array_to_torch!(value_ref, ArrayView4<i64>);

        try_convert_array_vec_to_torch!(value_ref, ArrayView1<u8>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView1<f32>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView1<f64>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView1<i32>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView1<i64>);

        try_convert_array_vec_to_torch!(value_ref, ArrayView2<u8>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView2<f32>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView2<f64>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView2<i32>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView2<i64>);

        try_convert_array_vec_to_torch!(value_ref, ArrayView3<u8>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView3<f32>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView3<f64>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView3<i32>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView3<i64>);

        try_convert_array_vec_to_torch!(value_ref, ArrayView4<u8>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView4<f32>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView4<f64>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView4<i32>);
        try_convert_array_vec_to_torch!(value_ref, ArrayView4<i64>);


        let err = ParseError::new(&format!("The type of feature with name \"{}\" is not supported to convert to Torch Tensor", name));
        Err(Box::new(err))
    }
}

impl<I, S> Iterator for ToTorchTensor<I, S> where
    I: Iterator<Item=ExampleType>,
    S: AsRef<str> + Hash + Eq + Display {
    type Item = Result<HashMap<String, Box<dyn Any>>, ErrorType>;

    fn next(&mut self) -> Option<Self::Item> {
        let example = match self.iter.next() {
            None => return None,
            Some(example) => example,
        };

        let (mut remaining_example, entries) = match filter_entries(example, &self.names_opt) {
            Ok(ret) => ret,
            Err(err) => return Some(Err(Box::new(err))),
        };

        let mut result = HashMap::<String, Box<dyn Any>>::new();
        for (name, val) in remaining_example.drain() {
            result.insert(name, val);
        }

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
//     I: Iterator<Item=HashMap<&'a str, Box<dyn Any + Sync + Send>>>
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
    K: Hash + Eq {

    type Item = HashMap<K, V>;

    fn next(&mut self) -> Option<Self::Item> {

        match self.iter.next() {

            None => None,
            Some(mut index) => {
                let new_index: HashMap<K, V> = self.keys.iter().filter_map(|query_key| {
                    match index.remove_entry(&query_key) {

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
    I: Iterator<Item=Result<V, E>>, {

    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {

        match self.iter.next() {

            None => None,
            Some(result) => Some(result.ok().unwrap())
        }
    }
}

impl<I, V, E> Iterator for UnwrapResult<I, V, E> where
    I: Iterator<Item=Result<V, E>>,
    E: Debug + Sync + Send, {

    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {

        match self.iter.next() {

            None => None,
            Some(result) => Some(result.unwrap())
        }
    }
}

impl<I> Iterator for LoadByTfRecordIndex<I> where
    I: Iterator<Item=loader::RecordIndex>, {

    type Item = Result<Vec<u8>, io::Error>;

    fn next(&mut self) -> Option<Self::Item> {

        match self.iter.next() {

            None => None,
            Some(index) => match self.loader.fetch(index) {
                Some(record) => Some(Ok(record)),
                None => Some(Err(make_load_index_error())),
            }
        }
    }
}

impl<I, S> Iterator for DecodeImage<I, S> where
    I: Iterator<Item=ExampleType>,
    S: AsRef<str> + Hash + Eq + Display {

    type Item = Result<ExampleType, ErrorType>;

    fn next(&mut self) -> Option<Self::Item> {
        let example = match self.iter.next() {
            None => return None,
            Some(example) => example,
        };
        Some(decode_image_on_example(example, &self.formats_opt))
    }
}

impl Iterator for ParallelDecodeImage {

    type Item = Result<ExampleType, ErrorType>;

    fn next(&mut self) -> Option<Self::Item> {
        if let None = self.worker_opt {
            return None;
        }

        debug!("{} elements buffered in parallel image decoding queue (consumer)", self.receiver.len());
        match self.receiver.recv().unwrap() {
            None => {
                debug!("Reach end of stream and stop parallel image decoding");
                self.worker_opt.take().unwrap().join();
                None
            }
            Some(val) => Some(val),
        }
    }
}


impl<I, R> Iterator for Shuffle<I, R> where
    I: Iterator,
    R: rand::Rng {
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
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

            debug!("{} elements in shuffle buffer", self.buffer.len());
            self.buffer.pop_back()
        }
        else {
            self.iter.next()
        }
    }
}

impl<I> Iterator for Prefetch<I> where
    I: Iterator + Sync + Send,
    I::Item: Sync + Send {

    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if let None = self.worker_opt {
            return None;
        }

        debug!("{} elements buffered in prefetch queue (consumer)", self.receiver.len());
        match self.receiver.recv().unwrap() {
            None => {
                debug!("Reach end of stream and stop prefetching");
                self.worker_opt.take().unwrap().join();
                None
            }
            Some(val) => Some(val),
        }
    }
}
