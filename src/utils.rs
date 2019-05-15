use std::io::{self, Cursor};
use std::any::Any;
use std::borrow::Borrow;
use std::cmp::Eq;
use std::hash::Hash;
use std::mem::transmute;
use std::panic::catch_unwind;
use std::fmt::Display;
use std::collections::HashMap;
use ndarray::{ArrayBase, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4};
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
use crate::{ExampleType, FeatureType, ErrorType, NonSyncExampleType};
use crate::parser;
use crate::error::ParseError;

pub fn bytes_to_example<'a, B, N>(buf: B, names_opt: Option<N>) -> Result<ExampleType, ErrorType> where
    B: Borrow<[u8]>,
    N: Borrow<&'a [&'a str]>,
{
    let example = match parser::parse_single_example(buf.borrow()) {
        Err(e) => return Err(Box::new(e)),
        Ok(example) => example,
    };

    let names = match names_opt {
        Some(names) => Some(*names.borrow()),
        None => None,
    };

    let (_, entries) = match filter_entries(example, names) {
        Ok(ret) => ret,
        Err(err) => return Err(Box::new(err)),
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

    Ok(result)
}

pub fn filter_entries<'a, V, N>(
    mut map: HashMap<String, V>,
    names_opt: Option<N>
) -> Result<(HashMap<String, V>, Vec<(String, V)>), ParseError> where
    N: Borrow<&'a [&'a str]>,
{
    let mut new_map = HashMap::new();
    let entries: Vec<_> = match names_opt {
        Some(names) => {
            let mut entries = Vec::new();
            for name in *names.borrow() {
                let name_ref: &str = name;
                let entry = match map.remove_entry(name_ref) {
                    Some(entry) => entry,
                    None => {
                        let err = ParseError::new(&format!("Feature with name \"{}\" is not found", name));
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


pub fn try_decode_image(bytes: &[u8], format_opt: Option<ImageFormat>) -> Result<Array3<u8>, ErrorType> {
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

pub fn decode_jpeg(data: &[u8]) -> Result<(Vec<u8>, (usize, usize, usize)), io::Error> {
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

pub fn decode_image_on_example<S>(
    mut example: ExampleType,
    formats_opt: &Option<HashMap<S, Option<ImageFormat>>>,
) -> Result<ExampleType, ErrorType> where
    S: AsRef<str> + Hash + Eq + Display
{

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

pub fn try_convert_to_tensor(name: &str, value_ref: FeatureType) -> Result<Box<dyn Any>, ErrorType> {

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


pub fn example_to_torch_tensor<'a, S, N>(
    example: ExampleType,
    names_opt: Option<N>) -> Result<NonSyncExampleType, ErrorType> where
    N: Borrow<&'a [&'a str]>,
{
    let names = match names_opt {
        Some(names) => Some(*names.borrow()),
        None => None,
    };

    let (mut remaining_example, entries) = match filter_entries(example, names) {
        Ok(ret) => ret,
        Err(err) => return Err(Box::new(err)),
    };

    let mut result = HashMap::<String, Box<dyn Any>>::new();
    for (name, val) in remaining_example.drain() {
        result.insert(name, val);
    }

    for (name, feature_ref) in entries {
        let ret = match try_convert_to_tensor(&name, feature_ref) {
            Err(err) => return Err(err),
            Ok(ret) => ret,
        };
        result.insert(name, ret);
    }

    Ok(result)
}
