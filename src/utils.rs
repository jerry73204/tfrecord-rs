use std::io::Cursor;
use std::any::Any;
use std::borrow::Borrow;
use std::cmp::Eq;
use std::hash::Hash;
use std::mem::transmute;
use std::panic::catch_unwind;
use std::fmt::Display;
use std::collections::{HashMap, HashSet};
use ndarray::{self, ArrayD, Array1, Array2, Array3, Array4,
              ArrayViewD, ArrayView1, ArrayView2, ArrayView3, ArrayView4,
              Axis};
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
use failure::Fallible;
use crate::{ExampleType, FeatureType};
use crate::parse::{parse_single_example, FeatureList};
use crate::error::{
    ItemNotFoundError,
    UnsuportedImageFormatError,
    UnsuportedValueTypeError,
    InconsistentValueTypeError,
};

pub fn bytes_to_example<'a>(
    buf: &[u8],
    names_opt: Option<HashSet<&str>>
) -> Fallible<ExampleType>
{
    let example = parse_single_example(buf.borrow())?;
    let (_, entries) = filter_entries(example, names_opt)?;

    let mut result = HashMap::new();
    for (name, value) in entries {
        let parsed_value: FeatureType = match value {
            FeatureList::Bytes(val) => Box::new(val),
            FeatureList::F32(val) => Box::new(val),
            FeatureList::I64(val) => Box::new(val),
        };

        result.insert(name, parsed_value);
    }

    Ok(result)
}

pub fn filter_entries<V>(
    mut map: HashMap<String, V>,
    names_opt: Option<HashSet<&str>>
) -> Fallible<(HashMap<String, V>, Vec<(String, V)>)>
{
    let mut new_map = HashMap::new();
    let entries: Vec<_> = match names_opt {
        Some(names) => {
            let mut entries = Vec::new();
            for name in names {
                let entry = match map.remove_entry(name as &str) {
                    Some(entry) => entry,
                    None => {
                        return Err(ItemNotFoundError { name: name.to_owned() }.into());
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


fn try_decode_image(bytes: &[u8], format_opt: Option<ImageFormat>) -> Fallible<Array3<u8>> {
    let format = match format_opt {
        Some(format) => format,
        None => {
            image::guess_format(bytes)?
        }
    };

    let (image, (width, height, channels)) = match format {
        ImageFormat::PNG => {
            let decoder = PNGDecoder::new(bytes)?;
            let (width, height) = decoder.dimensions();
            let image = decoder.read_image()?;
            (image, (width as usize, height as usize, 3))
        }
        ImageFormat::JPEG => {
            let (image, dims) = decode_jpeg(&bytes)?;
            (image, dims)
        }
        ImageFormat::GIF => {
            let decoder = GIFDecoder::new(bytes)?;
            let (width, height) = decoder.dimensions();
            let image = decoder.read_image()?;
            (image, (width as usize, height as usize, 3))
        }
        ImageFormat::WEBP => {
            let decoder = WebpDecoder::new(bytes)?;
            let (width, height) = decoder.dimensions();
            let image = decoder.read_image()?;
            (image, (width as usize, height as usize, 3))
        }
        ImageFormat::PNM => {
            let decoder = PNMDecoder::new(bytes)?;
            let (width, height) = decoder.dimensions();
            let image = decoder.read_image()?;
            (image, (width as usize, height as usize, 3))
        }
        ImageFormat::TIFF => {
            let decoder = TIFFDecoder::new(Cursor::new(bytes))?;
            let (width, height) = decoder.dimensions();
            let image = decoder.read_image()?;
            (image, (width as usize, height as usize, 3))
        }
        ImageFormat::TGA => {
            let decoder = TGADecoder::new(Cursor::new(bytes))?;
            let (width, height) = decoder.dimensions();
            let image = decoder.read_image()?;
            (image, (width as usize, height as usize, 3))
        }
        ImageFormat::BMP => {
            let decoder = BMPDecoder::new(Cursor::new(bytes))?;
            let (width, height) = decoder.dimensions();
            let image = decoder.read_image()?;
            (image, (width as usize, height as usize, 3))
        }
        ImageFormat::ICO => {
            let decoder = ICODecoder::new(Cursor::new(bytes))?;
            let (width, height) = decoder.dimensions();
            let image = decoder.read_image()?;
            (image, (width as usize, height as usize, 3))
        }
        _ => {
            return Err(UnsuportedImageFormatError.into());
        }
    };

    let array = Array3::from_shape_vec((height, width, channels), image)?;
    Ok(array)
}

fn decode_jpeg(data: &[u8]) -> Fallible<(Vec<u8>, (usize, usize, usize))> {
    catch_unwind(|| -> Fallible<_> {
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
    formats_opt: Option<HashMap<S, Option<ImageFormat>>>,
) -> Fallible<ExampleType> where
    S: AsRef<str> + Hash + Eq + Display + Send
{
    let mut result = ExampleType::new();
    let entries = match formats_opt {
        Some(formats) => {
            let mut entries = Vec::new();
            for (select_name, format_opt) in formats {
                let select_name_ = select_name.as_ref();
                let (name, value_ref) = match example.remove_entry(select_name_) {
                    Some(entry) => entry,
                    None => {
                        return Err(ItemNotFoundError { name: select_name_.to_owned() }.into());
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
            let array = try_decode_image(bytes, format_opt)?;
            result.insert(name, Box::new(array));
        }
        else if let Some(bytes_list) = value_ref.downcast_ref::<Vec<Vec<u8>>>() {
            let mut images = Vec::new();
            for bytes in bytes_list {
                let array = try_decode_image(bytes, format_opt)?;
                images.push(array);
            }
            result.insert(name, Box::new(images));
        }
        else {
            return Err(UnsuportedValueTypeError.into());
        }
    }

    Ok(result)
}

macro_rules! try_convert_array_to_torch (
    ( $value_ref:ident, $device:ident, $dtype:ty ) => (
        match $value_ref.downcast_ref::<$dtype>() {
            None => {}
            Some(val) => {
                let dims = &val.shape()
                    .into_iter()
                    .map(|v| *v as i64)
                    .collect::<Vec<_>>();
                let tensor = tch::Tensor::of_slice(val.to_owned().into_raw_vec().as_slice())
                    .to_device($device)
                    .view(dims);
                return Ok(Box::new(tensor));
            }
        }
    )
);

macro_rules! try_convert_arrayview_to_torch (
    ( $value_ref:ident, $device:ident, $dtype:ty ) => (
        match $value_ref.downcast_ref::<$dtype>() {
            None => {}
            Some(val) => {
                let dims = &val.shape()
                    .into_iter()
                    .map(|v| *v as i64)
                    .collect::<Vec<_>>();
                let tensor = tch::Tensor::of_slice(val.to_owned().into_raw_vec().as_slice())
                    .to_device($device)
                    .view(dims);
                return Ok(Box::new(tensor));
            }
        }
    )
);

macro_rules! try_convert_array_vec_to_torch (
    ( $value_ref:ident, $device:ident, $dtype:ty ) => (
        match $value_ref.downcast_ref::<Vec<$dtype>>() {
            None => {}
            Some(list) => {
                let tensor_list = list.into_iter()
                    .map(|val| {
                        let dims = &val.shape()
                            .into_iter()
                            .map(|v| *v as i64)
                            .collect::<Vec<_>>();
                        let tensor = tch::Tensor::of_slice(&val.to_owned().into_raw_vec().as_slice())
                            .to_device($device)
                            .view(dims);
                        tensor
                    })
                    .collect::<Vec<_>>();
                return Ok(Box::new(tensor_list));
            }
        }
    )
);

macro_rules! try_convert_arrayview_vec_to_torch (
    ( $value_ref:ident, $device:ident, $dtype:ty ) => (
        match $value_ref.downcast_ref::<Vec<$dtype>>() {
            None => {}
            Some(list) => {
                let tensor_list = list.into_iter()
                    .map(|val| {
                        let dims = &val.shape()
                            .into_iter()
                            .map(|v| *v as i64)
                            .collect::<Vec<_>>();
                        let tensor = tch::Tensor::of_slice(&val.to_owned().into_raw_vec().as_slice())
                            .to_device($device)
                            .view(dims);
                        tensor
                    })
                    .collect::<Vec<_>>();
                return Ok(Box::new(tensor_list));
            }
        }
    )
);

macro_rules! try_convert_vec_to_torch (
    ( $value_ref:ident, $device:ident, $dtype:ty ) => (
        match $value_ref.downcast_ref::<Vec<$dtype>>() {
            None => {}
            Some(val) => {
                return Ok(Box::new(tch::Tensor::of_slice(val).to_device($device)));
            }
        }
    )
);

macro_rules! try_convert_vec_vec_to_torch (
    ( $value_ref:ident, $device:ident, $dtype:ty ) => (
        match $value_ref.downcast_ref::<Vec<Vec<$dtype>>>() {
            None => {}
            Some(list) => {
                let tensor_list = list.into_iter()
                    .map(|val| tch::Tensor::of_slice(val).to_device($device))
                    .collect::<Vec<_>>();
                return Ok(Box::new(tensor_list));
            }
        }
    )
);

fn try_convert_to_tensor(
    name: &str,
    value_ref: FeatureType,
    device: tch::Device
) -> Fallible<FeatureType>
{

    // TODO: optimize type matching
    try_convert_vec_to_torch!(value_ref, device, u8);
    try_convert_vec_to_torch!(value_ref, device, i32);
    try_convert_vec_to_torch!(value_ref, device, i64);
    try_convert_vec_to_torch!(value_ref, device, f32);
    try_convert_vec_to_torch!(value_ref, device, f64);

    try_convert_vec_vec_to_torch!(value_ref, device, u8);
    try_convert_vec_vec_to_torch!(value_ref, device, i32);
    try_convert_vec_vec_to_torch!(value_ref, device, i64);
    try_convert_vec_vec_to_torch!(value_ref, device, f32);
    try_convert_vec_vec_to_torch!(value_ref, device, f64);

    try_convert_array_to_torch!(value_ref, device, ArrayD<u8>);
    try_convert_array_to_torch!(value_ref, device, ArrayD<f32>);
    try_convert_array_to_torch!(value_ref, device, ArrayD<f64>);
    try_convert_array_to_torch!(value_ref, device, ArrayD<i32>);
    try_convert_array_to_torch!(value_ref, device, ArrayD<i64>);

    try_convert_array_to_torch!(value_ref, device, Array1<u8>);
    try_convert_array_to_torch!(value_ref, device, Array1<f32>);
    try_convert_array_to_torch!(value_ref, device, Array1<f64>);
    try_convert_array_to_torch!(value_ref, device, Array1<i32>);
    try_convert_array_to_torch!(value_ref, device, Array1<i64>);

    try_convert_array_to_torch!(value_ref, device, Array2<u8>);
    try_convert_array_to_torch!(value_ref, device, Array2<f32>);
    try_convert_array_to_torch!(value_ref, device, Array2<f64>);
    try_convert_array_to_torch!(value_ref, device, Array2<i32>);
    try_convert_array_to_torch!(value_ref, device, Array2<i64>);

    try_convert_array_to_torch!(value_ref, device, Array3<u8>);
    try_convert_array_to_torch!(value_ref, device, Array3<f32>);
    try_convert_array_to_torch!(value_ref, device, Array3<f64>);
    try_convert_array_to_torch!(value_ref, device, Array3<i32>);
    try_convert_array_to_torch!(value_ref, device, Array3<i64>);

    try_convert_array_to_torch!(value_ref, device, Array4<u8>);
    try_convert_array_to_torch!(value_ref, device, Array4<f32>);
    try_convert_array_to_torch!(value_ref, device, Array4<f64>);
    try_convert_array_to_torch!(value_ref, device, Array4<i32>);
    try_convert_array_to_torch!(value_ref, device, Array4<i64>);

    try_convert_array_vec_to_torch!(value_ref, device, ArrayD<u8>);
    try_convert_array_vec_to_torch!(value_ref, device, ArrayD<f32>);
    try_convert_array_vec_to_torch!(value_ref, device, ArrayD<f64>);
    try_convert_array_vec_to_torch!(value_ref, device, ArrayD<i32>);
    try_convert_array_vec_to_torch!(value_ref, device, ArrayD<i64>);

    try_convert_array_vec_to_torch!(value_ref, device, Array1<u8>);
    try_convert_array_vec_to_torch!(value_ref, device, Array1<f32>);
    try_convert_array_vec_to_torch!(value_ref, device, Array1<f64>);
    try_convert_array_vec_to_torch!(value_ref, device, Array1<i32>);
    try_convert_array_vec_to_torch!(value_ref, device, Array1<i64>);

    try_convert_array_vec_to_torch!(value_ref, device, Array2<u8>);
    try_convert_array_vec_to_torch!(value_ref, device, Array2<f32>);
    try_convert_array_vec_to_torch!(value_ref, device, Array2<f64>);
    try_convert_array_vec_to_torch!(value_ref, device, Array2<i32>);
    try_convert_array_vec_to_torch!(value_ref, device, Array2<i64>);

    try_convert_array_vec_to_torch!(value_ref, device, Array3<u8>);
    try_convert_array_vec_to_torch!(value_ref, device, Array3<f32>);
    try_convert_array_vec_to_torch!(value_ref, device, Array3<f64>);
    try_convert_array_vec_to_torch!(value_ref, device, Array3<i32>);
    try_convert_array_vec_to_torch!(value_ref, device, Array3<i64>);

    try_convert_array_vec_to_torch!(value_ref, device, Array4<u8>);
    try_convert_array_vec_to_torch!(value_ref, device, Array4<f32>);
    try_convert_array_vec_to_torch!(value_ref, device, Array4<f64>);
    try_convert_array_vec_to_torch!(value_ref, device, Array4<i32>);
    try_convert_array_vec_to_torch!(value_ref, device, Array4<i64>);

    try_convert_arrayview_to_torch!(value_ref, device, ArrayViewD<u8>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayViewD<f32>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayViewD<f64>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayViewD<i32>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayViewD<i64>);

    try_convert_arrayview_to_torch!(value_ref, device, ArrayView1<u8>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView1<f32>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView1<f64>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView1<i32>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView1<i64>);

    try_convert_arrayview_to_torch!(value_ref, device, ArrayView2<u8>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView2<f32>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView2<f64>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView2<i32>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView2<i64>);

    try_convert_arrayview_to_torch!(value_ref, device, ArrayView3<u8>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView3<f32>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView3<f64>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView3<i32>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView3<i64>);

    try_convert_arrayview_to_torch!(value_ref, device, ArrayView4<u8>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView4<f32>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView4<f64>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView4<i32>);
    try_convert_arrayview_to_torch!(value_ref, device, ArrayView4<i64>);

    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayViewD<u8>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayViewD<f32>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayViewD<f64>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayViewD<i32>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayViewD<i64>);

    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView1<u8>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView1<f32>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView1<f64>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView1<i32>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView1<i64>);

    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView2<u8>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView2<f32>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView2<f64>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView2<i32>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView2<i64>);

    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView3<u8>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView3<f32>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView3<f64>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView3<i32>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView3<i64>);

    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView4<u8>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView4<f32>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView4<f64>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView4<i32>);
    try_convert_arrayview_vec_to_torch!(value_ref, device, ArrayView4<i64>);

    Err(UnsuportedValueTypeError.into())
}

pub fn example_to_torch_tensor(
    example: ExampleType,
    names_opt: Option<HashSet<&str>>,
    device: tch::Device,
) -> Fallible<ExampleType>
{
    let (mut remaining_example, entries) = filter_entries(example, names_opt)?;

    let mut result = ExampleType::new();
    for (name, val) in remaining_example.drain() {
        result.insert(name, val);
    }

    for (name, feature_ref) in entries {
        let ret = try_convert_to_tensor(&name, feature_ref, device)?;
        result.insert(name, ret);
    }

    Ok(result)
}

macro_rules! try_make_batch_array (
    ( $name:ident, $features:ident, $dtype:ty ) => (
        let correct_guess = match $features[0].downcast_ref::<$dtype>() {
            None => false,
            Some(_) => true,
        };

        if correct_guess {
            let mut arrays = Vec::new();

            for array_ref in $features.drain(..) {
                match array_ref.downcast_ref::<$dtype>() {
                    None => {
                        return Err(InconsistentValueTypeError.into());
                    },
                    Some(array) => {
                        let new_array = array.to_owned().insert_axis(Axis(0));
                        arrays.push(new_array);
                    },
                };
            }

            let views = arrays.iter()
                .map(|array| array.view())
                .collect::<Vec<_>>();

            let result = ndarray::stack(Axis(0), &views).unwrap();
            return Ok(Box::new(result));
        }
    )
);

macro_rules! try_make_batch_arrayview (
    ( $name:ident, $features:ident, $dtype:ty ) => (
        let correct_guess = match $features[0].downcast_ref::<$dtype>() {
            None => false,
            Some(_) => true,
        };

        if correct_guess {
            let mut arrays = Vec::new();

            for array_ref in $features.drain(..) {
                match array_ref.downcast_ref::<$dtype>() {
                    None => {
                        return Err(InconsistentValueTypeError.into());
                    },
                    Some(array) => {
                        let new_array = array.to_owned().insert_axis(Axis(0));
                        arrays.push(new_array);
                    },
                };
            }

            let views = arrays.iter()
                .map(|array| array.view())
                .collect::<Vec<_>>();

            let result = ndarray::stack(Axis(0), &views).unwrap();
            return Ok(Box::new(result));
        }
    )
);

fn try_make_batch(name: &str, mut features: Vec<FeatureType>) -> Fallible<FeatureType> {
    try_make_batch_array!(name, features, ArrayD<u8>);
    try_make_batch_array!(name, features, ArrayD<f32>);
    try_make_batch_array!(name, features, ArrayD<f64>);
    try_make_batch_array!(name, features, ArrayD<i32>);
    try_make_batch_array!(name, features, ArrayD<i64>);

    try_make_batch_array!(name, features, Array1<u8>);
    try_make_batch_array!(name, features, Array1<f32>);
    try_make_batch_array!(name, features, Array1<f64>);
    try_make_batch_array!(name, features, Array1<i32>);
    try_make_batch_array!(name, features, Array1<i64>);

    try_make_batch_array!(name, features, Array2<u8>);
    try_make_batch_array!(name, features, Array2<f32>);
    try_make_batch_array!(name, features, Array2<f64>);
    try_make_batch_array!(name, features, Array2<i32>);
    try_make_batch_array!(name, features, Array2<i64>);

    try_make_batch_array!(name, features, Array3<u8>);
    try_make_batch_array!(name, features, Array3<f32>);
    try_make_batch_array!(name, features, Array3<f64>);
    try_make_batch_array!(name, features, Array3<i32>);
    try_make_batch_array!(name, features, Array3<i64>);

    try_make_batch_array!(name, features, Array4<u8>);
    try_make_batch_array!(name, features, Array4<f32>);
    try_make_batch_array!(name, features, Array4<f64>);
    try_make_batch_array!(name, features, Array4<i32>);
    try_make_batch_array!(name, features, Array4<i64>);

    try_make_batch_arrayview!(name, features, ArrayViewD<u8>);
    try_make_batch_arrayview!(name, features, ArrayViewD<f32>);
    try_make_batch_arrayview!(name, features, ArrayViewD<f64>);
    try_make_batch_arrayview!(name, features, ArrayViewD<i32>);
    try_make_batch_arrayview!(name, features, ArrayViewD<i64>);

    try_make_batch_arrayview!(name, features, ArrayView1<u8>);
    try_make_batch_arrayview!(name, features, ArrayView1<f32>);
    try_make_batch_arrayview!(name, features, ArrayView1<f64>);
    try_make_batch_arrayview!(name, features, ArrayView1<i32>);
    try_make_batch_arrayview!(name, features, ArrayView1<i64>);

    try_make_batch_arrayview!(name, features, ArrayView2<u8>);
    try_make_batch_arrayview!(name, features, ArrayView2<f32>);
    try_make_batch_arrayview!(name, features, ArrayView2<f64>);
    try_make_batch_arrayview!(name, features, ArrayView2<i32>);
    try_make_batch_arrayview!(name, features, ArrayView2<i64>);

    try_make_batch_arrayview!(name, features, ArrayView3<u8>);
    try_make_batch_arrayview!(name, features, ArrayView3<f32>);
    try_make_batch_arrayview!(name, features, ArrayView3<f64>);
    try_make_batch_arrayview!(name, features, ArrayView3<i32>);
    try_make_batch_arrayview!(name, features, ArrayView3<i64>);

    try_make_batch_arrayview!(name, features, ArrayView4<u8>);
    try_make_batch_arrayview!(name, features, ArrayView4<f32>);
    try_make_batch_arrayview!(name, features, ArrayView4<f64>);
    try_make_batch_arrayview!(name, features, ArrayView4<i32>);
    try_make_batch_arrayview!(name, features, ArrayView4<i64>);

    return Err(UnsuportedValueTypeError.into());
}

pub fn make_batch(
    mut examples: Vec<ExampleType>,
) -> Fallible<ExampleType> {

    let names: Vec<_> = examples[0].keys()
        .map(|key| key.to_owned())
        .collect();
    let mut result = ExampleType::new();

    for name in names {
        let values = examples.iter_mut()
            .map(|example| example.remove(&name).unwrap())
            .collect();
        let batch =try_make_batch(&name, values)?;
        result.insert(name, batch);
    }

    Ok(result)
}
