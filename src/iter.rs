use std::{io, ops, error, hash, cmp, marker, fmt};
use std::collections::{HashMap, HashSet};
use tch;
// use tensorflow as tf;
use crate::parser;
use crate::loader;
use crate::error::{ParseError, make_loading_error};

// Trait defiintions

pub trait DsIterator: Iterator
{
    fn into_tf_example(self, names: Option<HashSet<String>>) -> IntoTfExample<Self> where Self: Sized + Clone {
        IntoTfExample {
            iter: self,
            names,
        }
    }

    fn into_torch_tensor(self) -> IntoTorchTensor<Self> where Self: Sized + Clone {
        IntoTorchTensor {
            iter: self,
        }
    }

    fn into_tf_tensor(self) -> IntoTfTensor<Self> where Self: Sized + Clone {
        IntoTfTensor {
            iter: self,
        }
    }

    fn filter_hashmap_entry<K>(self, keys: HashSet<K>) -> FilterHashMapEntry<Self, K> where Self: Sized + Clone {
        FilterHashMapEntry {
            iter: self,
            keys,
        }
    }

    fn assert_ok<V, E>(self) -> AssertOk<Self, V, E> where Self: Sized + Clone {
        AssertOk {
            iter: self,
            dummy_value: marker::PhantomData,
            dummy_error: marker::PhantomData,
        }
    }

    fn load_by_tfrecord_index<V, E>(self, loader: loader::TFRecordLoader) -> LoadByTfRecordIndex<Self> where Self: Sized + Clone {
        LoadByTfRecordIndex {
            iter: self,
            loader,
        }
    }
}

// Struct definitions

type FeatureDict = HashMap<String, Feature>;

pub enum Feature
{
    BytesList(Vec<Vec<u8>>),
    F32List(Vec<f32>),
    I64List(Vec<i64>),
    BytesSeqList(Vec<Vec<Vec<u8>>>),
    F32SeqList(Vec<Vec<f32>>),
    I64SeqList(Vec<Vec<i64>>),
}

pub enum FeatureShape<'a>
{
    Fixed(Vec<i64>),
    FixedRef(&'a [i64]),
}

pub struct IntoTfExample<I>
{
    names: Option<HashSet<String>>,
    iter: I,
}

pub struct IntoTorchTensor<I>
{
    iter: I,
}

pub struct IntoTfTensor<I>
{
    iter: I,
}

pub struct FilterHashMapEntry<I, K>
{
    keys: HashSet<K>,
    iter: I,
}

pub struct AssertOk<I, V, E>
{
    iter: I,
    dummy_value: marker::PhantomData<V>,
    dummy_error: marker::PhantomData<E>,
}

pub struct LoadByTfRecordIndex<I>
{
    iter: I,
    loader: loader::TFRecordLoader,
}

// impl

impl<I> Iterator for IntoTfExample<I> where
    I: Clone + Iterator<Item=Vec<u8>>,
{
    type Item = Result<FeatureDict, Box<error::Error>>;

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
    I: Clone + Iterator<Item=FeatureDict>
{
    type Item = Result<HashMap<String, tch::Tensor>, ParseError>;

    fn next(&mut self) -> Option<Self::Item>
    {
        match self.iter.next()
        {
            None => None,
            Some(example) => {
                let tensor_map_result: Result<_, _> = example.into_iter()
                    .map(|(name, value)| {
                        // TODO: correctly convert values
                        let result = match value
                        {
                            // Feature::BytesList(val) => {
                            //     Err(ParseError::new("Cannot convert BytesList to tch::Tensor. Consider using decode_image()"))
                            // }
                            Feature::F32List(val) => {
                                Ok(tch::Tensor::of_slice(&val))
                            }
                            Feature::I64List(val) => {
                                Ok(tch::Tensor::of_slice(&val))
                            }
                            // Feature::BytesSeqList(val) => {
                            //     Err(ParseError::new("Cannot convert BytesList to tch::Tensor. Consider using decode_image()"))
                            // }
                            // Feature::F32SeqList(val) => {
                            //     Ok(tch::Tensor::of_slice(&val))
                            // }
                            // Feature::I64SeqList(val) => {
                            //     Ok(tch::Tensor::of_slice(&val))
                            // }
                            _ => Err(ParseError::new("Cannot convert BytesList to tch::Tensor. Consider using decode_image()")),
                        };

                        (name, result)
                    })
                    .fold(Ok(HashMap::new()), |overall_result, (name, result)| {
                        match overall_result
                        {
                            Err(e) => Err(e),
                            Ok(mut acc) => {
                                match result
                                {
                                    Err(e) => Err(e),
                                    Ok(tensor) => {
                                        acc.insert(name, tensor);
                                        Ok(acc)
                                    }
                                }
                            }
                        }
                    });

                Some(tensor_map_result)
            }
        }
    }
}


// TODO: implementation
// impl<'a, I> Iterator for IntoTfTensor<'a, I> where
//     I: Clone + Iterator<Item=FeatureDict>
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
    I: Clone + Iterator<Item=HashMap<K, V>>,
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

impl<I, V, E> Iterator for AssertOk<I, V, E> where
    I: Clone + Iterator<Item=Result<V, E>>,
    E: error::Error,
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
    I: Clone + Iterator<Item=loader::RecordIndex>,
{
    type Item = Result<Vec<u8>, io::Error>;

    fn next(&mut self) -> Option<Self::Item>
    {
        match self.iter.next()
        {
            None => None,
            Some(index) => match self.loader.fetch(index) {
                Some(record) => Some(Ok(record)),
                None => Some(Err(make_loading_error())),
            }
        }
    }
}
