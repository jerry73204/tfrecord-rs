extern crate protobuf;
extern crate crc;

mod from_tf;

use std::error;
use std::path;
use std::fs;
use std::io;
use std::collections::HashMap;
use std::io::{Read, Seek};
use crc::crc32;
use byteorder::{ReadBytesExt, LittleEndian};
use from_tf::example::{Example, SequenceExample};
use from_tf::feature::Feature_oneof_kind;

pub enum FeatureValue
{
    Bytes(Vec<Vec<u8>>),
    Float32(Vec<f32>),
    Int64(Vec<i64>),
}

pub enum FeatureList
{
    BytesList(Vec<Vec<Vec<u8>>>),
    Float32List(Vec<Vec<f32>>),
    Int64List(Vec<Vec<i64>>),
}

pub fn parse_single_example(payload: &[u8]) -> Result<HashMap<String, FeatureValue>, Box<error::Error>>
{
    let mut example: Example = protobuf::parse_from_bytes(payload)?;
    let features = example.take_features().take_feature();
    let mut result = HashMap::<String, FeatureValue>::new();

    for (name, feature) in features
    {
        match feature.kind
        {
            Some(Feature_oneof_kind::bytes_list(mut val)) => {
                result.insert(name, FeatureValue::Bytes(val.take_value().into_vec()));
            }
            Some(Feature_oneof_kind::float_list(mut val)) => {
                result.insert(name, FeatureValue::Float32(val.take_value()));
            }
            Some(Feature_oneof_kind::int64_list(mut val)) => {
                result.insert(name, FeatureValue::Int64(val.take_value()));
            }
            None => (),
        }
    }

    Ok(result)
}

pub fn parse_single_sequence_example(payload: &[u8]) -> Result<(HashMap<String, FeatureValue>, HashMap<String, FeatureList>), Box<error::Error>>
{
    let mut seq_example: SequenceExample = protobuf::parse_from_bytes(payload)?;
    let context = seq_example.take_context().take_feature();
    let feature_list = seq_example.take_feature_lists().take_feature_list();

    let mut context_result = HashMap::<String, FeatureValue>::new();
    let mut feature_result = HashMap::<String, FeatureList>::new();

    for (name, feature) in context
    {
        match feature.kind
        {
            Some(Feature_oneof_kind::bytes_list(mut val)) => {
                context_result.insert(name, FeatureValue::Bytes(val.take_value().into_vec()));
            }
            Some(Feature_oneof_kind::float_list(mut val)) => {
                context_result.insert(name, FeatureValue::Float32(val.take_value()));
            }
            Some(Feature_oneof_kind::int64_list(mut val)) => {
                context_result.insert(name, FeatureValue::Int64(val.take_value()));
            }
            None => (),
        }
    }

    for (name, mut proto_list) in feature_list
    {
        let feat_vec = proto_list.take_feature().into_vec();

        if feat_vec.len() == 0
        {
            continue;
        }

        let mut values = match feat_vec[0].kind
        {
            Some(Feature_oneof_kind::bytes_list(_)) => {
                FeatureList::BytesList(Vec::<Vec<Vec<u8>>>::new())
            }
            Some(Feature_oneof_kind::float_list(_)) => {
                FeatureList::Float32List(Vec::<Vec<f32>>::new())
            }
            Some(Feature_oneof_kind::int64_list(_)) => {
                FeatureList::Int64List(Vec::<Vec<i64>>::new())
            }
            None => {
                continue;
            }
        };

        for feature in feat_vec
        {
            match feature.kind
            {
                Some(Feature_oneof_kind::bytes_list(mut val)) => {
                    if let FeatureList::BytesList(ref mut vals) = values
                    {
                        vals.push(val.take_value().into_vec());
                    }
                    else
                    {
                        return Err(Box::new(make_corrupted_error()));
                    }
                }
                Some(Feature_oneof_kind::float_list(mut val)) => {
                    if let FeatureList::Float32List(ref mut vals) = values
                    {
                        vals.push(val.take_value());
                    }
                    else
                    {
                        return Err(Box::new(make_corrupted_error()));
                    }
                }
                Some(Feature_oneof_kind::int64_list(mut val)) => {
                    if let FeatureList::Int64List(ref mut vals) = values
                    {
                        vals.push(val.take_value());
                    }
                    else
                    {
                        return Err(Box::new(make_corrupted_error()));
                    }
                }
                None => (),
            }
        }

        feature_result.insert(name, values);
    }

    Ok((context_result, feature_result))
}

pub fn build_record_index(path: &path::Path, check_integrity: bool) -> Result<Vec<(u64, u64)>, Box<error::Error>>
{
    let mut record_index: Vec<(u64, u64)> = Vec::new();
    let mut file = fs::File::open(path)?;

    let checksum = |buf: &[u8]| {
        let cksum = crc32::checksum_castagnoli(buf);
        ((cksum >> 15) | (cksum << 17)).wrapping_add(0xa282ead8u32)
    };

    let try_read_len = |file: &mut fs::File| -> Result<Option<u64>, Box<error::Error>> {
        let mut len_buf = [0u8; 8];

        match file.read(&mut len_buf)
        {
            Ok(0) => Ok(None),
            Ok(n) if n == len_buf.len() => {
                let len = (&len_buf[..]).read_u64::<LittleEndian>()?;

                if check_integrity
                {
                    let answer_cksum = file.read_u32::<LittleEndian>()?;
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
                    file.seek(io::SeekFrom::Current(4))?;
                    Ok(Some(len))
                }
            }
            Ok(_) => Err(Box::new(make_truncated_error())),
            Err(e) => Err(Box::new(e)),
        }
    };

    let verify_record_integrity = |file: &mut fs::File, len: u64| -> Result<(), Box<error::Error>> {
        let mut buf = Vec::<u8>::new();
        file.take(len).read_to_end(&mut buf)?;
        let answer_cksum = file.read_u32::<LittleEndian>()?;

        if answer_cksum == checksum(&buf.as_slice())
        {
            Ok(())
        }
        else
        {
            Err(Box::new(make_corrupted_error()))
        }
    };

    loop
    {
        match try_read_len(&mut file)?
        {
            None => break,
            Some(len) => {
                let offset = file.seek(io::SeekFrom::Current(0))?;
                if check_integrity
                {
                    verify_record_integrity(&mut file, len)?;
                }
                else
                {
                    file.seek(io::SeekFrom::Current(len as i64 + 4))?;
                }
                record_index.push((offset, len));
            }
        }
    }

    Ok(record_index)
}


fn make_corrupted_error() -> io::Error {
    io::Error::new(io::ErrorKind::Other, "corrupted error")
}

fn make_truncated_error() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "corrupted error")
}
