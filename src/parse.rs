use std::collections::HashMap;
use failure::Fallible;
use crate::from_tf::example;
use crate::from_tf::event::{
    self,
    Event_oneof_what,
//     LogMessage,
//     LogMessage_Level,
//     SessionLog,
//     SessionLog_SessionStatus,
//     TaggedRunMetadata,
};
// use crate::from_tf::summary::{Summary, Summary_Value};
use crate::from_tf::feature::Feature_oneof_kind;
use crate::error::CorruptedRecordError;

pub type Example = HashMap<String, FeatureList>;
pub type SeqExample = (HashMap<String, FeatureList>, HashMap<String, FeatureSeqList>);

pub enum FeatureList {
    Bytes(Vec<Vec<u8>>),
    F32(Vec<f32>),
    I64(Vec<i64>),
}

pub enum FeatureSeqList {
    Bytes(Vec<Vec<Vec<u8>>>),
    F32(Vec<Vec<f32>>),
    I64(Vec<Vec<i64>>),
}

pub fn parse_single_example(payload: &[u8]) -> Fallible<Example> {
    let mut example: example::Example = protobuf::parse_from_bytes(payload)?;
    let features = example.take_features().take_feature();
    let mut result = HashMap::<String, FeatureList>::new();

    for (name, feature) in features {
        match feature.kind {
            Some(Feature_oneof_kind::bytes_list(mut val)) => {
                result.insert(name, FeatureList::Bytes(val.take_value().into_vec()));
            }
            Some(Feature_oneof_kind::float_list(mut val)) => {
                result.insert(name, FeatureList::F32(val.take_value()));
            }
            Some(Feature_oneof_kind::int64_list(mut val)) => {
                result.insert(name, FeatureList::I64(val.take_value()));
            }
            None => (),
        }
    }

    Ok(result)
}

pub fn parse_single_sequence_example(payload: &[u8]) -> Fallible<SeqExample> {
    let mut seq_example: example::SequenceExample = protobuf::parse_from_bytes(payload)?;
    let context = seq_example.take_context().take_feature();
    let feature_list = seq_example.take_feature_lists().take_feature_list();

    let mut context_result = HashMap::<String, FeatureList>::new();
    let mut feature_result = HashMap::<String, FeatureSeqList>::new();

    for (name, feature) in context {
        match feature.kind {
            Some(Feature_oneof_kind::bytes_list(mut val)) => {
                context_result.insert(name, FeatureList::Bytes(val.take_value().into_vec()));
            }
            Some(Feature_oneof_kind::float_list(mut val)) => {
                context_result.insert(name, FeatureList::F32(val.take_value()));
            }
            Some(Feature_oneof_kind::int64_list(mut val)) => {
                context_result.insert(name, FeatureList::I64(val.take_value()));
            }
            None => (),
        }
    }

    for (name, mut proto_list) in feature_list {
        let feat_vec = proto_list.take_feature().into_vec();

        if feat_vec.len() == 0 {
            continue;
        }

        let mut values = match feat_vec[0].kind {
            Some(Feature_oneof_kind::bytes_list(_)) => {
                FeatureSeqList::Bytes(Vec::<Vec<Vec<u8>>>::new())
            }
            Some(Feature_oneof_kind::float_list(_)) => {
                FeatureSeqList::F32(Vec::<Vec<f32>>::new())
            }
            Some(Feature_oneof_kind::int64_list(_)) => {
                FeatureSeqList::I64(Vec::<Vec<i64>>::new())
            }
            None => {
                continue;
            }
        };

        for feature in feat_vec {
            match feature.kind {
                Some(Feature_oneof_kind::bytes_list(mut val)) => {
                    if let FeatureSeqList::Bytes(ref mut vals) = values {
                        vals.push(val.take_value().into_vec());
                    }
                    else {
                        return Err(CorruptedRecordError.into());
                    }
                }
                Some(Feature_oneof_kind::float_list(mut val)) => {
                    if let FeatureSeqList::F32(ref mut vals) = values {
                        vals.push(val.take_value());
                    }
                    else {
                        return Err(CorruptedRecordError.into());
                    }
                }
                Some(Feature_oneof_kind::int64_list(mut val)) => {
                    if let FeatureSeqList::I64(ref mut vals) = values {
                        vals.push(val.take_value());
                    }
                    else {
                        return Err(CorruptedRecordError.into());
                    }
                }
                None => (),
            }
        }

        feature_result.insert(name, values);
    }

    Ok((context_result, feature_result))
}

pub fn parse_event(payload: &[u8]) -> Fallible<()> {
    let event: event::Event = protobuf::parse_from_bytes(payload)?;

    println!("###");
    println!("- wall_time: {:?}\tstep: {:?}", event.wall_time, event.step);

    match event.what {
        None => {
            println!("- none");
        },
        Some(Event_oneof_what::file_version(version)) => {
            println!("- file_version");
            println!("- version: {:?}", version);
        }
        Some(Event_oneof_what::graph_def(bytes)) => {
            println!("- graph_def");
        }
        Some(Event_oneof_what::summary(summary)) => {
            println!("- summary");

            for val in summary.value.iter() {
                println!("  - name: {:?}\ttag: {:?}", val.node_name, val.tag);
                match val.metadata.as_ref() {
                    Some(meta) => {
                        println!("    - meta_name: {:?}\tmeta_desc: {:?}", meta.display_name, meta.summary_description);
                        match meta.plugin_data.as_ref() {
                            Some(data) => {
                                println!("      - plugin_name: {:?}, data_len: {:?}", data.plugin_name, data.content.len());
                            }
                            None => {}
                        }
                    }
                    None => {},
                }
            }
        }
        Some(Event_oneof_what::log_message(log)) => {
            println!("- log_message");
            println!("- level: {:?}\tmessage: {:?}", log.level, log.message);
        }
        Some(Event_oneof_what::session_log(log)) => {
            println!("- session_log");
            println!("- status: {:?}\tcheckpoint_path: {:?}\tmsg: {:?}", log.status, log.checkpoint_path, log.msg);
        }
        Some(Event_oneof_what::tagged_run_metadata(meta)) => {
            println!("- tagged_run_metadata");
            println!("- tag: {:?}\trun_metadata: {:?}", meta.tag, meta.run_metadata);
        }
        Some(Event_oneof_what::meta_graph_def(bytes)) => {
            println!("- meta_graph_def");
        }
    }

    Ok(())
}
