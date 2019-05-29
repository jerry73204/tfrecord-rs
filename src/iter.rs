use std::io;
use std::fmt::Debug;
use std::thread::{self, JoinHandle};
use std::marker::PhantomData;
use std::collections::vec_deque::VecDeque;
use rand::prelude::*;
use crossbeam::channel::Receiver;
use crate::loader;
use crate::error::make_load_index_error;

// Trait defiintions

pub trait DsIterator: Iterator + Sized {
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
        Self: 'static + Send,
        Self::Item: 'static + Send, {

        let (sender, receiver) = crossbeam::channel::bounded(buf_size);

        let worker = thread::spawn(move || {
            debug!("Prefetch producer started");
            loop {
                debug!("{} elements in prefetch buffer reported by sender", sender.len());
                match self.next() {
                    None => {
                        if let Err(err) = sender.send(None) {
                            warn!("Prefetch producer error: {}", err);
                        }
                        debug!("Prefetch producer finished");
                        return;
                    }
                    Some(val) => {
                        if let Err(err) = sender.send(Some(val)) {
                            warn!("Prefetch producer error: {}", err);
                            return;
                        }
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
    E: Debug + Send, {

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
    I: Iterator + Send,
    I::Item: Send {

    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if let None = self.worker_opt {
            return None;
        }

        debug!("{} elements in prefetch buffer reported by consumer", self.receiver.len());
        match self.receiver.recv().unwrap() {
            None => {
                debug!("Prefetch consumer finished");
                self.worker_opt.take().unwrap().join().unwrap();
                None
            }
            Some(val) => Some(val),
        }
    }
}
