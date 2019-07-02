use std::convert::From;
use tch;
use ndarray::{
    self,
    ArrayD, Array1, Array2, Array3, Array4,
    ArrayViewD, ArrayView1, ArrayView2, ArrayView3, ArrayView4,
};

pub struct ToTorchTensor<T> {
    value: T,
}

macro_rules! impl_array_to_torch (
    ( $dtype:ty ) => (
        impl From<$dtype> for ToTorchTensor<$dtype> {
            fn from(value: $dtype) -> Self {
                ToTorchTensor {
                    value
                }
            }
        }

        impl<'a> From<&'a $dtype> for ToTorchTensor<&'a $dtype> {
            fn from(value: &'a $dtype) -> Self {
                ToTorchTensor {
                    value
                }
            }
        }

        impl From<ToTorchTensor<$dtype>> for tch::Tensor {
            fn from(input: ToTorchTensor<$dtype>) -> Self {
                let value = input.value;
                let dims = &value.shape()
                    .into_iter()
                    .map(|v| *v as i64)
                    .collect::<Vec<_>>();
                let tensor = tch::Tensor::of_slice(value.to_owned().into_raw_vec().as_slice())
                    .view(dims);
                tensor
            }
        }

        impl<'a> From<ToTorchTensor<&'a $dtype>> for tch::Tensor {
            fn from(input: ToTorchTensor<&'a $dtype>) -> Self {
                let value = input.value;
                let dims = &value.shape()
                    .into_iter()
                    .map(|v| *v as i64)
                    .collect::<Vec<_>>();
                let tensor = tch::Tensor::of_slice(value.to_owned().into_raw_vec().as_slice())
                    .view(dims);
                tensor
            }
        }
    )
);

macro_rules! impl_arrayview_to_torch (
    ( $dtype:ty ) => (
        impl<'a> From<$dtype> for ToTorchTensor<$dtype> {
            fn from(value: $dtype) -> Self {
                ToTorchTensor {
                    value
                }
            }
        }

        impl<'a> From<&'a $dtype> for ToTorchTensor<&'a $dtype> {
            fn from(value: &'a $dtype) -> Self {
                ToTorchTensor {
                    value
                }
            }
        }

        impl<'a> From<ToTorchTensor<$dtype>> for tch::Tensor {
            fn from(input: ToTorchTensor<$dtype>) -> Self {
                let value = input.value;
                let dims = &value.shape()
                    .into_iter()
                    .map(|v| *v as i64)
                    .collect::<Vec<_>>();
                let tensor = tch::Tensor::of_slice(value.to_owned().into_raw_vec().as_slice())
                    .view(dims);
                tensor
            }
        }

        impl<'a> From<ToTorchTensor<&'a $dtype>> for tch::Tensor {
            fn from(input: ToTorchTensor<&'a $dtype>) -> Self {
                let value = input.value;
                let dims = &value.shape()
                    .into_iter()
                    .map(|v| *v as i64)
                    .collect::<Vec<_>>();
                let tensor = tch::Tensor::of_slice(value.to_owned().into_raw_vec().as_slice())
                    .view(dims);
                tensor
            }
        }
    )
);

macro_rules! impl_array_vec_to_torch (
    ( $dtype:ty ) => (
        impl<'a> From<Vec<$dtype>> for ToTorchTensor<Vec<$dtype>> {
            fn from(value: Vec<$dtype>) -> Self {
                ToTorchTensor {
                    value
                }
            }
        }

        impl<'a> From<&'a Vec<$dtype>> for ToTorchTensor<&'a Vec<$dtype>> {
            fn from(value: &'a Vec<$dtype>) -> Self {
                ToTorchTensor {
                    value
                }
            }
        }

        impl<'a> From<ToTorchTensor<Vec<$dtype>>> for Vec<tch::Tensor> {
            fn from(input: ToTorchTensor<Vec<$dtype>>) -> Self {
                let value = input.value;
                let tensor_list = value.into_iter()
                    .map(|val| {
                        let dims = &val.shape()
                            .into_iter()
                            .map(|v| *v as i64)
                            .collect::<Vec<_>>();
                        let tensor = tch::Tensor::of_slice(&val.to_owned().into_raw_vec().as_slice())
                            .view(dims);
                        tensor
                    })
                    .collect::<Vec<_>>();
                tensor_list
            }
        }

        impl<'a> From<ToTorchTensor<&'a Vec<$dtype>>> for Vec<tch::Tensor> {
            fn from(input: ToTorchTensor<&'a Vec<$dtype>>) -> Self {
                let value = input.value;
                let tensor_list = value.into_iter()
                    .map(|val| {
                        let dims = &val.shape()
                            .into_iter()
                            .map(|v| *v as i64)
                            .collect::<Vec<_>>();
                        let tensor = tch::Tensor::of_slice(&val.to_owned().into_raw_vec().as_slice())
                            .view(dims);
                        tensor
                    })
                    .collect::<Vec<_>>();
                tensor_list
            }
        }
    )
);

macro_rules! impl_arrayview_vec_to_torch (
    ( $dtype:ty ) => (
        impl<'a> From<Vec<$dtype>> for ToTorchTensor<Vec<$dtype>> {
            fn from(value: Vec<$dtype>) -> Self {
                ToTorchTensor {
                    value
                }
            }
        }

        impl<'a> From<&'a Vec<$dtype>> for ToTorchTensor<&'a Vec<$dtype>> {
            fn from(value: &'a Vec<$dtype>) -> Self {
                ToTorchTensor {
                    value
                }
            }
        }

        impl<'a> From<ToTorchTensor<Vec<$dtype>>> for Vec<tch::Tensor> {
            fn from(input: ToTorchTensor<Vec<$dtype>>) -> Self {
                let value = input.value;
                let tensor_list = value.into_iter()
                    .map(|val| {
                        let dims = &val.shape()
                            .into_iter()
                            .map(|v| *v as i64)
                            .collect::<Vec<_>>();
                        let tensor = tch::Tensor::of_slice(&val.to_owned().into_raw_vec().as_slice())
                            .view(dims);
                        tensor
                    })
                    .collect::<Vec<_>>();
                tensor_list
            }
        }

        impl<'a> From<ToTorchTensor<&'a Vec<$dtype>>> for Vec<tch::Tensor> {
            fn from(input: ToTorchTensor<&'a Vec<$dtype>>) -> Self {
                let value = input.value;
                let tensor_list = value.into_iter()
                    .map(|val| {
                        let dims = &val.shape()
                            .into_iter()
                            .map(|v| *v as i64)
                            .collect::<Vec<_>>();
                        let tensor = tch::Tensor::of_slice(&val.to_owned().into_raw_vec().as_slice())
                            .view(dims);
                        tensor
                    })
                    .collect::<Vec<_>>();
                tensor_list
            }
        }
    )
);

impl_array_to_torch!(ArrayD<u8>);
impl_array_to_torch!(ArrayD<f32>);
impl_array_to_torch!(ArrayD<f64>);
impl_array_to_torch!(ArrayD<i32>);
impl_array_to_torch!(ArrayD<i64>);

impl_array_to_torch!(Array1<u8>);
impl_array_to_torch!(Array1<f32>);
impl_array_to_torch!(Array1<f64>);
impl_array_to_torch!(Array1<i32>);
impl_array_to_torch!(Array1<i64>);

impl_array_to_torch!(Array2<u8>);
impl_array_to_torch!(Array2<f32>);
impl_array_to_torch!(Array2<f64>);
impl_array_to_torch!(Array2<i32>);
impl_array_to_torch!(Array2<i64>);

impl_array_to_torch!(Array3<u8>);
impl_array_to_torch!(Array3<f32>);
impl_array_to_torch!(Array3<f64>);
impl_array_to_torch!(Array3<i32>);
impl_array_to_torch!(Array3<i64>);

impl_array_to_torch!(Array4<u8>);
impl_array_to_torch!(Array4<f32>);
impl_array_to_torch!(Array4<f64>);
impl_array_to_torch!(Array4<i32>);
impl_array_to_torch!(Array4<i64>);

impl_array_vec_to_torch!(ArrayD<u8>);
impl_array_vec_to_torch!(ArrayD<f32>);
impl_array_vec_to_torch!(ArrayD<f64>);
impl_array_vec_to_torch!(ArrayD<i32>);
impl_array_vec_to_torch!(ArrayD<i64>);

impl_array_vec_to_torch!(Array1<u8>);
impl_array_vec_to_torch!(Array1<f32>);
impl_array_vec_to_torch!(Array1<f64>);
impl_array_vec_to_torch!(Array1<i32>);
impl_array_vec_to_torch!(Array1<i64>);

impl_array_vec_to_torch!(Array2<u8>);
impl_array_vec_to_torch!(Array2<f32>);
impl_array_vec_to_torch!(Array2<f64>);
impl_array_vec_to_torch!(Array2<i32>);
impl_array_vec_to_torch!(Array2<i64>);

impl_array_vec_to_torch!(Array3<u8>);
impl_array_vec_to_torch!(Array3<f32>);
impl_array_vec_to_torch!(Array3<f64>);
impl_array_vec_to_torch!(Array3<i32>);
impl_array_vec_to_torch!(Array3<i64>);

impl_array_vec_to_torch!(Array4<u8>);
impl_array_vec_to_torch!(Array4<f32>);
impl_array_vec_to_torch!(Array4<f64>);
impl_array_vec_to_torch!(Array4<i32>);
impl_array_vec_to_torch!(Array4<i64>);

impl_arrayview_to_torch!(ArrayViewD<'a, u8>);
impl_arrayview_to_torch!(ArrayViewD<'a, f32>);
impl_arrayview_to_torch!(ArrayViewD<'a, f64>);
impl_arrayview_to_torch!(ArrayViewD<'a, i32>);
impl_arrayview_to_torch!(ArrayViewD<'a, i64>);

impl_arrayview_to_torch!(ArrayView1<'a, u8>);
impl_arrayview_to_torch!(ArrayView1<'a, f32>);
impl_arrayview_to_torch!(ArrayView1<'a, f64>);
impl_arrayview_to_torch!(ArrayView1<'a, i32>);
impl_arrayview_to_torch!(ArrayView1<'a, i64>);

impl_arrayview_to_torch!(ArrayView2<'a, u8>);
impl_arrayview_to_torch!(ArrayView2<'a, f32>);
impl_arrayview_to_torch!(ArrayView2<'a, f64>);
impl_arrayview_to_torch!(ArrayView2<'a, i32>);
impl_arrayview_to_torch!(ArrayView2<'a, i64>);

impl_arrayview_to_torch!(ArrayView3<'a, u8>);
impl_arrayview_to_torch!(ArrayView3<'a, f32>);
impl_arrayview_to_torch!(ArrayView3<'a, f64>);
impl_arrayview_to_torch!(ArrayView3<'a, i32>);
impl_arrayview_to_torch!(ArrayView3<'a, i64>);

impl_arrayview_to_torch!(ArrayView4<'a, u8>);
impl_arrayview_to_torch!(ArrayView4<'a, f32>);
impl_arrayview_to_torch!(ArrayView4<'a, f64>);
impl_arrayview_to_torch!(ArrayView4<'a, i32>);
impl_arrayview_to_torch!(ArrayView4<'a, i64>);

impl_arrayview_vec_to_torch!(ArrayViewD<'a, u8>);
impl_arrayview_vec_to_torch!(ArrayViewD<'a, f32>);
impl_arrayview_vec_to_torch!(ArrayViewD<'a, f64>);
impl_arrayview_vec_to_torch!(ArrayViewD<'a, i32>);
impl_arrayview_vec_to_torch!(ArrayViewD<'a, i64>);

impl_arrayview_vec_to_torch!(ArrayView1<'a, u8>);
impl_arrayview_vec_to_torch!(ArrayView1<'a, f32>);
impl_arrayview_vec_to_torch!(ArrayView1<'a, f64>);
impl_arrayview_vec_to_torch!(ArrayView1<'a, i32>);
impl_arrayview_vec_to_torch!(ArrayView1<'a, i64>);

impl_arrayview_vec_to_torch!(ArrayView2<'a, u8>);
impl_arrayview_vec_to_torch!(ArrayView2<'a, f32>);
impl_arrayview_vec_to_torch!(ArrayView2<'a, f64>);
impl_arrayview_vec_to_torch!(ArrayView2<'a, i32>);
impl_arrayview_vec_to_torch!(ArrayView2<'a, i64>);

impl_arrayview_vec_to_torch!(ArrayView3<'a, u8>);
impl_arrayview_vec_to_torch!(ArrayView3<'a, f32>);
impl_arrayview_vec_to_torch!(ArrayView3<'a, f64>);
impl_arrayview_vec_to_torch!(ArrayView3<'a, i32>);
impl_arrayview_vec_to_torch!(ArrayView3<'a, i64>);

impl_arrayview_vec_to_torch!(ArrayView4<'a, u8>);
impl_arrayview_vec_to_torch!(ArrayView4<'a, f32>);
impl_arrayview_vec_to_torch!(ArrayView4<'a, f64>);
impl_arrayview_vec_to_torch!(ArrayView4<'a, i32>);
impl_arrayview_vec_to_torch!(ArrayView4<'a, i64>);
