/* Copyright 2013 Nicolau Leal Werneck
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

#include "ArrayManager.h"

#define ERR_MSG_LEN 1024

template <class T>
ArrayManager<T>::ArrayManager(): specific_dimensions(false), ndim_min(-1), ndim_max(-1),
                              array_obj(NULL), data(NULL), size(NULL)
{ }

template <class T>
ArrayManager<T>::~ArrayManager() {
  if (array_obj)
    Py_DECREF(array_obj);
}

template <class T>
inline T& ArrayManager<T>::operator[](npy_int j) { return data[j]; }


template <class T>
inline T& ArrayManager<T>::operator()(npy_int x1) { return data[x1]; }


template <class T>
inline T& ArrayManager<T>::operator()(npy_int x2, npy_int x1) { return data[x2 * size[1] + x1]; }


template <class T>
inline T& ArrayManager<T>::operator()(npy_int x3, npy_int x2, npy_int x1) {
  return data[((x3 * size[2]) + x2) * size[1] + x1]; }


template <class T>
inline T& ArrayManager<T>::operator()(npy_int x4, npy_int x3, npy_int x2, npy_int x1) {
  return data[(((x4 * size[3] + x3) * size[2]) + x2) * size[1] + x1]; }


template <class T>
ArrayManager<T>& ArrayManager<T>::operator=(PyObject* obj) {

  // array_obj = (PyArrayObject*) PyArray_FROM_OTF(obj, NPY_INT32, NPY_ARRAY_INOUT_ARRAY);
  get_pyarray_with_type(obj);

  // Check number of dimensions if necessary.
  if (specific_dimensions) {
    if (PyArray_NDIM(array_obj) < ndim_min) {
      char err_msg[ERR_MSG_LEN];
      snprintf(err_msg, ERR_MSG_LEN, "Incorrect number of dimensions (%d < %d)",
               PyArray_NDIM(array_obj), ndim_min);
      throw std::runtime_error(err_msg);
    }
    else if (PyArray_NDIM(array_obj) > ndim_max) {
      char err_msg[ERR_MSG_LEN];
      snprintf(err_msg, ERR_MSG_LEN, "Incorrect number of dimensions (%d > %d)",
             PyArray_NDIM(array_obj), ndim_max);
      throw std::runtime_error(err_msg);
    }
  }

  size = PyArray_DIMS(array_obj);
  // TODO allow a check for specific sizes too.

  data = (T*) PyArray_DATA(array_obj);

  return *this;
}


template <class T>
void ArrayManager<T>::get_pyarray_with_type(PyObject* obj) {
  array_obj = (PyArrayObject*) PyArray_FROM_OTF(obj, NPY_NOTYPE, NPY_ARRAY_INOUT_ARRAY);
  if (array_obj == NULL) throw std::exception();
}


template <>
void ArrayManager<double>::get_pyarray_with_type(PyObject* obj) {
  array_obj = (PyArrayObject*) PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
  if (array_obj == NULL) throw std::exception();
}


template <>
void ArrayManager<float>::get_pyarray_with_type(PyObject* obj) {
  array_obj = (PyArrayObject*) PyArray_FROM_OTF(obj, NPY_FLOAT, NPY_ARRAY_INOUT_ARRAY);
  if (array_obj == NULL) throw std::exception();
}


template <>
void ArrayManager<int>::get_pyarray_with_type(PyObject* obj) {
  array_obj = (PyArrayObject*) PyArray_FROM_OTF(obj, NPY_INT, NPY_ARRAY_INOUT_ARRAY);
  if (array_obj == NULL) throw std::exception();
}


template <class T>
void ArrayManager<T>::set_dimensions(int input_ndim) {
  // This method must be called on a non-initialized object.
  if (data != NULL) throw std::exception();

  specific_dimensions = true;
  ndim_min = input_ndim;
  ndim_max = input_ndim;
}


template <class T>
void ArrayManager<T>::set_dimensions(int input_ndim_min, int input_ndim_max) {
  // This method must be called on a non-initialized object.
  if (data != NULL) throw std::exception();
  if (input_ndim_min > input_ndim_max) throw std::exception();

  specific_dimensions = true;
  ndim_min = input_ndim_min;
  ndim_max = input_ndim_max;
}
