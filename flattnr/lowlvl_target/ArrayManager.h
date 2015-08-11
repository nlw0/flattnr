/* Copyright 2013, 2014 Nicolau Leal Werneck
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

#ifndef ARRAYMANAGER_H
#define ARRAYMANAGER_H
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <stdexcept>

template <class T>
class ArrayManager : public TupleStreamExtractable {
  bool specific_dimensions;
  int ndim_min, ndim_max;

  void get_pyarray_with_type(PyObject* obj);

public:
  PyArrayObject *array_obj;

  // Pointer to the actual data array from array_obj.
  T* data;

  // Size of the array in each dimension. Simply points to the
  // "dimensions" from the array_obj. "size" is the name used in
  // Python.
  npy_intp* size;

  ArrayManager();

  // The main reason for this class to be created. Running PyDECREF on
  // the array reference at the object destructor.
  ~ArrayManager();

  // Square brackets simply gives access to the "data" pointer from
  // the array.
  inline T& operator[](npy_int);

  // Overload the parenthesis to allow for indexing with multiple
  // dimensions. Remember square brackets can never receive multiple
  // arguments, so this is why we used the parenthesis. In practice
  // you can sometimes save some multiplications by handling the index
  // calculation yourself, and using [], but these methods here can be
  // very handy, and can often be fast enough.  And final observation:
  // there are no bound checks whatsoever. This class is only supposed
  // to help you access the data from numpy arrays to perform fast
  // calculations. You need to know what you are doing.
  inline T& operator()(npy_int);
  inline T& operator()(npy_int, npy_int);
  inline T& operator()(npy_int, npy_int, npy_int);
  inline T& operator()(npy_int, npy_int, npy_int, npy_int);

  ArrayManager& operator=(PyObject* obj);

  void set_dimensions(int);
  void set_dimensions(int, int);
};
#endif /* ARRAYMANAGER_H */
