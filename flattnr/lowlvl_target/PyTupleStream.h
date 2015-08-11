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

#ifndef PYTUPLESTREAM_H
#define PYTUPLESTREAM_H
#include <Python.h>
#include <exception>
#include <string>
#include <stdexcept>

class TupleStreamExtractable {
public:
  virtual TupleStreamExtractable& operator=(PyObject*) = 0;
};

class TupleStream {
  int count;
  int Nargs;
  PyObject *args;
  bool failbit;

public:  
  friend TupleStream& operator>>(TupleStream &, long int &);
  friend TupleStream& operator>>(TupleStream &, double &);
  friend TupleStream& operator>>(TupleStream &, TupleStreamExtractable &);

  class ArgsCountException: public std::exception {};
  class TypeErrorException: public std::exception {};
  class FailStateException: public std::exception {};

  TupleStream(PyObject *input_args);

  bool fail();
  void set_fail_generic(std::string);
  void set_fail_nargs();
  void set_fail_typeerror();
  bool eof();
};

TupleStream& operator>>(TupleStream &, long int &);
TupleStream& operator>>(TupleStream &, double &);
TupleStream& operator>>(TupleStream &, TupleStreamExtractable &);

#endif /* PYTUPLESTREAM_H */
