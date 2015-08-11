#include <Python.h>

#include "PyTupleStream.cpp"
#include "ArrayManager.cpp"

static PyObject * run_target(PyObject *self, PyObject *args) {
  TupleStream qq(args);

  long int v;

  ArrayManager<double> m;
  m.set_dimensions(2);

  qq >> v >> m;

  return Py_BuildValue("id", v + 310, m(0,0)+m(1,1));
}

static PyMethodDef LowlvlTargetMethods[] =
  {
    {"run_target", run_target, METH_VARARGS,
     "Run the damn thing.\n"},
    {NULL, NULL, 0, NULL}
  };

PyMODINIT_FUNC

init_core(void) {
  (void) Py_InitModule("_core", LowlvlTargetMethods);
  import_array();
  srand((int) time(NULL));
}
