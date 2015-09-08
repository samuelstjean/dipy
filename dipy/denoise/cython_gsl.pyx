#    CythonGSL provides a set of Cython declarations for the GNU Scientific Library (GSL).
#    Copyright (C) 2012 Thomas V. Wiecki
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

def get_include():
    import sys, os

    if sys.platform == "win32":
        gsl_include = os.getenv('LIB_GSL')
        if gsl_include is None:
            # Environmental variable LIB_GSL not set, use hardcoded path.
            gsl_include = r"c:\Program Files\GnuWin32\include"
        else:
            gsl_include += "/include"
    else:
        gsl_include = os.popen('gsl-config --cflags').read()[2:-1]

    assert gsl_include != '', "Couldn't find gsl. Make sure it's installed and in the path."

    return gsl_include

def get_library_dir():
    import sys, os

    if sys.platform == "win32":
        lib_gsl_dir = os.getenv('LIB_GSL')
        if lib_gsl_dir is None:
            # Environmental variable LIB_GSL not set, use hardcoded path.
            lib_gsl_dir = r"c:\Program Files\GnuWin32\lib"
        else:
            lib_gsl_dir += "/lib"
    else:
        lib_gsl_dir = os.popen('gsl-config --libs').read().split()[0][2:]

    return lib_gsl_dir

def get_libraries():
    return ['gsl', 'gslcblas']

def get_cython_include_dir():
    import cython_gsl, os.path
    return os.path.split(cython_gsl.__path__[0])[0]


from libc.math cimport *
from libc.stdio cimport *

cdef enum:
  GSL_SUCCESS = 0
  GSL_FAILURE  = -1
  GSL_CONTINUE = -2  # iteration has not converged
  GSL_EDOM     = 1   # input domain error, e.g sqrt(-1)
  GSL_ERANGE   = 2   # output range error, e.g. exp(1e100)
  GSL_EFAULT   = 3   # invalid pointer
  GSL_EINVAL   = 4   # invalid argument supplied by user
  GSL_EFAILED  = 5   # generic failure
  GSL_EFACTOR  = 6   # factorization failed
  GSL_ESANITY  = 7   # sanity check failed - shouldn't happen
  GSL_ENOMEM   = 8   # malloc failed
  GSL_EBADFUNC = 9   # problem with user-supplied function
  GSL_ERUNAWAY = 10  # iterative process is out of control
  GSL_EMAXITER = 11  # exceeded max number of iterations
  GSL_EZERODIV = 12  # tried to divide by zero
  GSL_EBADTOL  = 13  # user specified an invalid tolerance
  GSL_ETOL     = 14  # failed to reach the specified tolerance
  GSL_EUNDRFLW = 15  # underflow
  GSL_EOVRFLW  = 16  # overflow
  GSL_ELOSS    = 17  # loss of accuracy
  GSL_EROUND   = 18  # failed because of roundoff error
  GSL_EBADLEN  = 19  # matrix, vector lengths are not conformant
  GSL_ENOTSQR  = 20  # matrix not square
  GSL_ESING    = 21  # apparent singularity detected
  GSL_EDIVERGE = 22  # integral or series is divergent
  GSL_EUNSUP   = 23  # requested feature is not supported by the hardware
  GSL_EUNIMPL  = 24  # requested feature not (yet) implemented
  GSL_ECACHE   = 25  # cache limit exceeded
  GSL_ETABLE   = 26  # table limit exceeded
  GSL_ENOPROG  = 27  # iteration is not making progress towards solution
  GSL_ENOPROGJ = 28  # jacobian evaluations are not improving the solution
  GSL_ETOLF    = 29  # cannot reach the specified tolerance in F
  GSL_ETOLX    = 30  # cannot reach the specified tolerance in X
  GSL_ETOLG    = 31  # cannot reach the specified tolerance in gradient
  GSL_EOF      = 32  # end of file

ctypedef int size_t

cdef extern from "gsl/gsl_sf_hyperg.h":

  double  gsl_sf_hyperg_0F1(double c, double x) nogil

  int  gsl_sf_hyperg_0F1_e(double c, double x, gsl_sf_result * result) nogil

  double  gsl_sf_hyperg_1F1_int(int m, int n, double x) nogil

  int  gsl_sf_hyperg_1F1_int_e(int m, int n, double x, gsl_sf_result * result) nogil

  double  gsl_sf_hyperg_1F1(double a, double b, double x) nogil

  int  gsl_sf_hyperg_1F1_e(double a, double b, double x, gsl_sf_result * result) nogil

  double  gsl_sf_hyperg_U_int(int m, int n, double x) nogil

  int  gsl_sf_hyperg_U_int_e(int m, int n, double x, gsl_sf_result * result) nogil

  int  gsl_sf_hyperg_U_int_e10_e(int m, int n, double x, gsl_sf_result_e10 * result) nogil

  double  gsl_sf_hyperg_U(double a, double b, double x) nogil

  int  gsl_sf_hyperg_U_e(double a, double b, double x) nogil

  int  gsl_sf_hyperg_U_e10_e(double a, double b, double x, gsl_sf_result_e10 * result) nogil

  double  gsl_sf_hyperg_2F1(double a, double b, double c, double x) nogil

  int  gsl_sf_hyperg_2F1_e(double a, double b, double c, double x, gsl_sf_result * result) nogil

  double  gsl_sf_hyperg_2F1_conj(double aR, double aI, double c, double x) nogil

  int  gsl_sf_hyperg_2F1_conj_e(double aR, double aI, double c, double x, gsl_sf_result * result) nogil

  double  gsl_sf_hyperg_2F1_renorm(double a, double b, double c, double x) nogil

  int  gsl_sf_hyperg_2F1_renorm_e(double a, double b, double c, double x, gsl_sf_result * result) nogil

  double  gsl_sf_hyperg_2F1_conj_renorm(double aR, double aI, double c, double x) nogil

  int  gsl_sf_hyperg_2F1_conj_renorm_e(double aR, double aI, double c, double x, gsl_sf_result * result) nogil

  double  gsl_sf_hyperg_2F0(double a, double b, double x) nogil

  int  gsl_sf_hyperg_2F0_e(double a, double b, double x, gsl_sf_result * result) nogil
