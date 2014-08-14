
from __future__ import division, print_function, absolute_import

import numpy as np

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

#import vtk
# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
ns, have_numpy_support, _ = optional_package('vtk.util.numpy_support')


def numpy_to_vtk_points(points):
    """ numpy points array to a vtk points array
    """
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(np.asarray(points), deep=True))
    return vtk_points


def numpy_to_vtk_colors(colors):
    """ numpy color array to a vtk color array
    
    if colors are not already in UNSIGNED_CHAR
        you may need to multiply by 255. 
        
    Example
    ----------
    >>>  vtk_colors = numpy_to_vtk_colors(255 * float_array)
    """
    vtk_colors = ns.numpy_to_vtk(np.asarray(colors), deep=True, 
                                 array_type=vtk.VTK_UNSIGNED_CHAR)
    return vtk_colors


def set_input(vtk_object, input):
    """ Generic input for vtk data, 
        depending of the type of input and vtk version

    Example
    ----------
    >>> poly_mapper = set_input(vtk.vtkPolyDataMapper(), poly_data)
    """
    if isinstance(input,vtk.vtkPolyData):
        if vtk.VTK_MAJOR_VERSION <= 5:
            vtk_object.SetInput(input)
        else:
            vtk_object.SetInputData(input)
    elif isinstance(input,vtk.vtkAlgorithmOutput):
        vtk_object.SetInputConnection(input)

    vtk_object.Update()
    return vtk_object


def trilinear_interp(input_array, indices):
    """ Evaluate the input_array data at the given indices
    """
    
    assert (input_array.ndim > 2 )," array need to be at least 3dimensions"
    assert (input_array.ndim < 5 )," dont support array with more than 4 dims"

    x_indices = indices[:,0]
    y_indices = indices[:,1]
    z_indices = indices[:,2]

    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    #Check if xyz1 is beyond array boundary:
    x1[np.where(x1==input_array.shape[0])] = x0.max()
    y1[np.where(y1==input_array.shape[1])] = y0.max()
    z1[np.where(z1==input_array.shape[2])] = z0.max()

    if input_array.ndim == 3:
        x = x_indices - x0
        y = y_indices - y0
        z = z_indices - z0
    elif input_array.ndim == 4:
        x = np.expand_dims(x_indices - x0, axis = 1)
        y = np.expand_dims(y_indices - y0, axis = 1)
        z = np.expand_dims(z_indices - z0, axis = 1)
        
    output = (input_array[x0,y0,z0]*(1-x)*(1-y)*(1-z) +
                 input_array[x1,y0,z0]*x*(1-y)*(1-z) +
                 input_array[x0,y1,z0]*(1-x)*y*(1-z) +
                 input_array[x0,y0,z1]*(1-x)*(1-y)*z +
                 input_array[x1,y0,z1]*x*(1-y)*z +
                 input_array[x0,y1,z1]*(1-x)*y*z +
                 input_array[x1,y1,z0]*x*y*(1-z) +
                 input_array[x1,y1,z1]*x*y*z)
    
    return output
