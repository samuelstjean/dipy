#!/usr/bin/env python
# VTK5.8.0 doc : http://www.vtk.org/doc/release/5.8/html/index.html

# TODO
# Understand the space
# Add other actors
# Visualize colors, Blend images and
# Transparency possibilities ?
# Get plane, matrix, normal, origin...

import nibabel as nb
import numpy as np
import vtk
from dipy.viz import fvtk


def cut(data_source, other_data_source = None):
    # Define variables
    global plane
    plane = vtk.vtkPlane()
    blender = vtk.vtkImageBlend()
    cutter = vtk.vtkCutter()
    cutter_mapper = vtk.vtkPolyDataMapper()
    cutter_actor = vtk.vtkActor()
    interactor = vtk.vtkRenderWindowInteractor()
    plane_widget = vtk.vtkImplicitPlaneWidget()
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    initial_normal = [1, 0, 0]

    # Set the render window
    render_window.SetSize(800, 600)

    # Set the renderer
    renderer.SetBackground(0.18, 0.18, 0.18)

    # Add the renderer to the render window
    render_window.AddRenderer(renderer)

    # Add interactor to the render window
    interactor.SetRenderWindow(render_window)

    # Get the volume properties
    xmin, xmax, ymin, ymax, zmin, zmax = data_source.GetWholeExtent()
    origin = [(xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2]
    
    # Blend images together
    blender.SetInput(0, data_source.GetOutput())
    if other_data_source is not None:
        blender.SetInput(1, other_data_source.GetOutput())
        blender.SetOpacity(0, 0.5)
        blender.SetOpacity(1, 0.5)

    # Set the cutter filters
    cutter.SetInputConnection(blender.GetOutputPort())
    cutter.SetCutFunction(plane)

    # Set the cutter mappers
    cutter_mapper.SetInputConnection(cutter.GetOutputPort())

    # Set the cutter actors
    cutter_actor.SetMapper(cutter_mapper)

    # Add cutter actors to the renderer
    renderer.AddActor(cutter_actor)

    # Add space axes
    renderer.AddActor(fvtk.axes(((xmax-xmin)/4, (ymax-ymin)/4, (zmax-zmin)/4)))
    minimum_length = min((xmax-xmin), (ymax-ymin), (zmax-zmin))
    renderer.AddActor(fvtk.axes((
        minimum_length/4,
        minimum_length/8,
        minimum_length/8)))

    plane_widget.SetInput(data_source.GetOutput())
    plane_widget.SetInteractor(interactor)
    plane_widget.AddObserver("InteractionEvent", myCallback)

    plane_widget.SetEnabled(1)
    plane_widget.SetDrawPlane(0)
    plane_widget.SetTubing(0)
    plane_widget.OutlineTranslationOff()
    plane_widget.OutsideBoundsOff()
    plane_widget.SetPlaceFactor(1)
    plane_widget.PlaceWidget()
    plane_widget.SetOrigin(
        origin[0],
        origin[1],
        origin[2])
    plane_widget.SetNormal(
        initial_normal[0],
        initial_normal[1],
        initial_normal[2])
    plane_widget.GetPlane(plane)

    # Render the scene
    renderer.Render()

    # Initialize the interactor
    interactor.Initialize()

    # Render the window
    render_window.Render()

    # Start the interactor
    interactor.Start()


# The callback function
def myCallback(obj, event):
    global plane
    obj.GetPlane(plane)


def load_data(filename):
    return np.array(nb.load(filename).get_data())
    

def get_vtk_data(ndarray):
    nb_composites = len(ndarray.shape)

    if nb_composites == 3:
        [sx, sy, sz] = ndarray.shape
        nb_channels = 1
    elif nb_composites == 4:
        [sx, sy, sz, nb_channels] = ndarray.shape
    else:
        exit()
    
    # Convert data to uint8 properly
    uint8_data = rescale_to_uint8(ndarray)
    uint8_data = np.swapaxes(uint8_data, 0, 2)
    uint8_data = uint8_data[::-1, :, :]
#     print uint8_data.shape
    string_data = uint8_data.tostring()

    # Set data importer
    data_source = vtk.vtkImageImport()
    data_source.SetNumberOfScalarComponents(nb_channels)
    data_source.SetDataScalarTypeToUnsignedChar()
    data_source.SetWholeExtent(0, sx-1, 0, sy-1, 0, sz-1)
    data_source.SetDataExtentToWholeExtent()
    data_source.CopyImportVoidPointer(string_data, len(string_data))
#     data_source.SetDataOrigin(sx/2, sy/2, sz/2)
    data_source.Update()
    
    return data_source

def rescale_to_uint8(data):
    temp = np.array(data, dtype=np.float64)
    temp -= np.min(temp)
    if np.max(temp) != 0.0:
        temp /= np.max(temp)
        temp *= 255.0
    temp = np.array(np.round(temp), dtype=np.uint8)
    return temp


def main():
    # Get some data
    vtk_data = None
    other_vtk_data = None
    data = load_data("/home/algo/Documents/data/set2/b0.nii")
    other_data = load_data("/home/algo/Documents/data/set2/fa.nii")
    
    vtk_data = get_vtk_data(data)
    other_vtk_data = get_vtk_data(other_data)
    
#     image_plane_widget.cut(data_source)
    cut(vtk_data, other_vtk_data)
#     implicit_plane_image_slicing.cut(data_source)

if __name__ == "__main__":
    main()
