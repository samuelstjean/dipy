import vtk
import fvtk
from dipy.align.streamlinear import center_streamlines
from nibabel import trackvis as tv


def load_bundle(name):
    fname = dname + name + '.trk'
    streams, hdr = tv.read(fname)
    streamlines = [s[0] for s in streams]
    return streamlines

def load_big_data():

    fname = '/home/eleftherios/Data/fancy_data/2013_02_08_Gabriel_Girard/streamlines_500K.trk'
    streams, hdr = tv.read(fname)
    streamlines = [s[0] for s in streams]
    return streamlines


if __name__ == '__main__':


    dname = '/home/eleftherios/bundle_paper/data/faisceaux/'

    bundle_names = ['CC_front', 'CC_middle', 'CC_back', \
                    'cingulum_left', 'cingulum_right', \
                    'CST_left', 'CST_right', \
                    'IFO_left', 'IFO_right', \
                    'ILF_left', 'ILF_right',
                    'SCP_left', 'SCP_right', \
                    'SLF_left', 'SLF_right', \
                    'uncinate_left', 'uncinate_right']

    #streamlines = load_bundle('CC_middle')
    streamlines = load_big_data()

    streamlines, _ = center_streamlines(streamlines)

    stream_actor, stream_poly_data = fvtk.line(streamlines,
                                        fvtk.line_colors(streamlines),
                                        mapper=None)

    # cube = vtk.vtkCubeSource()
    # cube.SetBounds(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0)
    # cube.Update()

    # cubeMapper = vtk.vtkPolyDataMapper()
    # cubeMapper.SetInput(cube.GetOutput())
    # cubeActor = vtk.vtkActor()
    # cubeActor.GetProperty().SetColor(1.0, 0.3, 0.3)
    # cubeActor.SetMapper(cubeMapper)

    # # Create 3D cells so vtkImplicitDataSet evaluates inside vs outside correctly
    # tri = vtk.vtkDelaunay3D()
    # tri.SetInput(cube.GetOutput())
    # tri.Update()
    # #tri.BoundingTriangulationOff()

    # # vtkImplicitDataSet needs some scalars to interpolate to find inside/outside
    # elev = vtk.vtkElevationFilter()
    # elev.SetInputConnection(tri.GetOutputPort())
    # # elev.SetScalarRange(90, 110)
    # elev.Update()

    # implicit = vtk.vtkImplicitDataSet()
    # implicit.SetDataSet(elev.GetOutput())
    #implicit.Update()

    # implicit = vtk.vtkBox()
    implicit = vtk.vtkSphere()
    implicit.SetCenter(0, 0, 0)
    implicit.SetRadius(10.)


    diameter = 100*np.sqrt(2)/2.


    # implicit.SetBounds(-diameter, diameter, -diameter, diameter,
    #                    -diameter, diameter)

    cube = vtk.vtkCubeSource()
    cube.SetBounds(-diameter, diameter, -diameter, diameter,
                   -diameter, diameter)
    cube.Update()

    # tri = vtk.vtkDelaunay3D()
    # tri.SetInput(cube.GetOutput())
    # tri.BoundingTriangulationOff()

    # # vtkImplicitDataSet needs some scalars to interpolate to find inside/outside
    # elev = vtk.vtkElevationFilter()
    # elev.SetInputConnection(tri.GetOutputPort())

    # implicit = vtk.vtkImplicitDataSet()
    # implicit.SetDataSet(elev.GetOutput())

    extract_geometry = vtk.vtkExtractPolyDataGeometry()
    extract_geometry.SetInput(stream_poly_data)
    extract_geometry.SetImplicitFunction(implicit)
    extract_geometry.ExtractBoundaryCellsOn()
    extract_geometry.ExtractInsideOn()
    extract_geometry.Update()

    extracted_mapper = vtk.vtkPolyDataMapper()
    extracted_mapper.SetInputConnection(extract_geometry.GetOutputPort())
    extracted_mapper.GlobalImmediateModeRenderingOn()

    # extracted_actor = vtk.vtkActor()
    # extracted_actor.SetMapper(extracted_mapper)

    extracted_actor = vtk.vtkLODActor()
    extracted_actor.SetMapper(extracted_mapper)
    extracted_actor.SetNumberOfCloudPoints(100000)

    ren = vtk.vtkRenderer()
    #ren.AddActor(stream_actor)
    ren.AddActor(extracted_actor)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    transform = vtk.vtkTransform()

    def filter_streamlines(obj, event):
        global implicit, transform

        obj.GetTransform(transform)
        implicit.SetCenter(transform.GetPosition())
        radius = 5 * max(transform.GetScale())
        implicit.SetRadius(radius)


    #ren.AddActor(fvtk.axes(scale=(100, 100, 100)))

    stream_actor.GetProperty().SetOpacity(.2)
    #ren.AddActor(stream_actor)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    box_widget = vtk.vtkBoxWidget()
    box_widget.SetInteractor(iren)
    box_widget.SetInput(cube.GetOutput())

    box_widget.HandlesOn()


    box_widget.PlaceWidget()
    box_widget.AddObserver('InteractionEvent', filter_streamlines)
    box_widget.On()


    iren.Start()





