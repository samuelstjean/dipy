# TODO: World coordinate system
# TODO: Add support for LUT input
# TODO: Add key events management (opacity,XYZ axes, translation, rotation
# TODO: Adapt assertions to dipy style
# TODO: Support VTK6
# TODO: Put utils in fvtk_utils

import vtk
import utils


class Guillotine:

    def __init__(self):
        self.plane = vtk.vtkPlane()
        self.nb_data_volumes = 0
        self.blender = vtk.vtkImageBlend()
        self.renderer = vtk.vtkRenderer()
        
        # Set the renderer
        self.renderer.SetBackground(0.0, 0.0, 0.0)

    def _myCallback(self, obj, event):
        # Parameters:
        #     obj: VTK object that calls this function
        #     event: VTK event that occurred to call this function
        #
        # Function: (Internal)
        #     Updates the plane that cuts the data volume (see vtkCommand).

        obj.GetPlane(self.plane)

    def add_actor(self, actor):
        # Parameters:
        #     actor: vtkActor
        #
        # Function: (Optional)
        #     Adds an actor to the rendering.

        self.renderer.AddActor(actor)

    def add_data_volume(self, data, opacity=0.5):
        # Parameters:
        #     data: Numpy NdArray
        #     opacity: float
        #
        # Function: (required)
        #     Adds a data volume to the blending with its
        #     corresponding opacity. Every data volume added should have the
        #     same extent and the one with the most components added first.

        # Transform data into vtk tangible object
        data_volume = utils.ndarray2vtkImageData(data)

        if self.nb_data_volumes > 0:
            assert (data_volume.GetExtent() ==
                    self.blender.GetOutput().GetWholeExtent()), \
                "data_volume extent doesn't fit the actual extent. (" + \
                str(data_volume.GetExtent()) + \
                " vs actual " + \
                str(self.blender.GetOutput().GetWholeExtent()) + \
                ")"
        self.blender.SetInput(self.nb_data_volumes, data_volume)
        self.blender.SetOpacity(self.nb_data_volumes, opacity)
        self.blender.UpdateWholeExtent()
        self.nb_data_volumes += 1

    def cut(self):
        # Function: (required)
        #     Show the cutting through an interactive widget to visualize data
        #     volumes.

        # Get blending result
        assert (self.nb_data_volumes > 0), \
            "No data volume, use function add_data_volume."
        data_volume = self.blender.GetOutput()

        # Get volume properties
        xdim, ydim, zdim = data_volume.GetDimensions()
        xmin, xmax, ymin, ymax, zmin, zmax = data_volume.GetExtent()
        origin = [(xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2]
        initial_normal = [1, 0, 0]

        # Set cutter filter
        cutter = vtk.vtkCutter()
        cutter.SetInput(data_volume)
        cutter.SetCutFunction(self.plane)

        # Set the cutter mappers
        cutter_mapper = vtk.vtkPolyDataMapper()
        cutter_mapper.SetInputConnection(cutter.GetOutputPort())

        # Set cutter actor
        cutter_actor = vtk.vtkActor()
        cutter_actor.SetMapper(cutter_mapper)
        self.add_actor(cutter_actor)
        
        # Set render window
        render_window = vtk.vtkRenderWindow()
        render_window.SetSize(800, 600)
        render_window.AddRenderer(self.renderer)
        
        # Set interactor
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        # Set plane widget
        plane_widget = vtk.vtkImplicitPlaneWidget()
        plane_widget.SetInput(data_volume)
        plane_widget.SetInteractor(interactor)
        plane_widget.AddObserver("InteractionEvent", self._myCallback)
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
        plane_widget.GetPlane(self.plane)

        # Render scene
        self.renderer.Render()

        # Initialize interactor
        interactor.Initialize()
        
        # Render window
        render_window.Render()
        
        # Start interactor
        interactor.Start()
