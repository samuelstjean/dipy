# TODO: Show some hair on this cut head
# TODO: World coordinate system
# TODO: Support VTK6
# TODO: Add support for LUT input
# TODO: Add key events management (opacity,XYZ axes, translation, rotation
# TODO: Put utils in fvtk_utils
# TODO: Adapt assertions to dipy style

import vtk
import utils


class Guillotine:

    def __init__(self):
        self.plane = vtk.vtkPlane()
        self.nb_data_volumes = 0
        self.blender = vtk.vtkImageBlend()
        self.cutter = vtk.vtkCutter()
        self.cutter_mapper = vtk.vtkPolyDataMapper()
        self.cutter_actor = vtk.vtkActor()
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.plane_widget = vtk.vtkImplicitPlaneWidget()
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()

        # Set the render window
        self.render_window.SetSize(800, 600)

        # Set the renderer
        self.renderer.SetBackground(0.0, 0.0, 0.0)

        # Add the renderer to the render window
        self.render_window.AddRenderer(self.renderer)

        # Add interactor to the render window
        self.interactor.SetRenderWindow(self.render_window)

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

        # Get the blending result
        assert (self.nb_data_volumes > 0), \
            "No data volume, use function add_data_volume."
        data_volume = self.blender.GetOutput()

        # Get the volume properties
        xdim, ydim, zdim = data_volume.GetDimensions()
        xmin, xmax, ymin, ymax, zmin, zmax = data_volume.GetExtent()
        origin = [(xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2]
        initial_normal = [1, 0, 0]

        # Set the cutter filters
        self.cutter.SetInput(data_volume)
        self.cutter.SetCutFunction(self.plane)

        # Set the cutter mappers
        self.cutter_mapper.SetInputConnection(self.cutter.GetOutputPort())

        # Set the cutter actors
        self.cutter_actor.SetMapper(self.cutter_mapper)

        # Add cutter actors to the renderer
        self.renderer.AddActor(self.cutter_actor)

        self.plane_widget.SetInput(data_volume)
        self.plane_widget.SetInteractor(self.interactor)
        self.plane_widget.AddObserver("InteractionEvent", self._myCallback)

        self.plane_widget.SetEnabled(1)
        self.plane_widget.SetDrawPlane(0)
        self.plane_widget.SetTubing(0)
        self.plane_widget.OutlineTranslationOff()
        self.plane_widget.OutsideBoundsOff()
        self.plane_widget.SetPlaceFactor(1)
        self.plane_widget.PlaceWidget()
        self.plane_widget.SetOrigin(
            origin[0],
            origin[1],
            origin[2])
        self.plane_widget.SetNormal(
            initial_normal[0],
            initial_normal[1],
            initial_normal[2])
        self.plane_widget.GetPlane(self.plane)

        # Render the scene
        self.renderer.Render()

        # Initialize the interactor
        self.interactor.Initialize()

        # Render the window
        self.render_window.Render()

        # Start the interactor
        self.interactor.Start()
