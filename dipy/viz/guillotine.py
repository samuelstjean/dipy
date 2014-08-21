# TODO: Add support for LUT input
# TODO: Add key events management (opacity,XYZ axes, translation, rotation
# TODO: Adapt assertions to dipy style
# TODO: Support VTK6
# TODO: Find a place to put utils.py

import vtk
import utils


class Guillotine:

    def __init__(self):
        self.plane = vtk.vtkPlane()
        self.nb_data_volumes = 0
        self.blender = vtk.vtkImageBlend()
        self.renderer = vtk.vtkRenderer()

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

    def add_data_volume(self, data, opacity=0.5, affine=None):
        # Parameters:
        #     data: Numpy NdArray
        #     opacity: float
        #     affine: 4x4 Numpy NdArray
        #
        # Function: (required)
        #     Adds a data volume to the blending with its corresponding
        #     opacity. Every data volume added should have the same extent
        #     and the one with the most components added first. The affine
        #     transformation is applied if present.

        # Transform data into vtk tangible object
        image_data = utils.ndarray2vtkImageData(data)

        # Set the transform
        transform = vtk.vtkTransform()
        if affine is not None:
            transform_matrix = vtk.vtkMatrix4x4()
            transform_matrix.DeepCopy((
                affine[0][0], affine[0][1], affine[0][2], affine[0][3],
                affine[1][0], affine[1][1], affine[1][2], affine[1][3],
                affine[2][0], affine[2][1], affine[2][2], affine[2][3],
                affine[3][0], affine[3][1], affine[3][2], affine[3][3]))
            transform.SetMatrix(transform_matrix)
            transform.Inverse()

        # Set the reslicing
        image_reslice = vtk.vtkImageReslice()
        image_reslice.SetInput(image_data)
        image_reslice.SetResliceTransform(transform)
        image_reslice.AutoCropOutputOn()
        image_reslice.SetInterpolationModeToCubic()

        # Set the transformed image
        transformed_image = vtk.vtkImageData()
        transformed_image = image_reslice.GetOutput()
        transformed_image.Update()

        # Check if the extent is compatible with previous volumes
        if self.nb_data_volumes > 0:
            assert (transformed_image.GetExtent() ==
                    self.blender.GetOutput().GetWholeExtent()), \
                "image data extent doesn't fit the actual extent. (" + \
                str(transformed_image.GetExtent()) + \
                " vs actual " + \
                str(self.blender.GetOutput().GetWholeExtent()) + \
                ")"

        # Blend volumes
        self.blender.SetInput(self.nb_data_volumes, transformed_image)
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
        extent = data_volume.GetExtent()
        ori = data_volume.GetOrigin()
        bbox = (
            (extent[0] + ori[0]),
            (extent[1] + ori[0]),
            (extent[2] + ori[1]),
            (extent[3] + ori[1]),
            (extent[4] + ori[2]),
            (extent[5] + ori[2]))

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
        plane_widget.SetNormal(0, 0, 1)
        plane_widget.GetPlane(self.plane)

        # Set axes
        axes = vtk.vtkCubeAxesActor()
        axes.SetBounds(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5])
        axes.SetTickLocationToBoth()
        axes.SetXUnits("mm")
        axes.SetYUnits("mm")
        axes.SetZUnits("mm")
        axes.SetFlyModeToOuterEdges()
        axes.SetCamera(self.renderer.GetActiveCamera())
        self.add_actor(axes)

        # Set the renderer
        self.renderer.SetBackground(0.18, 0.18, 0.18)
        self.renderer.GetActiveCamera().SetPosition(0, 0, -1)
        self.renderer.ResetCamera()

        # Render scene
        self.renderer.Render()

        # Initialize interactor
        interactor.Initialize()

        # Render window
        render_window.Render()

        # Start interactor
        interactor.Start()
