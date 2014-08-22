#TODO
#    Command line
#    Tutorial
#    Support VTK6
#    Find a place to put utils.py
#    Add support for LUT input

import vtk
import utils


class Guillotine:
    def __init__(self):
        """Data volume slicing utility
            Once instantiated, several volumes can be added to this class to be
            mixed and cut together in an interactive window. Also, any actors
            like streamlines or glyphs could be added so different data could
            be visualized at once. The main feature of this class is the
            ability to explore volume data through interactive slicing.

        Methods
        -------
            add_actor : Adds actor to the renderer (optional)
            add_data_volume : Adds volume data in the blending
            show : Shows the cut volumes and other actors in a window
        """

        self.plane = vtk.vtkPlane()
        self.nb_data_volumes = 0
        self.blender = vtk.vtkImageBlend()
        self.renderer = vtk.vtkRenderer()
        self.plane_widget = vtk.vtkImplicitPlaneWidget()
        self.axes = vtk.vtkCubeAxesActor()

    def _key_press(self, object, event):
        """Executes action of a key
            This function is called every time a key is hit to change the
            camera angle, the cut slice and some display features.

        Parameters
        ----------
            object : vtkObject
                Object that calls this function
            event : vtkEvent
                Event that triggered the calling of this function

        Keys
        ----
            x : Clips plane normal to x axis
            y : Clips plane normal to y axis
            z : Clips plane normal to z axis
            r : Resets the camera to initial standard position (look z axis)
            Delete : Toggles display of the axes
            Right : Turns the volume to the right
            Left : Turns the volume to the left
            Up : Turns the volume up
            Down : Turns the volume down
            Prior(Page down) : Decreases the slice index along the normal
            Next (Page Up) : Increases the slice index along the normal
            Shift-Right : Rolls the camera right
            Shift-Left : Rolls the camera left
            Shift-Up : Zooms in the camera's view angle
            Shift-Down : Zooms out the camera's view angle
            Shift-Prior : (Page down) Fast prior key
            Shift-Next : (Page Up) Fast next key
        """

        key = object.GetKeySym()
        shift_pressed = object.GetShiftKey()

        if key == "x":
            self.plane_widget.SetNormal(1, 0, 0)
        elif key == "y":
            self.plane_widget.SetNormal(0, 1, 0)
        elif key == "z":
            self.plane_widget.SetNormal(0, 0, 1)
        elif key == "r":
            self.renderer.GetActiveCamera().SetPosition(0, 0, -1)
            self.renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
            self.renderer.GetActiveCamera().SetViewUp(0, 1, 0)
            self.renderer.GetActiveCamera().SetViewAngle(30)
            self.renderer.ResetCamera()
        elif key == "Delete":
            self.axes.SetXAxisVisibility(not self.axes.GetXAxisVisibility())
            self.axes.SetYAxisVisibility(not self.axes.GetYAxisVisibility())
            self.axes.SetZAxisVisibility(not self.axes.GetZAxisVisibility())
        elif key == "Right":
            if shift_pressed:
                self.renderer.GetActiveCamera().Roll(15)
            else:
                self.renderer.GetActiveCamera().Azimuth(15)
        elif key == "Left":
            if shift_pressed:
                self.renderer.GetActiveCamera().Roll(-15)
            else:
                self.renderer.GetActiveCamera().Azimuth(-15)
        elif key == "Down":
            if shift_pressed:
                self.renderer.GetActiveCamera().Zoom(0.9)
            else:
                self.renderer.GetActiveCamera().Elevation(-15)
        elif key == "Up":
            if shift_pressed:
                self.renderer.GetActiveCamera().Zoom(1.1)
            else:
                self.renderer.GetActiveCamera().Elevation(15)
        elif key == "Prior":
            factor = 1.0
            if shift_pressed:
                factor = 5.0
            origin = self.plane_widget.GetOrigin()
            normal = self.plane_widget.GetNormal()
            self.plane_widget.SetOrigin((origin[0] + factor*normal[0],
                                         origin[1] + factor*normal[1],
                                         origin[2] + factor*normal[2]))
        elif key == "Next":
            factor = 1.0
            if shift_pressed:
                factor = 5.0
            origin = self.plane_widget.GetOrigin()
            normal = self.plane_widget.GetNormal()
            self.plane_widget.SetOrigin((origin[0] - factor*normal[0],
                                         origin[1] - factor*normal[1],
                                         origin[2] - factor*normal[2]))

        self.plane_widget.InvokeEvent("InteractionEvent")

    def _update_plane(self, object, event):
        """Updates the cutting plane
            When this function is called, the plane is updated according to
            its normal and origin settings.

        Parameters
        ----------
            object : vtkObject
                Object that calls this function
            event : vtkEvent
                Event that triggered the calling of this function
        """

        object.GetPlane(self.plane)
        object.GetInteractor().Render()

    def add_actor(self, actor):
        """Adds actor to the renderer (optional)
            This function is useful to add any extra actor needed in the
            display.

        Parameters
        ----------
            actor : vtkActor
                Actor to add in the display
        """

        self.renderer.AddActor(actor)

    def add_data_volume(self, data, opacity=0.5, affine=None):
        """ Adds volume data in the blending
            This function takes a given volume, transforms it according to its
            optional affine matrix and blends it with previous volume, if
            present, and with optional opacity.

        Parameters
        ----------
            data : 3D ndarray
            opacity : float
            affine : 4x4 ndarray

        Note
        ----
        Every data volume added should have the same size and the one with
        the most components added first (rgb before grayscale).
        """

        image_data = utils.ndarray2vtkImageData(data)

        # Set the transform (identity if none given)
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
        image_reslice.SetInterpolationModeToLinear()
        image_reslice.Update()

        # Check if the extent is compatible with previous volumes
        if self.nb_data_volumes > 0:
            assert(
                image_reslice.GetOutput().GetExtent() ==
                self.blender.GetOutput().GetWholeExtent()), \
                "image data extent doesn't fit the actual extent. (" + \
                str(image_reslice.GetOutput().GetExtent()) + \
                " vs actual " + \
                str(self.blender.GetOutput().GetWholeExtent()) + \
                ")"

        # Blend volumes
        self.blender.SetInput(self.nb_data_volumes, image_reslice.GetOutput())
        self.blender.SetOpacity(self.nb_data_volumes, opacity)
        self.blender.UpdateWholeExtent()
        self.nb_data_volumes += 1

    def show(self):
        """Shows the cut volumes and other actors in a window
            Shows the cut and other optionaly added actors through an
            interactive widget to visualize data volumes. add_dat_volume
            function has to be called at least once before calling this one.

        Keys
        ----
            i : Toggles display of the interactor
            q or e : exits
        """

        # Get blending result
        assert(
            self.nb_data_volumes > 0), \
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
        interactor.AddObserver("KeyPressEvent", self._key_press)

        # Set plane widget
        self.plane_widget.SetInput(data_volume)
        self.plane_widget.SetInteractor(interactor)
        self.plane_widget.AddObserver("InteractionEvent", self._update_plane)
        self.plane_widget.SetDrawPlane(0)
        self.plane_widget.SetScaleEnabled(0)
        self.plane_widget.SetTubing(0)
        self.plane_widget.OutlineTranslationOff()
        self.plane_widget.OutsideBoundsOff()
        self.plane_widget.SetPlaceFactor(1)
        self.plane_widget.PlaceWidget()
        self.plane_widget.SetNormal(0, 0, 1)

        # Set axes
        self.axes.SetBounds(bbox[0],
                            bbox[1],
                            bbox[2],
                            bbox[3],
                            bbox[4],
                            bbox[5])
        self.axes.SetTickLocationToBoth()
        self.axes.SetFlyModeToOuterEdges()
        self.axes.SetCamera(self.renderer.GetActiveCamera())
        self.add_actor(self.axes)

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
