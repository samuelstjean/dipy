
"""
1. Import libraries
"""

import numpy as np
import nibabel as nib
import vtk
import sys

# from glob import glob

from dipy.viz import fvtk
import dipy.viz.fvtk_actors as vtk_a
import dipy.viz.fvtk_util as vtk_u

"""
2. Read/write trackvis streamline files with nibabel.
"""

# dname = '/home/local/USHERBROOKE/stjs2902/Dropbox/share_sam_maxime/3T_mic_1.2mm/'
# method = 'aonlm/'

# stuff = \
#     ['bundles/bundles_det_cst.left.trk',
#      'bundles/bundles_det_or.left.trk',
#      'bundles/bundles_det_cst.right.trk',
#      'bundles/bundles_det_ifof.right.trk',
#      'bundles/bundles_det_cc_3.trk',
#      'bundles/bundles_det_cc_mohawk_endpoints.trk',
#      'bundles/bundles_det_or.right.trk',
#      'bundles/bundles_det_cst_with_cerebellum.right.trk',
#      'bundles/bundles_det_ilf.right.trk',
#      'bundles/bundles_det_cg.right.trk',
#      'bundles/bundles_det_cc_6.trk',
#      'bundles/bundles_det_comm_tracts.trk',
#      'bundles/bundles_det_cg.left.trk',
#      'bundles/bundles_det_cc_7.trk',
#      'bundles/bundles_det_uf.right.trk',
#      'bundles/bundles_det_ifof.left.trk',
#      'bundles/bundles_det_slf3.left.trk',
#      'bundles/bundles_det_cc_full.trk',
#      'bundles/bundles_det_cc_2.trk',
#      'bundles/bundles_det_slf3.right.trk',
#      'bundles/bundles_det_mdlf.right.trk',
#      'bundles/bundles_det_af.right.trk',
#      'bundles/bundles_det_cc_4.trk',
#      'bundles/bundles_det_slf2.left.trk',
#      'bundles/bundles_det_ilf.left.trk',
#      'bundles/bundles_det_cst_with_cerebellum.left.trk',
#      'bundles/bundles_det_uf.left.trk',
#      'bundles/bundles_det_cc_mohawk.trk',
#      'bundles/bundles_det_mdlf.left.trk',
#      'bundles/bundles_det_slf2.right.trk',
#      'bundles/bundles_det_af.left.trk',
#      'bundles/bundles_det_cc_5.trk',
#      'bundles/bundles_det_slf1.right.trk']


# for bundle in stuff:
    # bundle = "bundles_det_ilf.left"

bundle = sys.argv[1]
print(bundle)
streamlines_file = bundle # + '.trk'
streams, hdr = nib.trackvis.read(streamlines_file, points_space='rasmm')
lines = [s[0] for s in streams]

"""
3. Generate and render fvtk streamline with line (without width)
"""
fvtk_lines = vtk_a.line(lines)

renderer = fvtk.ren()
fvtk.add(renderer, fvtk_lines)
# fvtk.show(renderer)

rotation = vtk_u.rotation_from_lines(lines, use_line_dir=True, use_full_eig=False)
cam = renderer.GetActiveCamera()
cam.ApplyTransform(rotation)
# cam.Roll(-90)
# fvtk.show(renderer)

fvtk.record(renderer, out_path=(bundle + '.png').replace('.trk', ''), size=(800, 800))
fvtk.clear(renderer)


def record(ren, cam_pos=None, cam_focal=None, cam_view=None,
           out_path=None, path_numbering=False, n_frames=1, az_ang=10,
           magnification=1, size=(300, 300), verbose=False):
    ''' This will record a video of your scene
    Records a video as a series of ``.png`` files of your scene by rotating the
    azimuth angle az_angle in every frame.
    Parameters
    -----------
    ren : vtkRenderer() object
        as returned from function ren()
    cam_pos : None or sequence (3,), optional
        camera position
    cam_focal : None or sequence (3,), optional
        camera focal point
    cam_view : None or sequence (3,), optional
        camera view up
    out_path : str, optional
        output directory for the frames
    path_numbering : bool
        when recording it changes out_path ot out_path + str(frame number)
    n_frames : int, optional
        number of frames to save, default 1
    az_ang : float, optional
        azimuthal angle of camera rotation.
    magnification : int, optional
        how much to magnify the saved frame
    Examples
    ---------
    >>> from dipy.viz import fvtk
    >>> r=fvtk.ren()
    >>> a=fvtk.axes()
    >>> fvtk.add(r,a)
    >>> #uncomment below to record
    >>> #fvtk.record(r)
    >>> #check for new images in current directory
    '''
    # if ren is None:
    #     ren = vtk.vtkRenderer()

    ren.ResetCamera()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren)
    picker = vtk.vtkCellPicker()
    # window.SetAAFrames(6)
    title = 'Dipy'
    window.SetWindowName(title)
    window.SetSize(size[0], size[1])
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(window)
    iren.SetPicker(picker)


    renderLarge = vtk.vtkRenderLargeImage()
    png_magnify = 1
    renderLarge.SetInput(ren)

    renderLarge.SetMagnification(png_magnify)
    renderLarge.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(renderLarge.GetOutputPort())
    filename = out_path
    writer.SetFileName(filename)
    writer.Write()
            # print('Look for fvtk.png in your current working directory.')

    # iren.AddObserver('KeyPressEvent', key_press)
    iren.SetInteractorStyle(style)
    iren.Initialize()
    picker.Pick(85, 126, 0, ren)
    window.Render()
    iren.Start()

    # window.RemoveAllObservers()
    # ren.SetRenderWindow(None)
    window.RemoveRenderer(ren)
    ren.SetRenderWindow(None)

        # ang = +az_ang


# dname = '/home/local/USHERBROOKE/stjs2902/Dropbox/share_sam_maxime/3T_mic_1.2mm/nlsam_sh_smooth/bundles/'
# bundle = "bundles_prob_ilf.right.trk"

# streamlines_file = dname + bundle
# streams, hdr = nib.trackvis.read(streamlines_file, points_space='rasmm')
# lines = [s[0] for s in streams]

# """
# 3. Generate and render fvtk streamline with line (without width)
# """
# fvtk_lines = vtk_a.line(lines)

# # renderer = fvtk.ren()
# fvtk.add(renderer, fvtk_lines)
# # fvtk.show(renderer)

# rotation = vtk_u.rotation_from_lines(lines, use_line_dir=True, use_full_eig=False)
# cam = renderer.GetActiveCamera()
# cam.ApplyTransform(rotation)
# cam.Roll(-90)
# fvtk.show(renderer)
# # """
# 4. Generate and render fvtk streamline with tubes (with width)
# """
# tube_width = 0.1
# fvtk_tubes = vtk_a.streamtube(lines, linewidth=tube_width)

# renderer = fvtk.ren()
# fvtk.add(renderer, fvtk_tubes)
# fvtk.show(renderer)

# """
# 5. Generate random color
# """
# colors = np.random.rand(len(lines), 3) #color per line
# #colors = [np.random.rand(len(lines[i]), 3)
# #			for i in range(len(lines))] #color per point
# #colors = np.array([0.8,0.0,0.0]) #color per bundle/display
# #colors = None #default color (direction rgb)

# """
# 6. Generate and render fvtk streamline with color
# """
# fvtk_lines = vtk_a.streamtube(lines, colors, linewidth=tube_width)

# renderer = fvtk.ren()
# fvtk.add(renderer, fvtk_lines)
# fvtk.show(renderer)
