from pyfbsdk import *
from pyfbsdk_additions import *
import os, glob
from math import sin, cos, radians

model_to_load = "Michelle"
render_takes = ["AI"] #Both determines which mocab is loaded and which takes are rendered
render_content = True
content_tag = None

content_path = os.path.join(r"C:\Users\maddo\Documents",r"GitHub\Contemporarty-Dance-AI\content\motion_builder")
character_model_path = content_path
motion_capture_path = os.path.join(content_path, "motion_capture")
rendered_content_path = os.path.join(content_path, "renders")

character_model = os.path.join(character_model_path, model_to_load + ".fbx")
motion_capture_data = sorted([f for f in glob.glob(motion_capture_path+r"\*") if f.endswith(".fbx") or f.endswith(".bvh")], key=str.upper)
t_pose = motion_capture_data.pop()
mocab_to_import = [m for m in motion_capture_data for t in render_takes if t.split('_')[0] in m.split('\\')[-1]]

class Camera():
    def __init__(self, obj_to_aim_at, distance, height, show_grid = False):
        self.distance = distance # from hips
        self.height = height # relative height (hip as reference)
        self.show_grid = show_grid
        self.camera, self.aim = self._initializeCamera(obj_to_aim_at)

    def _setConstraintReferenceByName(self, constraint, model, referenceName ):
        for i in range( 0, constraint.ReferenceGroupGetCount() ):
            if constraint.ReferenceGroupGetName( i ) == referenceName:
                constraint.ReferenceAdd( i, model )
            
    def _initializeCamera(self, obj_to_aim_at):
        # Create camera
        camera = FBCamera('Render_Camera')
        camera.Show = True
        camera.ViewShowGrid = self.show_grid
        camera.BackGroundColor = FBColor(0.600397,0.575328,0.624)#.65,.6,.65
        # Create Aim for Camera
        model = FBFindModelByLabelName(obj_to_aim_at)
        point_cube = FBModelCube(model.Name+'_Aim')
        aim = FBCreateObject( 'Browsing/Templates/Constraints', 'Aim', model.Name+'_Aim' )
        FBSystem().Scene.Constraints.append( aim ) # Get a reference to the current MotionBuilder scene from the underlying system properties of MotionBuilder.
        self._setConstraintReferenceByName( aim, camera, 'Constrained Object' )
        self._setConstraintReferenceByName( aim, point_cube, 'Aim At Object' )
        aim.Active = True
        return camera, aim

class Floor():
    def __init__(self, texture_img, texture_label, width_factor=1, translate=True, show_floor=True):
        self.texture_label = texture_label
        self.width_factor = width_factor
        self.translate = translate
        self.show_floor=show_floor
        self._create_floor(texture_img)

    def _create_floor(self, texture_img):
        plane = FBModelPlane('Floor')
        plane.Show = self.show_floor
        plane.Translation = FBVector3d(0, 0, -1)
        plane.Scaling = FBVector3d(5*self.width_factor, 1, 5)
        texture = FBTexture( texture_img )
        layered_texture = FBLayeredTexture(self.texture_label)
        layered_texture.Layers.append(texture)
        layered_texture.Scaling = FBVector3d(6.13*self.width_factor, 6.13, 1)
        if self.translate:
            layered_texture.Translation = FBVector3d(.13,.25,0)#FBVector3d(.0175,-.07,0)
        layered_texture.HardSelect() # HardSelect layeredTexture to bring up its setting UI
        material = FBMaterial('dance_floor')
        '''material.Emissive = FBColor(.2,.2,.2)#(.09, .06, .1)
        material.Ambient = FBColor(.2,.2,.2)#(1,1,1)
        material.Diffuse = FBColor(.2,.2,.2)#(.93, 1, .99)'''
        material.Specular = FBColor(.46,.51,.52)#(.63,.68,.68)#(1,1,1)
        material.Shininess = 1#1#4.8
        material.SetTexture(layered_texture, FBMaterialTextureType.kFBMaterialTextureDiffuse)
        plane.Materials.append(material)


class Render_Options():
    def __init__(self, render_type, takes_to_render, output_dir, render=True, label_name=None):
        self.render_type = ['_####.jpg','.avi'][render_type]
        self.takes_to_render = takes_to_render
        self.output_dir = output_dir
        self.render = render
        self.label_name = '_(' + label_name + ')' if label_name else ""
        self.options = self._init_options()

    def _init_options(self):
        options = FBVideoGrabber().GetOptions()
        #options.TimeSpan is set during take render runtime using system.CurrentTake.LocalTimeSpan
        options.TimeSteps = FBTime(0, 0, 0, 1)
        options.CameraResolution = FBCameraResolutionMode().kFBResolutionHD#HD/Custom/FullScreen
        options.BitsPerPixel = FBVideoRenderDepth().FBVideoRender24Bits#32
        options.FieldMode = FBVideoRenderFieldMode().FBFieldModeNoField
        options.ViewingMode = FBVideoRenderViewingMode().FBViewingModeXRay
        #options.OutputFileName = self.output_dir
        options.ShowSafeArea = False
        options.ShowTimeCode = False
        options.ShowCameraLabel = False
        options.AntiAliasing = False
        options.RenderAudio = False
        options.StillImageCompression = 100
        #options.AudioRenderFormat = self._init_audio_options()
        return options

    def _init_audio_options(self):
        audio_options = FBAudioRenderOptions()
        audio_options.ChannelMode = FBAudioChannelMode.kFBAudioChannelModeStereo
        audio_options.BitDepthMode = FBAudioBitDepthMode.kFBAudioBitDepthMode_16
        audio_options.RateMode = FBAudioRateMode.kFBAudioRateMode_44100
        return audio_options

    def _setCameraLocation(self, camera, object_to_aim_at):
        # Get BVH:Hips vectors
        model = FBFindModelByLabelName(object_to_aim_at)
        v_trans = FBVector3d()
        v_rot = FBVector3d()
        model.GetVector(v_rot, FBModelTransformationType.kModelRotation, True)
        model.GetVector(v_trans, FBModelTransformationType.kModelTranslation, True)
        v_direction = [v_rot[0]/abs(v_rot[0]), v_rot[1]/abs(v_rot[1])]
        # Set Look At Location to Hips Position
        look_at = [c for c in FBSystem().Scene.Constraints if model.Name in c.Name][0]
        look_at.ReferenceGet(1).SetVector(v_trans, FBModelTransformationType.kModelTranslation, True)
        # Determine location of camera
        v_trans[0] += camera.distance*sin(radians(v_rot[1]))
        v_trans[1] *= camera.height
        v_trans[2] += camera.distance*cos(radians(abs(v_rot[1])))*-v_direction[0]
        camera.camera.SetVector(v_trans, FBModelTransformationType.kModelTranslation, True)

    def renderTake(self, index, camera, object_to_aim_at):
        if self.render:
            scene = FBSystem().Scene
            FBSystem().CurrentTake = scene.Takes[index]
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            filename = FBSystem().CurrentTake.Name + self.label_name + self.render_type
            self.options.OutputFileName = os.path.join(self.output_dir, filename)
            self.options.TimeSpan = FBSystem().CurrentTake.LocalTimeSpan
            '''if not FBSystem().Scene.Renderer.IsCameraSwitcherInPane(0):
                                                    FBSystem().Scene.Renderer.SetCameraSwitcherInPane( 0, True )
                                                FBSystem().Scene.Renderer.SetCameraInPane( camera.camera, 0 )
            FBCameraSwitcher().CurrentCamera.Name = camera.camera'''
            scene.Renderer.UseCameraSwitcher = True
            FBCameraSwitcher().CurrentCamera = camera.camera
            self._setCameraLocation(camera, object_to_aim_at)
            return FBApplication().FileRender(self.options)
        else:
            return False

def deselectComponents():
    for component in FBSystem().Scene.Components:
        component.Selected = False

def characterizeModel(rootNamespace):
    deselectComponents()
    # Create a character.
    character = FBCharacter(rootNamespace)#[:-1]
    # Map model joints to character joints
    for component in [c for c in FBSystem().Scene.Components if rootNamespace in c.LongName]:
        slot = character.PropertyList.Find(component.Name+'Link')
        if slot is not None:
            slot.append(component)
    # Flag that the character has been characterized.
    character.SetCharacterizeOn(True)
    # Create a control rig using Forward and Inverse Kinematics,
    # as specified by the "True" parameter.
    #character.CreateControlRig(True)
    # Set the control rig to active.
    #character.ActiveInput = True
    return character

def import_dances(mocab_files):
    import_options = FBMotionFileOptions(FBStringList('~'.join(mocab_files)))
    import_options.CreateInsteadOfMerge = True #does opposite now?
    import_options.CreateUnmatchedModels = True
    import_options.ImportDOF = True
    FBApplication().FileImportWithOptions(import_options)
    return FBSystem().CurrentTake

def main():
    FBApplication().FileNew()
    t_take = import_dances([t_pose])
    import_dances(mocab_to_import)
    hip_camera = Camera('BVH:Hips', 400, 1.5)
    Floor(os.path.abspath(os.path.join( content_path,"tiled_floor.jpg" )), "tiled_floor")
    renderer = Render_Options(1, render_takes, rendered_content_path, render_content, content_tag)

    scene = FBSystem().Scene
    if model_to_load:
        renderer.options.ViewingMode = FBVideoRenderViewingMode().FBViewingModeModelsOnly
        FBApplication().FileMerge(character_model, False) 
        FBSystem().CurrentTake = t_take
        mixamo = characterizeModel('mixamorig')
        mixamo.ActiveInput = True
        characterizeModel('BVH')
        scene.Characters[0].InputCharacter = scene.Characters[1]
        scene.Characters[0].InputType = FBCharacterInputType.kFBCharacterInputCharacter

    for index, take in enumerate(scene.Takes):
        for i in [index for t in renderer.takes_to_render if t in take.Name]:
            print renderer.renderTake(i, hip_camera, 'BVH:Hips')
            
if __name__ in ('__main__', '__builtin__'):
    main()
    