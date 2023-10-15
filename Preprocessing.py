import nibabel as nib
from nipype.interfaces import afni
from nipype.interfaces.ants import ApplyTransforms, RegistrationSynQuick
import numpy as np
import pandas as pd
import SimpleITK as sitk
import numpy as np
import SimpleITK as sitk
import os
from nipype.interfaces.fsl import BET
import multiprocessing
import copy, pprint



class Preprocessing():
    def __init__(self,original_path):
        self.template_path=os.path.join(original_path, "template","AVG_TOF_MNI_SS_down.nii.gz")
        self.template_cube_path = os.path.join(original_path, "template", "willis_cube_down.nii.gz")

    #def skull_stripping(self,in_file: str, out_file: str, out_brain_masked: str):
        """Generate mask file using AFNI SkullStrip and Automask.

        Args:
            in_file (str): TOF file used as AFNI SkullStrip input.
            out_file (str): Binary mask file path.
            out_brain_masked (str): Brain masked file path.

        Returns:
           nib.Nifti1Image : binary mask file.
        """
        # I/O parsing
    #    skullstrip = afni.SkullStrip()

    #    skullstrip.inputs.in_file = in_file
    #    skullstrip.inputs.args = "-overwrite"
    #    skullstrip.inputs.out_file = out_file
    #    skullstrip.inputs.outputtype = "NIFTI_GZ"
    #    skullstrip.run()

    #    automask = afni.Automask()
    #    automask.inputs.in_file = out_file
    #    automask.inputs.args = "-overwrite"
    #    automask.inputs.outputtype = "NIFTI"
    #    automask.inputs.out_file = out_file
    #    automask.inputs.brain_file = out_brain_masked
    #    automask.run()

        # Binary conversion of mask
    #    return nib.load(out_file)
        
    def skull_stripping(self,in_file: str, out_file: str): 
        """Generate mask file using AFNI SkullStrip and Automask.

        Args:
            in_file (str): TOF file used as AFNI SkullStrip input.
            out_file (str): Binary mask file path.
            
        """
        skullstrip=BET()
        
        skullstrip.inputs.in_file = in_file
        skullstrip.inputs.out_file = out_file
        skullstrip.inputs.frac=0.2
        skullstrip.run()
        
        
    def apply_transform(self,
            fixed_file: str,
            transform: str,
            output: str,
            interpolation="NearestNeighbor",
            invert=False,
    ) -> np.ndarray:
        """Apply transformation using ANTs on the template file.
        template_file: str,

        Args:
            template_file (str): File to be register.
            fixed_file (str): Reference file used for registration.
            transform (str): Transformation to apply on the template file.
            output (str): Where to save the registration result.
            interpolation (str, optional): Type of interpolation. Defaults to "NearestNeighbor".
            invert (bool, optional): Indicate if the transformation should be reversed.
                                     Defaults to False.

        Returns:
            np.ndarray: Data of the template file registered.
        """
        template_file = self.template_cube_path
        at = ApplyTransforms()
        at.inputs.dimension = 3
        at.inputs.input_image = template_file
        at.inputs.reference_image = fixed_file
        at.inputs.output_image = output
        at.inputs.transforms = [transform]
        at.inputs.invert_transform_flags = [invert]
        at.inputs.interpolation = interpolation
        at.inputs.default_value = 0
        at.run()

        return nib.load(output).get_fdata("unchanged")       
        

    def register_template(self,
            fixed_file: str,
            output_dir: str,
            return_transform=False):
        # Command line
        """Equivalent bash command line call to ANTs.
                    moving_file: str,
            template_file: str,

        antsRegistration --verbose 1 \
            --dimensionality 3 \
            --float 0 \
            --collapse-output-transforms 1 \
            --output [ WILLIS_ANTS,WILLIS_ANTSWarped.nii.gz,WILLIS_ANTSInverseWarped.nii.gz ] \
            --interpolation Linear \
            --use-histogram-matching 1 \
            --winsorize-image-intensities [ 0.005,0.995 ] \
            -x [ NN_rabox_mask.nii.gz, NULL ] \
            --initial-moving-transform [ NN_rabox.nii.gz,/Users/kw2/masks/AVG_TOF_MNI.nii.gz,1 ] \
            --transform Rigid[ 0.1 ] \
            --metric MI[ NN_rabox.nii.gz,/Users/kw2/masks/AVG_TOF_MNI.nii.gz,1,32,Regular,0.25 ] \
            --convergence [ 1000x500x250x0,1e-6,10 ] \
            --shrink-factors 12x8x4x2 \
            --smoothing-sigmas 4x3x2x1vox \
            --transform Affine[ 0.1 ] \
            --metric MI[ NN_rabox.nii.gz,/Users/kw2/masks/AVG_TOF_MNI.nii.gz,1,32,Regular,0.25 ] \
            --convergence [ 1000x500x250x0,1e-6,10 ] \
            --shrink-factors 12x8x4x2 \
            --smoothing-sigmas 4x3x2x1vox

        Args:
            fixed_file (str): Reference used to compute transformations.
            moving_file (str): File to register to fixed file.
            template_file (str): Template file where transformations are applied.
            output_dir (str): output directory to put transformations.
            return_transform (bool, optional): Return transformation. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, Optional[str]]]:
                template data registered, with transformation if enabled.
        """
        moving_file=self.template_path
        # template_file=self.template_cube_path
        # Nipype
        reg = RegistrationSynQuick()
        reg.inputs.fixed_image = fixed_file
        reg.inputs.moving_image = moving_file
        # Use 1 to have deterministic registration
        reg.inputs.num_threads = multiprocessing.cpu_count()
        reg.inputs.transform_type = 'a'
        reg.inputs.output_prefix = os.path.join(output_dir, "WILLIS_ANTS")


        #outputs = reg._list_outputs()
        #pprint.pprint(outputs)
        #print(reg.cmdline)
        reg.run()

        # Template Transformation
        sphere = self.apply_transform(
            fixed_file,
            os.path.join(output_dir, "WILLIS_ANTS0GenericAffine.mat"),
            os.path.join(output_dir, "willis_cube.nii.gz"),
        )

        if return_transform:
            return sphere, os.path.join(output_dir, "WILLIS_ANTS0GenericAffine.mat")

        return sphere



    """
    def register_template(self,
            fixed_file: str,
            output_dir: str,
            return_transform=False):
        # Command line
        Equivalent bash command line call to ANTs.
                    moving_file: str,
            template_file: str,

        antsRegistration --verbose 1 \
            --dimensionality 3 \
            --float 0 \
            --collapse-output-transforms 1 \
            --output [ WILLIS_ANTS,WILLIS_ANTSWarped.nii.gz,WILLIS_ANTSInverseWarped.nii.gz ] \
            --interpolation Linear \
            --use-histogram-matching 1 \
            --winsorize-image-intensities [ 0.005,0.995 ] \
            -x [ NN_rabox_mask.nii.gz, NULL ] \
            --initial-moving-transform [ NN_rabox.nii.gz,/Users/kw2/masks/AVG_TOF_MNI.nii.gz,1 ] \
            --transform Rigid[ 0.1 ] \
            --metric MI[ NN_rabox.nii.gz,/Users/kw2/masks/AVG_TOF_MNI.nii.gz,1,32,Regular,0.25 ] \
            --convergence [ 1000x500x250x0,1e-6,10 ] \
            --shrink-factors 12x8x4x2 \
            --smoothing-sigmas 4x3x2x1vox \
            --transform Affine[ 0.1 ] \
            --metric MI[ NN_rabox.nii.gz,/Users/kw2/masks/AVG_TOF_MNI.nii.gz,1,32,Regular,0.25 ] \
            --convergence [ 1000x500x250x0,1e-6,10 ] \
            --shrink-factors 12x8x4x2 \
            --smoothing-sigmas 4x3x2x1vox

        Args:
            fixed_file (str): Reference used to compute transformations.
            moving_file (str): File to register to fixed file.
            template_file (str): Template file where transformations are applied.
            output_dir (str): output directory to put transformations.
            return_transform (bool, optional): Return transformation. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, Optional[str]]]:
                template data registered, with transformation if enabled.
        
        moving_file=self.template_path
        # template_file=self.template_cube_path
        # Nipype
        reg = Registration()
        reg.inputs.verbose = True
        reg.inputs.dimension = 3
        reg.inputs.float = False
        reg.inputs.collapse_output_transforms = True
        #reg.inputs.write_composite_transform = True
        reg.inputs.output_warped_image = os.path.join(output_dir, "AVG_TOF_MNI_native.nii.gz")
        reg.inputs.output_inverse_warped_image = os.path.join(output_dir, "NN_rabox_MNI.nii.gz")
        reg.inputs.output_transform_prefix = os.path.join(output_dir, "WILLIS_ANTS")
        reg.inputs.interpolation = "NearestNeighbor"
        reg.inputs.use_histogram_matching = True
        reg.inputs.winsorize_lower_quantile = 0.025
        reg.inputs.winsorize_upper_quantile = 0.975
        reg.inputs.initial_moving_transform_com = 1
        reg.inputs.transforms = ["Rigid", "Affine"]
        reg.inputs.transform_parameters = [(0.1,), (0.1,)]
        reg.inputs.metric = ["MI", "MI"]
        reg.inputs.fixed_image = fixed_file
        reg.inputs.moving_image = moving_file
        reg.inputs.metric_weight = [1, 1]
        reg.inputs.radius_or_number_of_bins = [32, 32]
        reg.inputs.sampling_strategy = ["Regular", "Regular"]
        reg.inputs.sampling_percentage = [0.25, 0.25]
        reg.inputs.number_of_iterations = [
            [1000, 500, 250, 100, 50],
            [1000, 500, 250, 100, 50],
        ]
        reg.inputs.convergence_threshold = [1.6e-6, 1.6e-6]
        reg.inputs.shrink_factors = [[12, 8, 4, 2, 1], [12, 8, 4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[4, 3, 2, 1, 0], [4, 3, 2, 1, 0]]
        reg.inputs.sigma_units = ["vox", "vox"]
        # Use 1 to have deterministic registration
        reg.inputs.num_threads = multiprocessing.cpu_count()
        outputs = reg._list_outputs()
        pprint.pprint(outputs)
        print(reg.cmdline)
        reg.run()

        # Template Transformation
        sphere = apply_transform(
            # template_file,
            fixed_file,
            os.path.join(output_dir, "WILLIS_ANTS0GenericAffine.mat"),
            os.path.join(output_dir, "willis_cube.nii.gz"),
        )

        if return_transform:
            return sphere, os.path.join(output_dir, "WILLIS_ANTS0GenericAffine.mat")

        return sphere
    """




    def crop_cube(self,
                  cube_file:str,
                  brain_file:str,
                  output_file:str) -> np.ndarray:
        """Crop the cube for training.

        Args:
            cube_file (str): the COW cube file.
            brain_file (str): the brain file for cropping.
            output_file (str): Where to save the train cube result.

        Returns:
            np.ndarray: the upper left coordinate[sag, coro, axial].
        """

        input_image = sitk.ReadImage(brain_file)
        input_data = sitk.GetArrayFromImage(input_image)

        cube_image = sitk.ReadImage(cube_file)
        cube = sitk.GetArrayFromImage(cube_image)
        original_spacing = cube_image.GetSpacing()
        original_origin = cube_image.GetOrigin()
        original_direction = cube_image.GetDirection()

        # Finding bounding index of cube

        (sag, coro, axial) = np.nonzero(cube)
        sag_min = sag.min()
        sag_max = sag.max()
        coro_min = coro.min()
        coro_max = coro.max()
        axial_min = axial.min()
        axial_max = axial.max()
        print("sag_min:", sag_min)
        print("sag_max:", sag_max)
        print("coro_min:", coro_min)
        print("coro_max:", coro_max)
        print("axial_min:", axial_min)
        print("axial_max:", axial_max)

        # height=sag_max-sag_min
        # sag_min=sag_min+int(height/5)
        # sag_max=sag_max-int(height/5)
        # height = coro_max - coro_min
        # coro_min=coro_min+int(height/5)
        # coro_max = coro_max - int(height / 5)
        # Getting the cube to predict on the TOF
        cube_tof = input_data[sag_min:sag_max, coro_min:coro_max, axial_min:axial_max]
        out_size = [int(sag_max - sag_min), int(coro_max - coro_min), int(axial_max - axial_min)]

        cube_new_image = sitk.GetImageFromArray(cube_tof)
        cube_new_image.SetSpacing(original_spacing)
        # cube_new_image.SetSize(out_size)
        cube_new_image.SetDirection(original_direction)
        cube_new_image.SetOrigin(original_origin)

        sitk.WriteImage(cube_new_image, output_file)

        return np.array([axial_min,coro_min,sag_min])