"""
The most important file for Grand-Challenge Algorithm submission is this process.py.
This is the file where you will extend our base algorithm class,
and modify the subclass of MyCoWSegAlgorithm for your awesome algorithm :)
Simply update the TODO in this file.

NOTE: remember to COPY your required files in your Dockerfile
COPY --chown=user:user <somefile> /opt/app/
"""

import numpy as np
import SimpleITK as sitk
import os
import nibabel as nib
#import torch

import sys

sys.path.append('/opt/app/nnunet/')

from base_algorithm import TASK, TRACK, BaseAlgorithm
from Preprocessing import Preprocessing
from Postprocessing import Postprocessing
#from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
import multiprocessing

#######################################################################################
# TODO: First, choose your track and task!
# track is either TRACK.CT or TRACK.MR
# task is either TASK.BINARY_SEGMENTATION or TASK.MULTICLASS_SEGMENTATION
track = TRACK.MR
task = TASK.MULTICLASS_SEGMENTATION
# END OF TODO
#######################################################################################


class MyCoWSegAlgorithm(BaseAlgorithm):
    """
    Your algorithm goes here.
    Simply update the TODO in this file.
    """

    def __init__(self):
        super().__init__(
            track=track,
            task=task,
        )
        self.original_path="/opt/app" # docker
        #self.original_path="/n02dat01/users/jlliu/Topcow_private"  
        self.output_path=os.path.join(self.original_path, "imagesTs")
        os.makedirs(self.output_path, exist_ok=True)
        self.transform=False
        self.module_folder=os.path.join(self.original_path,  "module","nnUNet_results", "Dataset055_ROIAugmentation","nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres")
        



        #######################################################################################
        # TODO: load and initialize your model here
        # self.model = ...
        # self.device = ...

        # END OF TODO
        #######################################################################################

    def preprocessing(self, origin_image_mr: str):
        """

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
        # skull stripping
        pre_processed_dir=os.path.join(self.output_path, 'pre_processed')
        os.makedirs(pre_processed_dir, exist_ok=True)
        pre=Preprocessing(self.original_path)
        #mask_file=os.path.join(pre_processed_dir, 'brain_mask.nii.gz')
        #brain_file=os.path.join(pre_processed_dir, 'brain.nii.gz')
        #pre.skull_stripping(origin_image_mr, brain_file)
        # register-transform
        if not self.transform:
            #cube = pre.register_template(brain_file, pre_processed_dir)# thera isn't bet can b used. so skip the skull strip
            # cube = pre.register_template(origin_image_mr, pre_processed_dir)
            fixed_image=origin_image_mr
            moving_image=os.path.join(self.original_path, "template","AVG_TOF_MNI_SS_down.nii.gz")
            output_dir=os.path.join(pre_processed_dir, "WILLIS_ANTS")
            input_image = os.path.join(self.original_path, "template", "willis_cube_down.nii.gz")
            output_image = os.path.join(pre_processed_dir, "willis_cube.nii.gz")
            transform_mat = os.path.join(pre_processed_dir, "WILLIS_ANTS0GenericAffine.mat")


            os.system('antsRegistrationSyNQuick.sh '+'-d 3 '+'-f '+fixed_image+' -m '+moving_image+' -o '+output_dir+' -t a')
            os.system(
                'antsApplyTransforms ' + ' -i ' + input_image + '-r ' + fixed_image + ' -o ' + output_image + ' -t ' + transform_mat)
        else:
            # cube = pre.apply_transform(
            #     brain_file,
            #     os.path.join(pre_processed_dir, "WILLIS_ANTS0GenericAffine.mat"),
            #     os.path.join(pre_processed_dir, "willis_cube.nii.gz")
            # )
            input_image=os.path.join(original_path, "template", "willis_cube_down.nii.gz")
            output_image=os.path.join(pre_processed_dir, "willis_cube.nii.gz")
            transform_mat=os.path.join(pre_processed_dir, "WILLIS_ANTS0GenericAffine.mat")
            fixed_image = origin_image_mr
            os.system(
                'antsApplyTransforms '+ ' -i ' + input_image + '-r ' + fixed_image + ' -o ' + output_image + ' -t '+transform_mat)
        # crop cube for training
        cube_file=os.path.join(pre_processed_dir, "willis_cube.nii.gz")
        ROI_file=os.path.join(pre_processed_dir, "prediction_0000.nii.gz")
        upper_left_coord=pre.crop_cube(cube_file, brain_file, ROI_file)

        return upper_left_coord, pre_processed_dir


    def postprocessing(self, upper_left_coord, pred_result:np.ndarray, whole_size):
        post=Postprocessing()
        cube_size=pred_result.shape
        output_pred_whole=post.cube_to_whole(upper_left_coord, whole_size, cube_size, pred_result)
        return output_pred_whole







    def predict(self,  image_ct: sitk.Image, image_mr: sitk.Image) -> np.array:
        """
        Inputs will be a pair of CT and MR .mha SimpleITK.Images
        Output is supposed to be an numpy array in (x,y,z)  *,
        """

        #######################################################################################
        # TODO: place your own prediction algorithm here
        # You are free to remove everything! Just return to us an npy in (x,y,z)
        # NOTE: If you extract the array from SimpleITK, note that
        #              SimpleITK npy array axis order is (z,y,x).
        #              Then you might have to transpose this to (x,y,z)
        #              (see below for an example).
        #######################################################################################

        # for example, if you use nnUnet
        # you can create a imagesTs folder with nii.gz files
        # check out SegRap challenge tutorial: https://github.com/HiLab-git/SegRap2023/blob/main/Docker_tutorial/SegRap2023_task1_OARs_nnUNet_Example/process.py
        
        self.result_path = os.path.join(self.output_path, 'result')
        os.makedirs(self.result_path, exist_ok=True)
        # imagesTs_path = os.path.join("./", "imagesTs")
        
        main_nii_path = os.path.join(self.output_path, "inference_case1_0000.nii.gz")

        if track == TRACK.MR:
            print("-> main_input is from TRACK.MR")
            main_input = image_mr
            sec_input = image_ct
        else:
            print("-> main_input is from TRACK.CT")
            main_input = image_ct
            sec_input = image_mr
        sitk.WriteImage(main_input, main_nii_path, useCompression=True)
        if os.path.exists(main_nii_path):
            print("-> input image is ready!")
        else:
            print("-> input image does not exist.")
        
        # if using both modalities as channels, name the other modality with _0001
        # see https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format_inference.md
        # sec_nii_path = os.path.join(imagesTs_path, "inference_case1_0001.nii.gz")
        # sitk.WriteImage(sec_input, sec_nii_path, useCompression=True)
        # Check your imagesTs folder
        # print("imagesTs_path = ", os.path.abspath(imagesTs_path))
        # print(f"list imagesTs_path = {os.listdir(imagesTs_path)}")

        input_=nib.load(main_nii_path)
        input_data=input_.get_fdata()
        affine=input_.affine
        hdr=input_.header
        whole_size=input_data.shape
        # add the model to predict
        upper_left_coord, pre_processed_dir = self.preprocessing(main_nii_path)
        torch.set_num_threads(multiprocessing.cpu_count())
        self.model_predict = predict_from_raw_data(pre_processed_dir,
                                                     self.result_path,
                                                     self.module_folder,
                                                     [4],# fold 4
                                                     0.5,
                                                     True,# use gaussian
                                                     False,# is mirroring
                                                     True,# perform ues gpu
                                                     True,# verbose
                                                     False,# save pro
                                                     True,# overwrite
                                                     'checkpoint_best.pth',
                                                     3,
                                                     3,
                                                     None,
                                                     1,
                                                     0,
                                                     torch.device('cpu'))
        pred_file = os.path.join(self.result_path, 'prediction.nii.gz')
        pred_array = nib.load(pred_file).get_fdata()
        pred_array = self.postprocessing(upper_left_coord, pred_array, whole_size)
        pred_whole_file=os.path.join(self.result_path, 'prediction_whole.nii.gz')
        
        new_nii=nib.Nifti1Image(pred_array,affine,hdr)
        nib.save(new_nii,pred_whole_file)



        # # e.g. this dummy example works for both binary and multi-class segmentation
        # # because label 1 can be either CoW (in binary seg) or BA (in multi-class)
        # stats = sitk.StatisticsImageFilter()
        # stats.Execute(main_input)
        # print("Main input max val: ", stats.GetMaximum())
        #
        # segmentation = sitk.BinaryThreshold(
        #     main_input,
        #     lowerThreshold=stats.GetMaximum() // 3,  # some arbitrary threshold
        #     upperThreshold=stats.GetMaximum(),
        # )
        # # NOTE: SimpleITK npy axis ordering is (z,y,x)!
        # pred_array = sitk.GetArrayFromImage(segmentation)
        #
        # # reorder from (z,y,x) to (x,y,z)
        # pred_array = pred_array.transpose((2, 1, 0)).astype(np.uint8)
        # print("pred_array.shape = ", pred_array.shape)
        # # The output np.array needs to have the same shape as track modality input
        # print(f"main_input.GetSize() = {main_input.GetSize()}")

        # END OF TODO
        #######################################################################################

        # return prediction array
        return pred_array




if __name__ == "__main__":
    # NOTE: running locally ($ python3 process.py) has advantage of faster debugging
    # but please ensure the docker environment also works before submitting
    #MyCoWSegAlgorithm().process()
    input_path="/opt/app/images/crown_001_0000_brain.nii.gz"
    #input_path = os.path.join("/n02dat01/users/jlliu/Topcow_private", "imagesTs", "inference_case1_0000.nii.gz")
    mr=sitk.ReadImage(input_path)
    ct=sitk.ReadImage(input_path)
    my=MyCoWSegAlgorithm()
    upper_left_coord, pre_processed_dir=my.preprocessing(input_path)
    print(upper_left_coord)
    #pred_array=my.predict(ct,mr)
    #print(np.unique(pred_array))
    cowsay_msg = """\n
  ____________________________________
< MyCoWSegAlgorithm().process()  Done! >
  ------------------------------------
         \   ^__^ 
          \  (oo)\_______
             (__)\       )\/\\
                 ||----w |
                 ||     ||
    """
    #print(cowsay_msg)
