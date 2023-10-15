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
import torch

import sys

sys.path.append('/opt/app/nnunet/')

from base_algorithm import TASK, TRACK, BaseAlgorithm
from Preprocessing import Preprocessing
from Postprocessing import Postprocessing
#from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results
import multiprocessing

#######################################################################################
# TODO: First, choose your track and task!
# track is either TRACK.CT or TRACK.MR
# task is either TASK.BINARY_SEGMENTATION or TASK.MULTICLASS_SEGMENTATION
track = TRACK.MR
task = TASK.BINARY_SEGMENTATION
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
            #cube = pre.register_template(brain_file, pre_processed_dir)# thera isn't bet can b used. so skip the skull strip#
            #cube = pre.register_template(origin_image_mr, pre_processed_dir)
            fixed_image=origin_image_mr
            moving_image=os.path.join(self.original_path, "template","AVG_TOF_MNI_SS_down.nii.gz")
            output_dir=os.path.join(pre_processed_dir, "WILLIS_ANTS")
            input_image = os.path.join(self.original_path, "template", "willis_cube_down.nii.gz")
            output_image = os.path.join(pre_processed_dir, "willis_cube.nii.gz")
            transform_mat = os.path.join(pre_processed_dir, "WILLIS_ANTS0GenericAffine.mat")


            os.system('antsRegistrationSyNQuick.sh '+'-d 3 '+'-f '+fixed_image+' -m '+moving_image+' -o '+output_dir+' -t a')
            cube = pre.apply_transform(
                fixed_image,
                os.path.join(pre_processed_dir, "WILLIS_ANTS0GenericAffine.mat"),
                os.path.join(pre_processed_dir, "willis_cube.nii.gz")
            )
            print('registration end')
            #os.system(
            #    'antsApplyTransforms ' + ' -i ' + input_image + '-r ' + fixed_image + ' -o ' + output_image + ' -t ' + transform_mat)
        else:
            cube = pre.apply_transform(
                brain_file,
                os.path.join(pre_processed_dir, "WILLIS_ANTS0GenericAffine.mat"),
                os.path.join(pre_processed_dir, "willis_cube.nii.gz")
            )
        # crop cube for training
        cube_file=os.path.join(pre_processed_dir, "willis_cube.nii.gz")
        ROI_file=os.path.join(pre_processed_dir, "prediction_0000.nii.gz")
        upper_left_coord=pre.crop_cube(cube_file, origin_image_mr, ROI_file)

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
        # self.model_predict = predict_from_raw_data(pre_processed_dir,
        #                                              self.result_path,
        #                                              self.module_folder,
        #                                              [4],# fold 4
        #                                              0.5,
        #                                              True,# use gaussian
        #                                              False,# is mirroring
        #                                              True,# perform ues gpu
        #                                              True,# verbose
        #                                              False,# save pro
        #                                              True,# overwrite
        #                                              'checkpoint_best.pth',
        #                                              3,
        #                                              1,
        #                                              None,
        #                                              1,
        #                                              0,
        #                                              torch.device('cpu'))
        self.model_predict=nnUNetPredictor(
                                            tile_step_size=0.5,
                                            use_gaussian=True,
                                            use_mirroring=False,
                                            perform_everything_on_gpu=False,
                                            device=torch.device('cpu'),
                                            verbose=True,
                                            verbose_preprocessing=False,
                                            allow_tqdm=True
                                            )
        self.model_predict.initialize_from_trained_model_folder(
            self.module_folder,
            use_folds=(4,),
            checkpoint_name='checkpoint_best.pth',
        )
        # predict a numpy array
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
        img, props = SimpleITKIO().read_images([os.path.join(pre_processed_dir, 'prediction_0000.nii.gz')])
        self.model_predict.predict_single_npy_array(img, props, None, os.path.join(self.result_path, 'prediction.nii.gz'), False)
        
        pred_file = os.path.join(self.result_path, 'prediction.nii.gz')
        while True:
            if os.path.exists(pred_file):
                print('the prediction file is done.')
                break
        pred_array_mul = nib.load(pred_file).get_fdata()
        pred_array=np.zeros(pred_array_mul.shape)
        pred_array[np.where((pred_array_mul>0))]=1
        
        pred_array = self.postprocessing(upper_left_coord, pred_array, whole_size)
        pred_whole_file=os.path.join(self.result_path, 'prediction_whole.nii.gz')
        print('result is done.')
        new_nii=nib.Nifti1Image(pred_array,affine,hdr)
        nib.save(new_nii,pred_whole_file)
        #self.save_segmentation(pred_array)



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




    # def predict_raw_data(self):
    #     if device.type == 'cuda':
    #         device = torch.device(type='cuda', index=0)  # set the desired GPU with CUDA_VISIBLE_DEVICES!
    #
    #     if device.type != 'cuda':
    #         perform_everything_on_gpu = False
    #
    #         # let's store the input arguments so that its clear what was used to generate the prediction
    #     my_init_kwargs = {}
    #     for k in inspect.signature(predict_from_raw_data).parameters.keys():
    #         my_init_kwargs[k] = locals()[k]
    #     my_init_kwargs = deepcopy(my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
    #     # safety precaution.
    #     recursive_fix_for_json_export(my_init_kwargs)
    #     maybe_mkdir_p(output_folder)
    #     save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))
    #
    #     if use_folds is None:
    #         use_folds = auto_detect_available_folds(model_training_output_dir, checkpoint_name)
    #
    #     # load all the stuff we need from the model_training_output_dir
    #     parameters, configuration_manager, inference_allowed_mirroring_axes, \
    #     plans_manager, dataset_json, network, trainer_name = \
    #         load_what_we_need(model_training_output_dir, use_folds, checkpoint_name)
    #
    #     # check if we need a prediction from the previous stage
    #     if configuration_manager.previous_stage_name is not None:
    #         if folder_with_segs_from_prev_stage is None:
    #             print(f'WARNING: The requested configuration is a cascaded model and requires predctions from the '
    #                   f'previous stage! folder_with_segs_from_prev_stage was not provided. Trying to run the '
    #                   f'inference of the previous stage...')
    #             folder_with_segs_from_prev_stage = join(output_folder,
    #                                                     f'prediction_{configuration_manager.previous_stage_name}')
    #             # we can only do this if we do not have multiple parts
    #             assert num_parts == 1 and part_id == 0, "folder_with_segs_from_prev_stage was not given and inference " \
    #                                                     "is distributed over more than one part (num_parts > 1). Cannot " \
    #                                                     "automatically run predictions for the previous stage"
    #             predict_from_raw_data(list_of_lists_or_source_folder,
    #                                   folder_with_segs_from_prev_stage,
    #                                   get_output_folder(plans_manager.dataset_name,
    #                                                     trainer_name,
    #                                                     plans_manager.plans_name,
    #                                                     configuration_manager.previous_stage_name),
    #                                   use_folds, tile_step_size, use_gaussian, use_mirroring, perform_everything_on_gpu,
    #                                   verbose, False, overwrite, checkpoint_name,
    #                                   num_processes_preprocessing, num_processes_segmentation_export, None,
    #                                   num_parts=num_parts, part_id=part_id, device=device)
    #
    #     # sort out input and output filenames
    #     if isinstance(list_of_lists_or_source_folder, str):
    #         list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
    #                                                                                    dataset_json['file_ending'])
    #     print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
    #     list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
    #     caseids = [os.path.basename(i[0])[:-(len(dataset_json['file_ending']) + 5)] for i in
    #                list_of_lists_or_source_folder]
    #     print(
    #         f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
    #     print(f'There are {len(caseids)} cases that I would like to predict')
    #
    #     output_filename_truncated = [join(output_folder, i) for i in caseids]
    #     seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + dataset_json['file_ending']) if
    #                                  folder_with_segs_from_prev_stage is not None else None for i in caseids]
    #     # remove already predicted files form the lists
    #     if not overwrite:
    #         tmp = [isfile(i + dataset_json['file_ending']) for i in output_filename_truncated]
    #         not_existing_indices = [i for i, j in enumerate(tmp) if not j]
    #
    #         output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
    #         list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
    #         seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
    #         print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
    #               f'That\'s {len(not_existing_indices)} cases.')
    #         # caseids = [caseids[i] for i in not_existing_indices]
    #
    #     # placing this into a separate function doesnt make sense because it needs so many input variables...
    #     preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
    #     # hijack batchgenerators, yo
    #     # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
    #     # way we don't have to reinvent the wheel here.
    #     num_processes = max(1, min(num_processes_preprocessing, len(list_of_lists_or_source_folder)))
    #     ppa = PreprocessAdapter(list_of_lists_or_source_folder, seg_from_prev_stage_files, preprocessor,
    #                             output_filename_truncated, plans_manager, dataset_json,
    #                             configuration_manager, num_processes)
    #     mta = MultiThreadedAugmenter(ppa, NumpyToTensor(), num_processes, 1, None, pin_memory=device.type == 'cuda')
    #     # mta = SingleThreadedAugmenter(ppa, NumpyToTensor())
    #
    #     if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) and \
    #             (len(
    #                 list_of_lists_or_source_folder) > 5):  # just a dumb heurisitic in order to skip compiling for few inference cases
    #         print('compiling network')
    #         network = torch.compile(network)
    #
    #     # precompute gaussian
    #     inference_gaussian = torch.from_numpy(
    #         compute_gaussian(configuration_manager.patch_size)).half()
    #     if perform_everything_on_gpu:
    #         inference_gaussian = inference_gaussian.to(device)
    #
    #     # num seg heads is needed because we need to preallocate the results in predict_sliding_window_return_logits
    #     label_manager = plans_manager.get_label_manager(dataset_json)
    #     num_seg_heads = label_manager.num_segmentation_heads
    #
    #     # go go go
    #     # spawn allows the use of GPU in the background process in case somebody wants to do this. Not recommended. Trust me.
    #     # export_pool = multiprocessing.get_context('spawn').Pool(num_processes_segmentation_export)
    #     # export_pool = multiprocessing.Pool(num_processes_segmentation_export)
    #     with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
    #         network = network.to(device)
    #
    #         r = []
    #         with torch.no_grad():
    #             for preprocessed in mta:
    #                 data = preprocessed['data']
    #                 if isinstance(data, str):
    #                     delfile = data
    #                     data = torch.from_numpy(np.load(data))
    #                     os.remove(delfile)
    #
    #                 ofile = preprocessed['ofile']
    #                 print(f'\nPredicting {os.path.basename(ofile)}:')
    #                 print(f'perform_everything_on_gpu: {perform_everything_on_gpu}')
    #
    #                 properties = preprocessed['data_properites']
    #
    #                 # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
    #                 # npy files
    #                 proceed = not check_workers_busy(export_pool, r, allowed_num_queued=2 * len(export_pool._pool))
    #                 while not proceed:
    #                     sleep(0.1)
    #                     proceed = not check_workers_busy(export_pool, r, allowed_num_queued=2 * len(export_pool._pool))
    #
    #                 # we have some code duplication here but this allows us to run with perform_everything_on_gpu=True as
    #                 # default and not have the entire program crash in case of GPU out of memory. Neat. That should make
    #                 # things a lot faster for some datasets.
    #                 prediction = None
    #                 overwrite_perform_everything_on_gpu = perform_everything_on_gpu
    #                 if perform_everything_on_gpu:
    #                     try:
    #                         for params in parameters:
    #                             # messing with state dict names...
    #                             if not isinstance(network, OptimizedModule):
    #                                 network.load_state_dict(params)
    #                             else:
    #                                 network._orig_mod.load_state_dict(params)
    #
    #                             if prediction is None:
    #                                 prediction = predict_sliding_window_return_logits(
    #                                     network, data, num_seg_heads,
    #                                     configuration_manager.patch_size,
    #                                     mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
    #                                     tile_step_size=tile_step_size,
    #                                     use_gaussian=use_gaussian,
    #                                     precomputed_gaussian=inference_gaussian,
    #                                     perform_everything_on_gpu=perform_everything_on_gpu,
    #                                     verbose=verbose,
    #                                     device=device)
    #                             else:
    #                                 prediction += predict_sliding_window_return_logits(
    #                                     network, data, num_seg_heads,
    #                                     configuration_manager.patch_size,
    #                                     mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
    #                                     tile_step_size=tile_step_size,
    #                                     use_gaussian=use_gaussian,
    #                                     precomputed_gaussian=inference_gaussian,
    #                                     perform_everything_on_gpu=perform_everything_on_gpu,
    #                                     verbose=verbose,
    #                                     device=device)
    #                         if len(parameters) > 1:
    #                             prediction /= len(parameters)
    #
    #                     except RuntimeError:
    #                         print(
    #                             'Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
    #                             'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...')
    #                         print('Error:')
    #                         traceback.print_exc()
    #                         prediction = None
    #                         overwrite_perform_everything_on_gpu = False
    #
    #                 if prediction is None:
    #                     for params in parameters:
    #                         network.load_state_dict(params)
    #                         if prediction is None:
    #                             prediction = predict_sliding_window_return_logits(
    #                                 network, data, num_seg_heads,
    #                                 configuration_manager.patch_size,
    #                                 mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
    #                                 tile_step_size=tile_step_size,
    #                                 use_gaussian=use_gaussian,
    #                                 precomputed_gaussian=inference_gaussian,
    #                                 perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
    #                                 verbose=verbose,
    #                                 device=device)
    #                         else:
    #                             prediction += predict_sliding_window_return_logits(
    #                                 network, data, num_seg_heads,
    #                                 configuration_manager.patch_size,
    #                                 mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
    #                                 tile_step_size=tile_step_size,
    #                                 use_gaussian=use_gaussian,
    #                                 precomputed_gaussian=inference_gaussian,
    #                                 perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
    #                                 verbose=verbose,
    #                                 device=device)
    #                     if len(parameters) > 1:
    #                         prediction /= len(parameters)
    #
    #                 print('Prediction done, transferring to CPU if needed')
    #                 prediction = prediction.to('cpu').numpy()
    #
    #                 if should_i_save_to_file(prediction, r, export_pool):
    #                     print(
    #                         'output is either too large for python process-process communication or all export workers are '
    #                         'busy. Saving temporarily to file...')
    #                     np.save(ofile + '.npy', prediction)
    #                     prediction = ofile + '.npy'
    #
    #                 # this needs to go into background processes
    #                 # export_prediction(prediction, properties, configuration_name, plans, dataset_json, ofile,
    #                 #                   save_probabilities)
    #                 print('sending off prediction to background worker for resampling and export')
    #                 r.append(
    #                     export_pool.starmap_async(
    #                         export_prediction_from_softmax,
    #                         ((prediction, properties, configuration_manager, plans_manager,
    #                           dataset_json, ofile, save_probabilities),)
    #                     )
    #                 )
    #                 print(f'done with {os.path.basename(ofile)}')
    #         [i.get() for i in r]
    #
    #     # we need these two if we want to do things with the predictions like for example apply postprocessing
    #     shutil.copy(join(model_training_output_dir, 'dataset.json'), join(output_folder, 'dataset.json'))
    #     shutil.copy(join(model_training_output_dir, 'plans.json'), join(output_folder, 'plans.json'))


if __name__ == "__main__":
    # NOTE: running locally ($ python3 process.py) has advantage of faster debugging
    # but please ensure the docker environment also works before submitting
    MyCoWSegAlgorithm().process()
    #input_path="/n02dat01/users/jlliu/TopCOW_MICCAI2023/TopCoW_Data_MICCAI2023/topcow_batch-1_40pairMRCT_30062023/imagesTr/topcow_mr_whole_001_0000.nii.gz"
    #input_path = os.path.join("/n02dat01/users/jlliu/Topcow_private", "imagesTs", "inference_case1_0000.nii.gz")
    #mr=sitk.ReadImage(input_path)
    #ct=sitk.ReadImage(input_path)
    #my=MyCoWSegAlgorithm()
    #upper_left_coord, pre_processed_dir=my.preprocessing(input_path)
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
