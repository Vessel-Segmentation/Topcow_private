import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk



class Postprocessing():




    def cube_to_whole(self,upper_left_coord:np.ndarray,
                      whole_size:np.ndarray,
                      cube_size:np.ndarray,
                      pred_label:np.ndarray) -> np.ndarray:
        output_pred = np.zeros(whole_size)
        sag_size=upper_left_coord[0] + cube_size[0]
        coro_size=upper_left_coord[1] + cube_size[1]
        axial_size=upper_left_coord[2] + cube_size[2]
        output_pred[upper_left_coord[0]:sag_size,upper_left_coord[1]:coro_size,upper_left_coord[2]:axial_size]=pred_label

        return output_pred

