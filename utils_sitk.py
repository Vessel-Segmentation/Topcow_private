import os

import SimpleITK as sitk


def convert_mha_nii(img_path, file_ending, save_path="."):
    """
    convert between .mha metaImage to .nii.gz nifti compressed
    file_ending='.nii.gz' | '.mha'
    """
    img = sitk.ReadImage(img_path)
    source_fname = os.path.basename(img_path)
    target_fname = source_fname.split(".")[0] + file_ending
    sitk.WriteImage(img, os.path.join(save_path, target_fname), useCompression=True)


if __name__ == "__main__":
    img_path = "E:/git_clone_repo/Topcow_private/test/expected_output/topcow_mr_whole_066_testdocker_bin_seg.mha"
    convert_mha_nii(img_path, ".nii.gz", save_path="E:/git_clone_repo/Topcow_private/test/expected_output/")

    img_path = "E:/git_clone_repo/Topcow_private/test/input/images/head-mr-angio/uuid_of_mr_whole_066.mha"
    convert_mha_nii(img_path, ".nii.gz", save_path="E:/git_clone_repo/Topcow_private/test/input/images/head-mr-angio/")

    print("Done")
