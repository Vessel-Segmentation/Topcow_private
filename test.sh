#!/usr/bin/env bash

# If you intended to pass a host directory, use absolute path.
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
#SCRIPTPATH=/mnt/e/git_clone_repo/Topcow_private
echo "SCRIPTPATH = ${SCRIPTPATH}"
#/mnt/e/git_clone_repo/Topcow_private
#/alidata/www/wwwroot/ftp
#bash ./build.sh

# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="30g"


#OUTPUT_VOL="topcow-test-docker-output"
OUTPUT_VOL=$SCRIPTPATH/test/output
echo "OUTPUT_VOL = ${OUTPUT_VOL}"

#docker volume create "topcow-test-docker-output"

#         --memory="${MEM_LIMIT}" \
#        --memory-swap="${MEM_LIMIT}" \        --pids-limit="256" \--ipc=host  \--memory-swap="30g" \ --rm \

#		--pids-limit="-1" \
#		--memory="${MEM_LIMIT}" \
#        --memory-swap="${MEM_LIMIT}" \

# Do not change any of the parameters to docker run, these are fixed
# This is to mimic a restricted Grand-Challenge running environment
# ie no internet and no new privileges etc.
docker run --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
		--ipc=host  \
        -v $SCRIPTPATH/test/input/images/head-ct-angio:/input/images/head-ct-angio \
        -v $SCRIPTPATH/test/input/images/head-mr-angio:/input/images/head-mr-angio \
        -v ${OUTPUT_VOL}:/output/ \
        cowsegmentation

#        --gpus=all \
###################################################################################
# Test if the docker outputs match the expected outputs in ./test/expected_output/
###################################################################################

echo "#################################################"
echo "##### Test 0 >>> segmentation mask check"
# Compare the segmentation output from Docker with the expected segmentation mask
# TODO: Provide the expected output segmentation mask of your algorithm in ./test/expected_output/
# TODO: In the python code snippet below change the following if necessary:

TASK="multiclass"  # "binary" or "multiclass"
IMAGE_FILENAME="uuid_of_mr_whole_066.mha"
EXPECTED_SEG_MASK="topcow_mr_whole_066_testdocker_bin_seg.mha"

echo "TODO: change TASK, IMAGE_FILENAME and EXPECTED_SEG_MASK if needed"

docker run --rm \
        -v ${OUTPUT_VOL}:/output/ \
        -v $SCRIPTPATH/test/expected_output/:/expected_output/ \
        biocontainers/simpleitk:v1.0.1-3-deb-py3_cv1 python3 -c """
import os
import SimpleITK as sitk

output_path = '/output/images/cow-${TASK}-segmentation/${IMAGE_FILENAME}'
print(f'{output_path} isfile? ', os.path.isfile(output_path))
expected_output_path = '/expected_output/${EXPECTED_SEG_MASK}'
print(f'{expected_output_path} isfile? ', os.path.isfile(expected_output_path))

output = sitk.ReadImage(output_path)
expected_output = sitk.ReadImage(expected_output_path) 

label_filter = sitk.LabelOverlapMeasuresImageFilter()
label_filter.Execute(output, expected_output)
dice_score = label_filter.GetDiceCoefficient()

print(f'dice_score = {dice_score}')

if dice_score == 1.0:
    print('[Success] Dice score=1, Test 0 passed!')
else:
    print('Dice score != 1, Test 0 failed!')
    print('[FAIL] Test 0 has FAILED!')
"""
echo "#################################################"
echo

echo "#################################################"
echo "##### Test 1 >>> /output/ folder check"
echo -e "\n$ ls -alR /output/ \n"

docker run --rm \
        -v ${OUTPUT_VOL}:/output/ \
        python:3.10-slim ls -alR /output/

echo "#################################################"
echo

echo "Please make sure you pass the above 2 tests before submitting your docker"

#docker volume rm ${OUTPUT_VOL}
