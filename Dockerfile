# Edit the base image here, e.g. to use
# Tensorflow: https://hub.docker.com/r/tensorflow/tensorflow/ 
# Pytorch: https://hub.docker.com/r/pytorch/pytorch/
# For Pytorch e.g.: FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
# Or list your pip packages in requirements.txt and install them
# FROM python:3.10-slim
#FROM python:3.10-slim
#FROM pytorch/pytorch:2.0.1-cuda10.1-cudnn7-runtime
#FROM original:latest
FROM cowsegmentation_new:latest

#CMD [ "bet"]

# Create the user
#RUN groupadd -r user && useradd -m --no-log-init -r -g user user

#RUN mkdir -p /opt/app /input /output \
#    && chown user:user /opt/app /input /output



#USER user
#WORKDIR /opt/app

#ENV FSLDIR="/opt/app/fsl_bet"
#ENV ANTSPATH="/opt/app/ants-2.4.1/bin"
#ENV ANTSPATH="/opt/app/ants"
#ENV PATH="/home/user/.local/bin:${PATH}"
#ENV PATH="/opt/app/fsl_bet/bin:/opt/app/ants-2.4.1/bin:/home/user/.local/bin:${PATH}"
#ENV PATH="/opt/app/fsl_bet/bin:/opt/app/ants:/home/user/.local/bin:${PATH}"

#RUN python -m pip install --user -U pip
#RUN rm /opt/app/process.py
#RUN rm -rf /opt/app/nnUNet
#RUN rm /opt/app/setup.py
#RUN rm /opt/app/setup.cfg

#RUN rm /home/user/.local/bin/antsApplyTransforms
#RUN rm /home/user/.local/bin/antsRegistrationSyNQuick.sh
#RUN rm /home/user/.local/bin/bet

#RUN rm -rf /opt/app/fsl_bet
#RUN rm -rf /opt/app/ants-2.4.1


# All required files are copied to the Docker container
#RUN mkdir /opt/app/fsl_bet
#RUN mkdir /opt/app/ants

#COPY ants-2.4.1 /opt/app/ants-2.4.1
#COPY ants /opt/app/ants
#COPY fsl_bet /opt/app/fsl_bet
#COPY bet /home/user/.local/bin/
#COPY antsApplyTransforms /opt/app/ants-2.4.1/bin/
#COPY antsRegistrationSyNQuick.sh /opt/app/ants/
#COPY PrintHeader /opt/app/ants-2.4.1/bin/
#COPY antsRegistration /opt/app/ants-2.4.1/bin/

#USER root

#RUN chmod 777 /opt/app/ants-2.4.1/bin/antsRegistration
#RUN chmod 777 /opt/app/ants-2.4.1/bin/antsApplyTransforms
#RUN chmod 777 /opt/app/ants/antsRegistrationSyNQuick.sh
#USER user



                        
#RUN mkdir /opt/app/template
#RUN mkdir /opt/app/module

#RUN python -m pip uninstall -y nnunetv2
#RUN rm -rf /opt/app/nnunetv2



#RUN rm /opt/app/setup.py
#RUN rm /opt/app/setup.cfg



#COPY --chown=user:user requirements.txt /opt/app/
RUN rm /opt/app/process.py
COPY --chown=user:user process.py /opt/app/
#RUN rm /home/user/.local/bin/antsRegistrationSyNQuick.sh
#RUN rm /home/user/.local/bin/antsRegistration
#RUN rm /home/user/.local/bin/antsApplyTransforms
#RUN rm -rf /opt/app/nnUNet_results
#RUN rm /opt/app/nnunetv2/inference/predict_from_raw_data.py
#COPY --chown=user:user predict_from_raw_data.py /opt/app/nnunetv2/inference/
#RUN rm /opt/app/nnunetv2/preprocessing/resampling/default_resampling.py
#COPY --chown=user:user default_resampling.py /opt/app/nnunetv2/preprocessing/resampling/
#RUN rm /opt/app/nnunetv2/inference/export_prediction.py
#COPY --chown=user:user export_prediction.py /opt/app/nnunetv2/inference/
#COPY --chown=user:user base_algorithm.py /opt/app/
#COPY --chown=user:user Postprocessing.py /opt/app/
#COPY --chown=user:user Preprocessing.py /opt/app/
#COPY --chown=user:user utils_sitk.py /opt/app/
#COPY --chown=user:user template /opt/app/template


#RUN rm /opt/app/module/nnUNet_results/Dataset055_ROIAugmentation/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_4/checkpoint_best.pth
#COPY --chown=user:user checkpoint_best.pth /opt/app/module/nnUNet_results/Dataset055_ROIAugmentation/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_4
#COPY --chown=user:user module /opt/app/module
#COPY --chown=user:user setup.cfg /opt/app/
#COPY --chown=user:user setup.py /opt/app/
#COPY --chown=user:user imageio-2.31.3.tar.gz /opt/app/
#RUN rm -rf /opt/app/nnunet
#COPY --chown=user:user nnunet /opt/app
#RUN rm -rf /opt/app/nnunetv2

#COPY --chown=user:user file_path_utilities.py /opt/app/
#COPY --chown=user:user find_class_by_name.py /opt/app/
#COPY --chown=user:user get_network_from_plans.py /opt/app/
#COPY --chown=user:user helpers.py /opt/app/
#COPY --chown=user:user collate_outputs.py /opt/app/
#COPY --chown=user:user configuration.py /opt/app/
#COPY --chown=user:user cropping.py /opt/app/
#COPY --chown=user:user dataset_name_id_conversion.py /opt/app/
#COPY --chown=user:user ddp_allgather.py /opt/app/
#COPY --chown=user:user default_n_proc_DA.py /opt/app/
#COPY --chown=user:user default_preprocessor.py /opt/app/
#COPY --chown=user:user default_resampling.py /opt/app/
#COPY --chown=user:user export_prediction.py /opt/app/
#COPY --chown=user:user json_export.py /opt/app/
#COPY --chown=user:user label_handler.py /opt/app/
#COPY --chown=user:user network_initialization.py /opt/app/
#COPY --chown=user:user paths.py /opt/app/
#COPY --chown=user:user plans_handler.py /opt/app/
#COPY --chown=user:user predict_from_raw_data.py /opt/app/
#COPY --chown=user:user sliding_window_prediction.py /opt/app/
#COPY --chown=user:user tensor_utilities.py /opt/app/
#COPY --chown=user:user utils.py /opt/app/

#RUN rm label_handling.py

#RUN python -m pip install --user setuptools
#COPY --chown=user:user setup.py /opt/app/
#RUN mkdir /opt/app/nnunetv2
#COPY --chown=user:user nnunetv2 /opt/app/nnunetv2
#COPY --chown=user:user pyproject.toml /opt/app/
#COPY --chown=user:user readme.md /opt/app/
#COPY --chown=user:user LICENSE /opt/app/
#RUN python -m pip install --user -e .




# COPY --chown=user:user <somefile> /opt/app/
# ...
# Install required python packages via pip from your requirements.txt
#RUN rm -rf /opt/app/imageio
#RUN python -m pip install --user -r requirements.txt
#RUN python -m pip install --user imageio-2.31.3.tar.gz

#RUN rm -rf /opt/app/imageio
#COPY --chown=user:user imageio /opt/app
#RUN python -m pip install --user imageio-2.31.3.tar.gz
#RUN python -m pip install --user -e .
#RUN rm imageio-2.31.3.tar.gz



#CMD [ "bet"]
# Entrypoint to your python code - executes process.py as a script
# ENTRYPOINT [ "python", "-m", "process" ]
