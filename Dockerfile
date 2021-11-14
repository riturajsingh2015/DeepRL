FROM jupyter/tensorflow-notebook

# Windows Render related
USER root
RUN apt-get update -y && \ 
    apt-get install -y xvfb && \
    apt-get install -y python-opengl 

# Optional, needed for some environments
RUN apt-get install -y cmake && \
    apt-get install -y zlib1g zlib1g-dev 

RUN apt-get install -y swig build-essential python-dev python3-dev

#USER ${NB_USER}

#RUN conda install swig

COPY --chown=${NB_UID}:${NB_GID} . /home/${NB_USER}/work
RUN pip install --quiet --no-cache-dir --requirement /home/${NB_USER}/work/requirements.txt && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

#COPY ./DQN_Cartpole.ipynb /home/${NB_USER}/work/DQN_Cartpole.ipynb
#COPY . /home/${NB_USER}
#WORKDIR /home/${NB_USER}/work

#RUN pip install -r /home/${NB_USER}/requirements.txt

# Reference image :- https://github.com/jianxu305/openai-gym-docker/blob/main/Dockerfile
