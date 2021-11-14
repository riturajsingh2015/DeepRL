FROM jupyter/tensorflow-notebook

# Windows Render related
USER root
RUN apt-get update -y && \ 
    apt-get install -y xvfb && \
    apt-get install -y python-opengl && \
	apt-get install -y swig build-essential python-dev python3-dev && \
    apt-get install -y cmake && \
    apt-get install -y zlib1g zlib1g-dev 

#COPY . /opt/app
#WORKDIR /opt/app
#RUN pip install -r requirements.txt


USER ${NB_USER}

RUN rm -r /home/${NB_USER}/work
COPY . /home/${NB_USER}
WORKDIR /home/${NB_USER}


#RUN pip install \
#        gym \
#        pyvirtualdisplay 

# Needed for some environments
#RUN conda install swig
#RUN pip install pyglet box2d-py atari_py pystan 

RUN pip install -r requirements.txt

#ipywidgets==7.6.5 jupyter_contrib_nbextensions

#COPY ./First_Deep_Notebook.ipynb /home/${NB_USER}/First_Deep_Notebook.ipynb
#COPY ./requirements.txt /home/${NB_USER}/requirements.txt

#COPY ./temp /home/${NB_USER}/

#EXPOSE 8888