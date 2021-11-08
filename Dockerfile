FROM deepnote/python:3.7

# Windows Render related
RUN apt update -y && \ 
    apt install -y xvfb && \
    apt install -y python-opengl 

# Optional, needed for some environments
RUN apt install -y cmake && \
    apt install -y zlib1g zlib1g-dev 


RUN pip install \
        gym \
        pyvirtualdisplay \
        pyglet 

# Needed for some environments
# RUN conda install swig
RUN pip install box2d-py atari_py pystan