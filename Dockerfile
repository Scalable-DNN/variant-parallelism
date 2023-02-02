FROM tensorflow/tensorflow:2.8.0-gpu
# FROM tensorflow/tensorflow:2.3.0-gpu

# COPY training /vp/src

WORKDIR /opt/project
COPY src/requirements.txt .

# RUN apt install htop vim nano -y
# RUN apt update
# RUN apt-get update && apt-get -y install
# RUN apt install graphviz python-graphviz pango graphviz -y
# RUN apt install graphviz -y

# Install graphviz
RUN curl https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/7.1.0/graphviz-7.1.0.tar.gz -o graphviz-7.1.0.tar.gz
RUN tar -xf graphviz-7.1.0.tar.gz
WORKDIR graphviz-7.1.0
RUN ./configure
RUN make
RUN make install

WORKDIR /opt/project
RUN pip install --upgrade pip
RUN pip install tensorflow-gpu==2.8
RUN pip install -r requirements.txt
# RUN pip3 install backcall==0.2.0 colorama==0.4.4 cycler==0.10.0 decorator==5.0.7 deprecation==2.1.0 graphviz==0.16 intervaltree==3.1.0 ipython==7.22.0 ipython-genutils==0.2.0 jedi==0.18.0 jinja2==3.0.0a1 joblib==1.0.1 jsonpickle==2.0.0 kiwisolver==1.3.1 lxml==4.6.3 MarkupSafe==2.0.0rc1 matplotlib==3.4.1 mpmath==1.2.1 networkx==2.5.1 numpy==1.20.2 packaging==20.9 pandas==1.2.4 parso==0.8.2 pickleshare==0.7.5 pillow==8.2.0 prompt-toolkit==3.0.18 pulp==2.1 pydotplus==2.0.2 pygments==2.8.1 pyparsing==3.0.0b2 python-dateutil==2.8.1 pytz==2021.1 pyvis==0.1.9 scikit-learn==0.24.1 scipy==1.6.2 setuptools==56.0.0 six==1.15.0 sortedcontainers==2.3.0 stringdist==1.0.9 sympy==1.8 threadpoolctl==2.1.0 tqdm==4.60.0 traitlets==5.0.5 wcwidth==0.2.5

# RUN groupadd -g 1015 navid && useradd -r -g navid navid -u 1014
# USER 1014:1015
# USER 197609:197121
ENV HOME /home/navid_cont
ENV WD /opt/project
WORKDIR /home/navid_cont