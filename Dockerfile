FROM ubuntu:latest

RUN apt-get update
RUN apt-get install python3 python3-pip -y

RUN pip3 install flask
RUN pip3 install Flask connexion
RUN pip3 install scikit-learn
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install joblib
RUN pip3 install pandas
RUN pip3 install pycaret
