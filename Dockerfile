FROM continuumio/miniconda:latest

# Update dependencies
RUN apt-get -y update

# Install git
RUN apt-get -y install git

# Install cloc
RUN apt-get -y install cloc

# Install OpenJDK-11
RUN apt-get -y update && \
    mkdir /usr/share/man/man1/ && \
    apt-get install -y openjdk-11-jdk && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
RUN export JAVA_HOME

WORKDIR /home/td_classifier

COPY environment.yml ./
COPY lib ./lib
COPY models ./models
COPY td_classifier_service.py ./
COPY logic_complete_analysis.py ./
COPY logic_individual_analysis.py ./

RUN conda env create -f environment.yml
RUN echo "source activate td_classifier" > ~/.bashrc
ENV PATH /opt/conda/envs/td_classifier/bin:$PATH

EXPOSE 5005