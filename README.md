# TD Classifier

## Description

This repository contains the source code of the **TD Classifier back-end** ([video](https://youtu.be/Cnt3cGb4dkE), [running instance](http://160.40.52.130:3000/tdclassifier)). 

TD Classifier is a novel tool that employs Machine Learning (ML) for classifying software classes as High/Not-High TD for any arbitrary Java project, just by pointing to its git repository. As ground truth for the development of the tool's classification framework, we considered a *"commonly agreed TD knowledge base"* ([Amanatidis et al., 2020](https://link.springer.com/article/10.1007/s10664-020-09869-w)), i.e., an empirical benchmark of classes that exhibit high levels of TD, based on the convergence of three widely-adopted TD assessment tools, namely [SonarQube](https://www.sonarqube.org/), [CAST](https://www.castsoftware.com/products/code-analysis-tools), and [Squore](https://www.vector.com/no/en/products/products-a-z/software/squore/squore-software-analytics-for-project-monitoring/). Then, we built a set of independent variables based on a wide range of software metrics spanning from code metrics to repository activity, retrieved by employing four popular open source tools, namely [PyDriller](https://github.com/ishepard/pydriller), [CK](https://github.com/mauricioaniche/ck), [PMD’s Copy/Paste Detector](https://pmd.github.io/latest/pmd_userdocs_cpd.html) , and [cloc](https://github.com/AlDanial/cloc#quick-start-). Therefore, the tool subsumes the collective knowledge that would be extracted by combining the results of the three TD assessment tools and relies on four open-source tools to automatically retrieve all independent variables and yield the identified high-TD classes. In that way, it enables identification and further experimentation of high-TD modules, without having to resort to a multitude of commercial and open source tools.

**Note**: the source code of the **TD Classifier frontend** is also publicly available as part of the overall SDK4ED Dashboard and can be found [here](https://gitlab.seis.iti.gr/sdk4ed/sdk4ed-dashboard).

## Installation

### Installation using Anaconda

In this section, we provide instructions on how the user can build the python Flask server of the TD Classifier from scratch, using the Anaconda virtual environment. The TD Classifier is developed to run on Unix and Windows systems with python 3.6.* innstalled. We suggest installing python via the Anaconda distribution as it provides an easy way to create a virtual environment and install dependencies. The configuration steps needed, are described below:

- **Step 1**: Download the latest [Anaconda distribution](https://www.anaconda.com/distribution/) and follow the installation steps described in the [Anaconda documentation](https://docs.anaconda.com/anaconda/install/windows/).
- **Step 2**: Open Anaconda cmd. Running Anaconda cmd activates the base environment. We need to create a specific environment to run TD Classifier backend. Create a new python 3.6.4 environment by running the following command:
```bash
conda create --name td_classifier python=3.6.4
```
This command will result in the creation of a conda environment named *td_classifier*. In order to activate the new environment, execute the following command:
```bash
conda activate td_classifier
```
- **Step 3**: Once your newly created environment is active, install the needed libraries by executing the following commands:
```bash
conda install -c anaconda numpy pandas scikit-learn waitress flask flask-cors requests pymongo
```
```bash
conda install -c conda-forge gitpython
```
and
```bash
pip install pydriller
```
- **Step 4**: To start the server, use the command promt inside the active environment and execute the commands described in section **Run Server**.

### Installation using Docker (recommended)

In this section, we provide instructions on how the user can build a new Docker Image that contains the python Flask app and the Conda environment of the of the TD Classifier. We highly recommend the users to select this way of installing the TD Classifier, as it constitutes the easiest way.

- **Step 1**: Download and install [Docker](https://www.docker.com/)
- **Step 2**: Clone the latest TD Classifier version and navigate to the home directory. You should see a [DockerFile](/blob/master/Dockerfile) and a [environment.yml](./blob/master/environment.yml) file, which contains the Conda environment dependencies. 
- **Step 3**: In the home directory of the TD Classifier, open cmd and execute the following command:
```bash
sudo docker build -t td_classifier_image .
``` 
This command will result in the creation of a Docker Image named *td_classifier_image*. In order to create a Docker Container from this image, execute the following command:
```bash
sudo docker run -it --name td_classifier_test -p 5005:5005 td_classifier_image
``` 
This command will generate and run a Docker Container named *td_classifier_test* in interactive session mode, i.e. it will open a command promt inside the Container. 
- **Step 4**: To start the server, use the command promt inside the running Container and execute the commands described in section **Run Server**.

### Installation of the Database

A MongoDB database dedicated to storing the output of the TD Classifier web service allows the tool to quickly retrieve past results upon demand, without having to go through the time-consuming process of re-executing the analysis process. TD Classifier does not require a running database instance to be functional. However, in case you require access to previously produced results, a database dedicated to store the output of the TD Classifier web services might be of help. In that case, MongoDB is a well-suited option for the purposes of the TD Classifier.

To quickly install a MongoDB using Docker, open cmd and execute the following command:
```bash
sudo docker run --detach  \
  -p 27017:27017  \
  --name mongodb  \
  --volume /home/<user_name>/Desktop/mongo_data:/data/db  \
  mongo
```
This command will generate and run a MongoDB Docker Container named *mongodb*, which will serve as the TD Classifier's dedicated DB.

## Run Server

You can run the server in various modes using Python to run the `td_classifier_service.py` script:

```
usage: td_classifier_service.py [-h] [-dh DB_HOST] [-dp DB_PORT] [-dn DB_DBNAME]
                                [--debug]
                                HOST PORT SERVER_MODE

positional arguments:
  HOST           Server HOST (e.g. "localhost")
  PORT           Server PORT (e.g. "5005")
  SERVER_MODE    builtin, waitress

optional arguments:
  -h, --help     show this help message and exit
  -dh DB_HOST    MongoDB HOST (e.g. "localhost") (default: localhost)
  -dp DB_PORT    MongoDB PORT (e.g. "27017") (default: 27017)
  -dn DB_DBNAME  Database NAME (default: td_classifier_service)
  --debug        Run builtin server in debug mode (default: False)
```

`HOST`, `PORT`, and `SERVER_MODE` arguments are **mandatory**. You can set them according to your needs.

`DB_HOST`, `DB_PORT`, and `DB_DBNAME` arguments are **optional** and assume that there is a MongoDB instance running either on a local machine or remotely. In case that there is no such MongoDB instance running, the TD Classifier will still return the results, but they will not be stored anywhere.

### Run built-in Flask server

```
         127.0.0.1:5005
Client <----------------> Flask
```

To start the TD Classifier using the built-in **Flask** server, use the command promt inside the active Conda or Container environment and execute the following command: 

```bash
python td_classifier_service.py 0.0.0.0 5005 builtin --debug
```

This command will start the built-in Flask server locally (0.0.0.0) on port 5005.

**MongoDB Integration**

In case there is a MongoDB instance running, use the command promt inside the active conda or Container environment and execute the following command: 

```bash
python td_classifier_service.py 0.0.0.0 5005 waitress -dh 160.40.52.130 -dp 27017 -dn td_classifier_service
```

This command will start the built-in Flask server locally on port 5005 and store the results on a MongoDB database named "td_classifier_service" running locally on port 27017.

**Warning**: The built-in Flask mode is useful for development since it has debugging enabled (e.g. in case of error the client gets a full stack trace). However, it is single-threaded. Do NOT use this mode in production!

### Run Waitress server

```
         127.0.0.1:5005
Client <----------------> Waitress <---> Flask
```

To start the TD Classifier using the **Waitress** server, use the command promt inside the active Conda or Container environment and execute the following command:

```bash
python td_classifier_service.py 0.0.0.0 5005 waitress
```

This command will start the Waitress server locally (0.0.0.0) on port 5005.

**MongoDB Integration**

In case there is a MongoDB instance running, use the command promt inside the active conda or Container environment and execute the following command: 

```bash
python td_classifier_service.py 0.0.0.0 5005 waitress -dh 160.40.52.130 -dp 27017 -dn td_classifier_service
```

This command will start the Waitress server locally on port 5000 and store the results on a MongoDB database named "td_classifier_service" running locally on port 27017.

**Warning**: The Waitress mode is higly recommended in real production environments, since it supports scaling and multiple-request handling features.


## Next Steps

Once the server is up and running, you can either exploit the TD Classifier APIs directly via requests, or setup the SDK4ED platform locally (see [here](https://gitlab.seis.iti.gr/sdk4ed-wiki/wiki-home/wikis/home)) to benefit from the already implemented and user-friendly TD Classifier frontend.