## Description 
 In this project, I create a machine learning model from the [sepsis dataset](https://www.kaggle.com/datasets/chaunguynnghunh/sepsis?select=README.md). Also, I will create an API that will integrate the ML model using FastAPI. The App will interact with the model to classify the sepsis illness. Finally, I will containerize my application with a docker.


[![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
[![MIT licensed](https://img.shields.io/badge/license-mit-blue?style=for-the-badge&logo=appveyor)](./LICENSE)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)


| Code      | Name        | Published Article |  Deployed App |
|-----------|-------------|:-------------|:------|
| P 6     | Embedding ML model into web application| Medium<br />LinkedIn| |

## Note on Data
This is a summary of the information contained in the data<br />
`ID`: number to represent patient ID<br />
`PRG`: Plasma glucose<br />
`PL`: Blood Work Result-1 (mu U/ml)<br />
`PR`: Blood Pressure (mm Hg)<br />
`SK`: Blood Work Result-2 (mm)<br />
`TS`: Blood Work Result-3 (mu U/ml)<br />
`M11`: Body mass index (weight in kg/(height in m)^2<br />
`BD2`: Blood Work Result-4 (mu U/ml)<br />
`Age`: patients age (years)<br />
`Insurance`: If a patient holds a valid insurance card<br />
`Sepsis`: **Positive**: if a patient in ICU will develop a sepsis , and **Negative**: otherwis<br />


## Screenshot of Deployed App




## Setup
Install the required packages to be able to run the evaluation locally.

You need to have [`Python3`](https://www.python.org/) on your system. Then you can clone this repo and being at the repo's root (`root :: repo_name> ...`)  follow the steps below:

- Windows:
        
        python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

- Linux & MacOs:
        
        python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

The both long command-lines have a same structure, they pipe multiple commands using the symbol **;** but you may manually execute them one after another.

1. **Create the Python's virtual environment** that isolates the required libraries of the project to avoid conflicts;
2. **Activate the Python's virtual environment** so that the Python kernel & libraries will be those of the isolated environment;
3. **Upgrade Pip, the installed libraries/packages manager** to have the up-to-date version that will work correctly;
4. **Install the required libraries/packages** listed in the `requirements.txt` file so that it will be allow to import them into the python's scripts and notebooks without any issue.

**NB:** For MacOs users, please install `Xcode` if you have an issue.


## Run FastAPI

- Run the demo apps (being at the repository root):
        
  FastAPI:
    
    - Demo

          uvicorn src.demo_01.api:app --reload 

    <!-- - Salary prediction

          uvicorn src.salary.api:app --reload  -->


  - Go to your browser at the following address, to explore the api's documentation :
        
      http://127.0.0.1:8000/docs


## Resources
Here are some ressources you would read to have a good understanding of FastAPI :
- [Tutorial - User Guide](https://fastapi.tiangolo.com/tutorial/)
- [Video - Building a Machine Learning API in 15 Minutes ](https://youtu.be/C82lT9cWQiA)
- [FastAPI for Machine Learning: Live coding an ML web application](https://www.youtube.com/watch?v=_BZGtifh_gw)
- [Video - Deploy ML models with FastAPI, Docker, and Heroku ](https://www.youtube.com/watch?v=h5wLuVDr0oc)
- [FastAPI Tutorial Series](https://www.youtube.com/watch?v=tKL6wEqbyNs&list=PLShTCj6cbon9gK9AbDSxZbas1F6b6C_Mx)
- [Http status codes](https://www.linkedin.com/feed/update/urn:li:activity:7017027658400063488?utm_source=share&utm_medium=member_desktop)



## Author
[Elvis Darko](https://www.linkedin.com/in/elvis-darko/)