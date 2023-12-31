## Project Description 
![sepsis](https://scirc.med.ufl.edu/files/2019/11/GettyImages-1076118668-600x400.jpg)
 
 Sepsis, a life-threatening condition arising from infection, poses a significant global healthcare challenge. To better comprehend sepsis occurrence, researchers are turning to patient data analysis for uncovering hidden patterns and predictors. 
 
 By harnessing advanced data analytics techniques and exploring diverse parameters such as vital signs, medical history, and demographic information, this project aims to identify early warning signs and risk factors for sepsis development. This knowledge holds immense potential for developing risk stratification models, early detection systems, and targeted interventions, ultimately leading to improved patient outcomes and optimized sepsis management protocols.

 The focus lies on creating a robust system that can accurately detect and classify sepsis cases, enabling healthcare providers to respond promptly and effectively to this life-threatening condition.
 

 ## Project Objectives
 - Create a machine learning model from the [sepsis dataset](https://www.kaggle.com/datasets/chaunguynnghunh/sepsis?select=README.md) that predicts whether a pateint has sepssis or not.  

 - Implement the FastAPI framework to create a user-friendly and efficient web interface for healthcare professionals to interact with the sepsis classification model.

 - Containerize the app with a docker


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
`M11`: Body mass index (weight in kg/(height in m)^2)<br />
`BD2`: Blood Work Result-4 (mu U/ml)<br />
`Age`: patients age (years)<br />
`Insurance`: If a patient holds a valid insurance card. **0**: Patient without insurance, and **1**: Patient with insurance<br />
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

          uvicorn src.demo_01.api:main --reload 



  - Go to your browser at the following address, to explore the api's documentation :
        
      http://127.0.0.1:8000/docs

## Run Streamlit App
A streamlit app was added for further exploration of the model. The streamlit app provides a simple Graphic User Interface where predicitons can be made from inputs.

- Run the demo app (being at the root of the repository):
        
        Streamlit run st_app.py


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