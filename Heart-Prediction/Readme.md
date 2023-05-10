## Heart-Disease-Prediction 

This Project is mainly divided into two parts:
 
 1. Exploring the dataset and traning the model using `Sklearn`.
 2. Building and hosting a `flask` 

**About the repository Structure :**

- `app.py` script which is used to run the application and is engine of this app. It contians API that gets input from the user and computes a predicted value based on the model.
- `model.py` contains code to build and train a Machine learning model.
- *templates* folder contains one file `index.html` It contains the frontend of the application. it is connected to Flask for python.
- *static* folder contains file `style.css` for styling. 

### Installation
If you have never ran a machine learning model on you machine then it is recommended to install these dependencies. 
open cmd as admin user

```
pip install numpy
```
```
pip install pandas
```
```
pip install -U scikit-learn
```
```
pip install Flask
```

### Run 

*To Run the Application*
- first run `model.py` then a pickel file will be created for your model.

```
python model.py
```
- then run `app.py` to deploy you model in the web 
```
python app.py
```
- in terminal a address will be generated like `http://127.0.0.1:5000` open it to deploy









  
  
  


