# OpenX Internship task

All tasks can be found in project be searching e.g. task 1

## API
1. Navigate to the InternshipTaskRestAPI directory.
2. Create a virtual environment using the command ```python3 -m venv venv```.
3. Activate the virtual environment.
4. Install all the required packages by running ```pip install -r requirements.txt```.
5. The models are already trained and saved in the file models.pickle. To create new models, run python manage.py train_models.
6. Run migrations using the commands ```python manage.py makemigrations``` and ```python manage.py migrate```.
8. Start the server using the command ```python manage.py runserver```.
9. Send a POST request to http://localhost:8000/api/predict with the required data in JSON format. An example body can be found in the InternshipTaskRestAPI directory.


# Task description

1. Load the Covertype Data Set https://archive.ics.uci.edu/ml/datasets/Covertype
2. Implement a very simple heuristic that will classify the data
    - It doesn't need to be accurate
3. Use Scikit-learn library to train two simple Machine Learning models
    - Choose models that will be useful as a baseline
4. Use TensorFlow library to train a neural network that will classify the data
    - Create a function that will find a good set of hyperparameters for the NN
    - Plot training curves for the best hyperparameters
5. Evaluate your neural network and other models
    - Choose appropriate plots and/or metrics to compare them
6. Create a very simple REST API that will serve your models
    - Allow users to choose a model
(heuristic, two other baseline models, or neural network)
    - Take all necessary input features and return a prediction
    - Do not host it anywhere, the code is enough
7. (Optional) Create a Docker container image to serve the API
Your solution should be clean, readable, modular, easy to run, and documented. Try to be
concise, do not add any extra functionalities.