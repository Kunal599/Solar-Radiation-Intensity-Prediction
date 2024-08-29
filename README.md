# Solar-Radiation-Intensity-Prediction

This project aims to predict solar irradiance using machine learning techniques including XGBoost and a Multi-Layer Perceptron (MLP). The dataset used is Data_SolarPrediction.csv.

Features
Data Preprocessing: Extract and transform date-time features.
Feature Selection: Identify key features using SelectKBest.
Feature Engineering: Apply various transformations to improve model performance.
Model Training: Train XGBoost and MLP models.
Evaluation: Assess model performance with metrics like RMSE and R2.
Dependencies
xgboost
numpy
pandas
matplotlib
seaborn
sklearn
tensorflow
Installation
Install required packages:

bash
Copy code
pip install xgboost tensorflow
Usage
Load Data:

python
Copy code
data = pd.read_csv("Data_SolarPrediction.csv")
Preprocess Data: Extract features and apply transformations.

Train Models:

python
Copy code
# XGBoost
model = XGBRegressor(learning_rate=0.1, max_depth=8)
model.fit(xtrain, ytrain)
python
Copy code
# MLP
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=14))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
model.fit(xtrain, ytrain, epochs=50, batch_size=32)
Evaluate Models:

python
Copy code
y_pred = model.predict(xtest)
rmse = np.sqrt(mean_squared_error(ytest, y_pred))
print("RMSE: {:.2f}".format(rmse))
Author
Kunal Harinkhede

