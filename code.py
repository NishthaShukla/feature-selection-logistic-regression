# --------------
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Load the data
df = pd.read_csv(path)
#Loading the Spam data from the path variable for the mini challenge

#Target variable is the 57 column i.e spam, non-spam classes 
X = df.iloc[:,:57]
y = df.iloc[:,57]
# Overview of the data
print(df.info())
print(df.describe())


#Dividing the dataset set in train and test set and apply base logistic model
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

# Calculate accuracy , print out the Classification report and Confusion Matrix.
accuracy = accuracy_score(y_test,y_pred)
cf = confusion_matrix(y_test, y_pred)

# Copy df in new variable df1
df1 = df.copy()

corr = df1.corr()
# Remove Correlated features above 0.75 and then apply logistic model
print(corr)

# Split the new subset of data and fit the logistic model on training data


# Calculate accuracy , print out the Classification report and Confusion Matrix for new data


# Apply Chi Square and fit the logistic model on train data use df dataset



# Calculate accuracy , print out the Confusion Matrix 


# Apply Anova and fit the logistic model on train data use df dataset



# Calculate accuracy , print out the Confusion Matrix 


# Apply PCA and fit the logistic model on train data use df dataset

   

# Calculate accuracy , print out the Confusion Matrix 


# Compare observed value and Predicted value




