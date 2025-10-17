from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd


df = pd.read_csv("data.csv")

X = df[["House_size","Bedrooms","Bathrooms","Location","Furnishing"]]
y = df["House_Price"]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 100)

model = LinearRegression()

model.fit(X_train,y_train)

joblib.dump(model,"house_price_dummy.pkl")

print(model.predict(X_test))