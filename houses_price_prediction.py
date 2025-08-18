# Importing Modules
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# Loading dataset
house_prices_df = pd.read_csv("data/lahore_housing_prices.csv")
# Creating separate columns for area, city and province
house_prices_df[['AreaName', 'City', 'Province']] = house_prices_df['Location'].str.split(",", expand=True)
house_prices_df['AreaName'] = house_prices_df['AreaName'].str.strip()
house_prices_df['City'] = house_prices_df['City'].str.strip()
house_prices_df['Province'] = house_prices_df['Province'].str.strip()
# Converting Kanal/Marla Area to Sqft
def area_to_sqft(area_str):
    area_str = str(area_str).lower().strip()
    if "kanal" in area_str:
        num = float(area_str.replace("kanal", "").strip())
        return num * 5445
    elif "marla" in area_str:
        num = float(area_str.replace("marla", "").strip())
        return num * 272.25
    else:
        return None

house_prices_df["Area_Sqft"] = house_prices_df["Area"].map(area_to_sqft)
# Using binary codes for houses and flats
type_code = {"House": 1, "Flat": 0}
house_prices_df["type_code"] = house_prices_df['Type'].map(type_code)
# Setting the inputs for training
inputs = house_prices_df[['type_code', 'Area_Sqft', 'Bath(s)', 'Bedroom(s)', 'AreaName', 'City', 'Province']]
targets = house_prices_df['Price']
# Dropping NaN values
inputs = inputs.dropna()
targets = targets[inputs.index]
# Categorical and numeric columns
categorical_cols = ['AreaName', 'City', 'Province']
numeric_cols = ['Area_Sqft', 'type_code', 'Bath(s)', 'Bedroom(s)']
# Preprocessor for encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)
# Pipeline model(preprocessor+model)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
# Predicting function
def predicting_price():
    print("********** Enter data to predict house prices (Lahore, Punjab) **********")
    area_sqft = float(input("Area (Sqft): "))
    type_codes = int(input("1 for House and 0 for Flat: "))
    baths = int(input("No. of Baths: "))
    bedrooms = int(input("No. of Bedrooms: "))
    area = input("Area name: ")

    new_house = pd.DataFrame([{
        'Area_Sqft': area_sqft,
        'type_code': type_codes,
        'Bath(s)': baths,
        'Bedroom(s)': bedrooms,
        'AreaName': area,
        'City': 'Lahore',
        'Province': 'Punjab'
    }])

    predicted_price = model.predict(new_house)[0]
    print(f'🏠 Predicted House Price: {predicted_price:,.0f} PKR')

if __name__ == "__main__":
    while True:
        predicting_price()
        check = input("Do you want to predict more? (yes/no): ").lower()
        if check == 'no':
            break
