from pathlib import Path

import numpy as np
import pandas as pd
from parser.krisha_parser import clean_housing_data, generate_housing_data
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


def get_feature_columns(df):
    
    numeric_columns = df.select_dtypes(include=[np.number, bool]).columns.tolist()
    EXCLUDE_FROM_FEATURES = {'цена', 'цена_за_м2'}
    return [col for col in numeric_columns if col not in EXCLUDE_FROM_FEATURES]


def train_and_evaluate(df):
    target_column = df.columns[0]
    feature_columns = get_feature_columns(df)

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=70,
            max_depth=16,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.06,
            max_depth=3,
            random_state=42,
        ),
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        accuracy = max(0.0, 100 * (1 - mape))
        r2 = r2_score(y_test, predictions)

        results[name] = {"rmse": rmse, "mae": mae, "mape": mape, "accuracy": accuracy, "r2": r2}
        trained_models[name] = model

        print(
            f"{name}: RMSE={rmse:,.2f}, MAE={mae:,.2f}, "
            f"Accuracy~={accuracy:.2f}%, R2={r2:.3f}"
        )

    best_model_name = min(results, key=lambda name: results[name]["rmse"])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]

    print("\nModel comparison complete")
    print("--------------------------------")
    print(f"Best model: {best_model_name}")
    print(f"Best RMSE: {best_metrics['rmse']:,.2f}")
    print(f"Best MAE: {best_metrics['mae']:,.2f}")
    print(f"Best Accuracy~: {best_metrics['accuracy']:.2f}%")
    print(f"Best R2: {best_metrics['r2']:.3f}")

    return best_model_name, best_model, results, X_test, y_test


def predict_sample_price(model_name, model, X_test, y_test):
    sample_data = X_test.iloc[[0]]
    actual_price = y_test.iloc[0]
    predicted_price = model.predict(sample_data)[0]
    # attempt to extract readable fields
    def get_val(col_names):
        for name in col_names:
            if name in sample_data.columns:
                return sample_data.iloc[0][name]
        return None

    rooms = get_val(['комнаты'])
    area = get_val(['площадь'])
    floor = get_val(['этаж'])
    total_floors = get_val(['этажность'])
    year_built = get_val(['год_постройки'])

    # detect one-hot district and type
    district = None
    for col in sample_data.columns:
        if col.startswith('район_') and sample_data.iloc[0][col] == 1:
            district = col.split('район_', 1)[1]
            break

    house_type = None
    for col in sample_data.columns:
        if col.startswith('тип_дома_') and sample_data.iloc[0][col] == 1:
            house_type = col.split('тип_дома_', 1)[1]
            break

    complex_class = None
    for col in sample_data.columns:
        if col.startswith('класс_жк_') and sample_data.iloc[0][col] == 1:
            complex_class = col.split('класс_жк_', 1)[1]
            break

    print("\nSample prediction")
    print("--------------------------------")
    print(f"Model: {model_name}")
    print(f"Район: {district}")
    print(f"Тип дома: {house_type}")
    print(f"Класс ЖК: {complex_class}")
    print(f"Комнаты: {int(rooms) if rooms is not None else 'N/A'}")
    print(f"Площадь: {float(area) if area is not None else 'N/A'} кв. м")
    print(f"Этаж: {int(floor) if floor is not None else 'N/A'} / {int(total_floors) if total_floors is not None else 'N/A'}")
    print(f"Год постройки: {int(year_built) if year_built is not None else 'N/A'}")
    print(f"Predicted price: {predicted_price:,.0f} KZT")
    print(f"Actual price: {actual_price:,.0f} KZT")


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / "astana_housing.csv"
    cleaned_path = project_root / "astana_housing_cleaned.csv"

    df = generate_housing_data(output_path=str(raw_path), num_rows=3000, seed=42)
    cleaned_df = clean_housing_data(df)
    cleaned_df.to_csv(cleaned_path, index=False, encoding="utf-8")

    print(f"Cleaned data saved to {cleaned_path.name}")
    print("Training and evaluating models...")
    best_model_name, best_model, _, X_test, y_test = train_and_evaluate(cleaned_df)
    predict_sample_price(best_model_name, best_model, X_test, y_test)


if __name__ == "__main__":
    main()






