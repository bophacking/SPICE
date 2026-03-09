from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and columns
with open("price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# -----------------------------
# Helper functions (post-model)
# -----------------------------

def price_level(price):
    if price < 9:
        return "Cheap"
    elif price < 14:
        return "Average"
    else:
        return "Expensive"


def estimate_profit(predicted_price, ingredient_cost):
    if pd.isna(ingredient_cost):
        return None
    return predicted_price - ingredient_cost

@app.route("/", methods=["GET", "POST"])
def index():
    table = None

    output_columns = [
        "menu_item_name",
        "predicted_selling_price",
        "price_level",
        "profit_estimation"
    ]

    if request.method == "POST":
        file = request.files.get("file")
        
        if file:
            df = pd.read_csv(file)

            # Drop unused columns safely
            df_input = df.drop(
                columns=[
                    "menu_item_name",
                    "key_ingredients_tags",
                    "date",
                    "actual_selling_price",
                    "restaurant_id"
                ],
                errors="ignore"
            )

            # One-hot encode
            df_encoded = pd.get_dummies(df_input)

            # Align columns with training data
            df_encoded = df_encoded.reindex(
                columns=model_columns,
                fill_value=0
            )

            # Predict
            df["predicted_selling_price"] = model.predict(df_encoded)
            
            df["price_level"] = df["predicted_selling_price"].apply(price_level)

            if "typical_ingredient_cost" in df.columns:
                df["profit_estimation"] = df.apply(
                    lambda row: estimate_profit(
                        row["predicted_selling_price"],
                        row["typical_ingredient_cost"]
                    ),
                    axis=1
                )
            else:
                df["profit_estimation"] = None

            df_output = df[output_columns]

            # Convert to HTML table
            table = df_output.head(20).to_html(
                classes="table table-striped",
                index=False
            )

    return render_template("index.html", table=table)


if __name__ == "__main__":
    app.run(debug=True)
