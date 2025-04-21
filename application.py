import joblib
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask, render_template, request

app = Flask(__name__)
loaded_model = joblib.load(MODEL_OUTPUT_PATH)

@app.route('/', methods=['POST', 'GET'])
def index():
    prediction = None  # Initialize prediction outside the if block
    if request.method == 'POST':
        try:
            lead_time = int(request.form["lead_time"])
            no_of_special_requests = int(request.form["no_of_special_request"])
            avg_price_per_room = float(request.form["avg_price_per_room"])
            arrival_month = int(request.form["arrival_month"])
            arrival_date = int(request.form["arrival_date"])
            market_segment_type = int(request.form["market_segment_type"])
            no_of_week_nights = int(request.form["no_of_week_nights"])
            no_of_weekend_nights = int(request.form["no_of_weekend_nights"])
            type_of_meal_plan = int(request.form["type_of_meal_plan"])
            room_type_reserved = int(request.form["room_type_reserved"])

            features = np.array([[lead_time, no_of_special_requests, avg_price_per_room, arrival_month, arrival_date,
                                 market_segment_type, no_of_week_nights, no_of_weekend_nights, type_of_meal_plan,
                                 room_type_reserved]])

            prediction = loaded_model.predict(features)[0]
        except ValueError:
            prediction = "Invalid input. Please enter numeric values."
        except KeyError as e:
            prediction = f"Missing form data: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True) # Added debug=True for development