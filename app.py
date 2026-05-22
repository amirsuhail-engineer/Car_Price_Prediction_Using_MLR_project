from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load Trained Model
model = pickle.load(open("car_price.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Get Form Data

        brand = int(request.form['brand'])

        year = int(request.form['year'])

        engine = float(request.form['engine'])

        fuel = int(request.form['fuel'])

        transmission = int(request.form['transmission'])

        mileage = int(request.form['mileage'])

        condition = int(request.form['condition'])

        model_name = int(request.form['model'])



        # Prediction Data (8 Features)

        features = np.array([[

            brand,
            year,
            engine,
            fuel,
            transmission,
            mileage,
            condition,
            model_name

        ]])



        # Prediction

        prediction = model.predict(features)[0]



        # Brand Names

        brand_names = {

            0: "Tesla",
            1: "BMW",
            2: "Audi",
            3: "Ford",
            4: "Honda",
            5: "Mercedes",
            6: "Toyota"

        }



        # Car Images

        car_images = {

            0: "Tesla-Motors-Model-S.jpg",
            1: "bmw-7-series-1.webp",
            2: "audi-a4-1.webp",
            3: "ford.avif",
            4: "honda.webp",
            5: "benz.avif",
            6: "toyota.jpg"

        }



        # Get Car Name

        car_name = brand_names.get(brand)



        # Get Car Image

        car_image = car_images.get(brand)



        return render_template(

            "index.html",

            prediction_text=f"Predicted Car Price : ₹ {round(prediction, 2)}",

            car_name=car_name,

            car_image=car_image

        )



    except Exception as e:



        return render_template(

            "index.html",

            prediction_text=f"Error : {e}",

            car_name="No Car",

            car_image="toyota.jpg"

        )



if __name__ == "__main__":

    app.run(debug=True)