# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import os
from datetime import datetime


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "student_mark_predictor.pkl"
)


# ============================================================
# CSV PATH
# ============================================================

CSV_PATH = os.path.join(
    BASE_DIR,
    "smp_data_from_app.csv"
)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

try:

    model = joblib.load(
        MODEL_PATH
    )

    print("\n")
    print("=" * 75)
    print("        STUDENT MARK PREDICTOR - ML ENGINE")
    print("=" * 75)
    print("[OK] Machine Learning model loaded successfully")
    print("[OK] Model file:",
          MODEL_PATH)
    print("=" * 75)
    print()

except Exception as error:

    print("\n")
    print("=" * 75)
    print("[ERROR] Failed to load ML model")
    print("[ERROR]", error)
    print("=" * 75)
    print()

    model = None


# ============================================================
# LOAD EXISTING CSV DATA
# ============================================================

if os.path.exists(CSV_PATH):

    try:

        df = pd.read_csv(
            CSV_PATH
        )

        if "Unnamed: 0" in df.columns:

            df = df.drop(
                columns=["Unnamed: 0"]
            )

        print(
            "[OK] Existing prediction history loaded."
        )

    except Exception as error:

        print(
            "[WARNING] Could not load CSV:",
            error
        )

        df = pd.DataFrame(
            columns=[
                "Study Hours",
                "Predicted Output"
            ]
        )

else:

    df = pd.DataFrame(
        columns=[
            "Study Hours",
            "Predicted Output"
        ]
    )

    print(
        "[INFO] No previous prediction history found."
    )


# ============================================================
# TERMINAL TABLE FUNCTION
# ============================================================

def print_prediction_table(
    study_hours,
    prediction,
    raw_prediction=None,
    request_type="PREDICTION"
):

    """
    Display prediction information
    in a clean terminal table.
    """

    current_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    print("\n")

    print(
        "=" * 82
    )

    print(
        "                 ML PREDICTION RESULT"
    )

    print(
        "=" * 82
    )


    # Header

    print(
        f"| {'TIME':<19} "
        f"| {'REQUEST':<12} "
        f"| {'INPUT: HOURS':<15} "
        f"| {'OUTPUT: MARKS':<15} |"
    )


    print(
        "-" * 82
    )


    # Data

    print(
        f"| {current_time:<19} "
        f"| {request_type:<12} "
        f"| {study_hours:<15.2f} "
        f"| {prediction:<15.2f}% |"
    )


    print(
        "=" * 82
    )


    if raw_prediction is not None:

        print(
            f"Raw ML Model Output : "
            f"{raw_prediction:.2f}"
        )


    print(
        f"Final Prediction    : "
        f"{prediction:.2f}%"
    )


    print(
        f"Input Study Hours   : "
        f"{study_hours:.2f} hours/day"
    )


    print(
        "=" * 82
    )

    print()


# ============================================================
# COMPLETE HISTORY TABLE
# ============================================================

def print_history_table():

    global df


    print("\n")

    print(
        "=" * 75
    )

    print(
        "                 PREDICTION HISTORY"
    )

    print(
        "=" * 75
    )


    if df.empty:

        print(
            "| No prediction records available. |"
        )

        print(
            "=" * 75
        )

        return


    print(
        f"| {'#':<5} "
        f"| {'Study Hours':<20} "
        f"| {'Predicted Marks':<20} |"
    )


    print(
        "-" * 75
    )


    for index, row in df.iterrows():

        hours = float(
            row["Study Hours"]
        )

        marks = float(
            row["Predicted Output"]
        )


        print(
            f"| {index + 1:<5} "
            f"| {hours:<20.2f} "
            f"| {marks:<19.2f}% |"
        )


    print(
        "=" * 75
    )


    print(
        f"Total Predictions: {len(df)}"
    )


    print()


# ============================================================
# CALCULATE PREDICTION
# ============================================================

def calculate_prediction(
    study_hours
):


    if model is None:

        raise RuntimeError(
            "Machine Learning model is not loaded."
        )


    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    features_value = np.array(
        [study_hours],
        dtype=float
    )


    # --------------------------------------------------------
    # ML MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        [features_value]
    )


    raw_prediction = float(
        np.asarray(
            prediction
        ).flatten()[0]
    )


    # --------------------------------------------------------
    # REFERENCE PREDICTION AT 23 HOURS
    # --------------------------------------------------------

    reference_features = np.array(
        [23.0],
        dtype=float
    )


    reference_prediction = model.predict(
        [reference_features]
    )


    reference_output = float(
        np.asarray(
            reference_prediction
        ).flatten()[0]
    )


    # --------------------------------------------------------
    # NORMALIZE TO 100%
    # --------------------------------------------------------

    if study_hours >= 23:

        output = 100.0

    else:

        if reference_output > 0:

            output = (
                raw_prediction
                /
                reference_output
            ) * 100

        else:

            output = raw_prediction


        # Never allow more than 100%

        output = max(
            0.0,
            min(
                output,
                100.0
            )
        )


    output = round(
        output,
        2
    )


    return (
        output,
        raw_prediction
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# API - PREDICT
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None:

        return {
            "success": False,
            "error": "ML model unavailable."
        }, 500


    # --------------------------------------------------------
    # READ JSON
    # --------------------------------------------------------

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return {
                "success": False,
                "error":
                    "No input data received."
            }, 400


        study_hours = float(
            data.get(
                "study_hours"
            )
        )


    except Exception:

        return {
            "success": False,
            "error":
                "Invalid study hours."
        }, 400


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if study_hours < 0:

        return {
            "success": False,
            "error":
                "Study hours cannot be negative."
        }, 400


    if study_hours > 24:

        return {
            "success": False,
            "error":
                "Study hours must be between 0 and 24."
        }, 400


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    try:

        output, raw_prediction = (
            calculate_prediction(
                study_hours
            )
        )


    except Exception as error:

        print(
            "[ERROR] Prediction failed:",
            error
        )


        return {
            "success": False,
            "error":
                "Prediction failed."
        }, 500


    # --------------------------------------------------------
    # PRINT INPUT + OUTPUT TABLE
    # --------------------------------------------------------

    print_prediction_table(

        study_hours=study_hours,

        prediction=output,

        raw_prediction=raw_prediction,

        request_type="LIVE"

    )


    # --------------------------------------------------------
    # RETURN JSON
    # --------------------------------------------------------

    return {

        "success": True,

        "study_hours":
            study_hours,

        "prediction":
            output,

        "raw_prediction":
            round(
                raw_prediction,
                2
            )

    }


# ============================================================
# API - SAVE PREDICTION
# ============================================================

@app.route(
    "/api/save",
    methods=["POST"]
)
def api_save():


    global df


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None:

        return {
            "success": False,
            "error":
                "ML model unavailable."
        }, 500


    # --------------------------------------------------------
    # READ INPUT
    # --------------------------------------------------------

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return {
                "success": False,
                "error":
                    "No input data received."
            }, 400


        study_hours = float(
            data.get(
                "study_hours"
            )
        )


    except Exception:

        return {
            "success": False,
            "error":
                "Invalid study hours."
        }, 400


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if study_hours < 0 or study_hours > 24:

        return {
            "success": False,
            "error":
                "Study hours must be between 0 and 24."
        }, 400


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        output, raw_prediction = (
            calculate_prediction(
                study_hours
            )
        )


    except Exception as error:

        print(
            "[ERROR] Prediction failed:",
            error
        )


        return {
            "success": False,
            "error":
                "Prediction failed."
        }, 500


    # --------------------------------------------------------
    # SAVE INTO DATAFRAME
    # --------------------------------------------------------

    new_prediction = pd.DataFrame(

        {
            "Study Hours": [
                study_hours
            ],

            "Predicted Output": [
                output
            ]
        }

    )


    df = pd.concat(

        [
            df,
            new_prediction
        ],

        ignore_index=True

    )


    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    try:

        df.to_csv(
            CSV_PATH,
            index=False
        )


    except Exception as error:

        print(
            "[ERROR] CSV save failed:",
            error
        )


        return {

            "success": False,

            "error":
                "Prediction generated but CSV save failed."

        }, 500


    # --------------------------------------------------------
    # TERMINAL OUTPUT
    # --------------------------------------------------------

    print_prediction_table(

        study_hours=study_hours,

        prediction=output,

        raw_prediction=raw_prediction,

        request_type="SAVED"

    )


    # --------------------------------------------------------
    # PRINT COMPLETE TABLE
    # --------------------------------------------------------

    print_history_table()


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "success": True,

        "study_hours":
            study_hours,

        "prediction":
            output,

        "raw_prediction":
            round(
                raw_prediction,
                2
            ),

        "total_predictions":
            len(df)

    }


# ============================================================
# API - HISTORY
# ============================================================

@app.route(
    "/api/history",
    methods=["GET"]
)
def api_history():


    global df


    records = df.to_dict(
        orient="records"
    )


    return {

        "success": True,

        "history":
            records,

        "total":
            len(records)

    }


# ============================================================
# API - ANALYTICS
# ============================================================

@app.route(
    "/api/analytics",
    methods=["GET"]
)
def api_analytics():


    global df


    if df.empty:

        return {

            "success": True,

            "total_predictions": 0,

            "average": 0,

            "highest": 0,

            "lowest": 0,

            "best_study_hours": 0

        }


    scores = pd.to_numeric(

        df[
            "Predicted Output"
        ],

        errors="coerce"

    ).dropna()


    if scores.empty:

        return {

            "success": True,

            "total_predictions": 0,

            "average": 0,

            "highest": 0,

            "lowest": 0,

            "best_study_hours": 0

        }


    average = float(
        scores.mean()
    )


    highest = float(
        scores.max()
    )


    lowest = float(
        scores.min()
    )


    best_index = scores.idxmax()


    best_hours = float(

        df.loc[
            best_index,
            "Study Hours"
        ]

    )


    return {

        "success": True,

        "total_predictions":
            len(scores),

        "average":
            round(
                average,
                2
            ),

        "highest":
            round(
                highest,
                2
            ),

        "lowest":
            round(
                lowest,
                2
            ),

        "best_study_hours":
            best_hours

    }


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.route(
    "/api/clear-history",
    methods=["POST"]
)
def clear_history():


    global df


    df = pd.DataFrame(

        columns=[
            "Study Hours",
            "Predicted Output"
        ]

    )


    try:

        df.to_csv(

            CSV_PATH,

            index=False

        )


        print("\n")

        print(
            "=" * 75
        )

        print(
            "             PREDICTION HISTORY CLEARED"
        )

        print(
            "=" * 75
        )

        print()


        return {

            "success": True,

            "message":
                "Prediction history cleared."

        }


    except Exception as error:

        return {

            "success": False,

            "error":
                str(error)

        }, 500


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":


    print("\n")


    print(
        "=" * 82
    )


    print(
        "             STUDENT MARK PREDICTOR"
    )


    print(
        "             MACHINE LEARNING SYSTEM"
    )


    print(
        "=" * 82
    )


    print(
        "Server          : http://127.0.0.1:5000"
    )


    print(
        "Input Feature   : Study Hours"
    )


    print(
        "Input Range     : 0 - 24 Hours"
    )


    print(
        "Output          : Predicted Marks"
    )


    print(
        "Output Range    : 0 - 100%"
    )


    print(
        "Model           : student_mark_predictor.pkl"
    )


    print(
        "Data Storage    : smp_data_from_app.csv"
    )


    print(
        "=" * 82
    )


    print(
        "Waiting for prediction requests..."
    )


    print(
        "=" * 82
    )


    print()


    # --------------------------------------------------------
    # START FLASK
    # --------------------------------------------------------

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )
