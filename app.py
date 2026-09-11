"""Data with Roots: a Flask web application about Machine Learning."""

from flask import Flask, abort, render_template, request

import content
import model
import logistic_model
import perceptron_model

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html", active="home")


@app.route("/ml/concepts")
def ml_concepts():
    return render_template("ml_concepts.html", active="ml")


@app.route("/ml/types")
def ml_types():
    return render_template("ml_types.html", active="ml")


@app.route("/use-cases/<int:number>")
def use_case(number):
    if number not in content.USE_CASES:
        abort(404)

    return render_template(
        "use_case.html",
        active="use_cases",
        number=number,
        case=content.USE_CASES[number],
    )


@app.route("/linear-regression/concepts")
def lr_concepts():
    return render_template("lr_concepts.html", active="supervised")


@app.route("/linear-regression/application", methods=["GET", "POST"])
def lr_application():
    prediction = None
    error = None
    entered_value = ""

    if request.method == "POST":
        entered_value = request.form.get("investment", "").strip()

        if not entered_value:
            error = "Enter an advertising investment to get a prediction."
        else:
            try:
                investment = float(entered_value)
            except ValueError:
                error = "Enter a valid number, for example 12.5"
            else:
                if investment < 0:
                    error = "The advertising investment cannot be negative."
                else:
                    prediction = model.predict(investment)

    return render_template(
        "lr_application.html",
        active="supervised",
        info=model.INFO,
        chart=model.CHART,
        sample=model.dataset.head(10).to_dict("records"),
        prediction=prediction,
        error=error,
        entered_value=entered_value,
    )


@app.route("/logistic-regression/concepts")
def logistic_concepts():
    return render_template("logistic_concepts.html", active="supervised")


@app.route("/logistic-regression/application", methods=["GET", "POST"])
def logistic_application():
    result = None
    error = None
    entered_value = ""

    if request.method == "POST":
        entered_value = request.form.get("investment", "").strip()

        if not entered_value:
            error = "Enter an advertising investment to get a classification."
        else:
            try:
                investment = float(entered_value)
            except ValueError:
                error = "Enter a valid number, for example 12.5"
            else:
                if investment < 0:
                    error = "The advertising investment cannot be negative."
                else:
                    result = logistic_model.classify(investment)

    return render_template(
        "logistic_application.html",
        active="supervised",
        info=logistic_model.INFO,
        chart=logistic_model.CHART,
        sample=logistic_model.dataset.head(10).to_dict("records"),
        result=result,
        error=error,
        entered_value=entered_value,
    )


@app.route("/logistic-regression/metrics")
def logistic_metrics():
    return render_template(
        "logistic_metrics.html",
        active="supervised",
        info=logistic_model.INFO,
        metrics=logistic_model.METRICS,
    )


@app.route("/perceptron/concepts")
def perceptron_concepts():
    return render_template("perceptron_concepts.html", active="supervised")


@app.route("/perceptron/application", methods=["GET", "POST"])
def perceptron_application():
    result = None
    error = None
    entered = {"budget": "", "duration_days": "", "channels": ""}

    if request.method == "POST":
        for field in entered:
            entered[field] = request.form.get(field, "").strip()

        if not all(entered.values()):
            error = "Fill in the three fields to get a classification."
        else:
            try:
                budget = float(entered["budget"])
                duration = float(entered["duration_days"])
                channels = float(entered["channels"])
            except ValueError:
                error = "Enter valid numbers in the three fields."
            else:
                if budget < 0 or duration < 0 or channels < 0:
                    error = "The values cannot be negative."
                else:
                    result = perceptron_model.classify(budget, duration, channels)

    return render_template(
        "perceptron_application.html",
        active="supervised",
        info=perceptron_model.INFO,
        chart=perceptron_model.CHART,
        sample=perceptron_model.dataset.head(10).to_dict("records"),
        result=result,
        error=error,
        entered=entered,
    )


@app.route("/perceptron/metrics")
def perceptron_metrics():
    return render_template(
        "perceptron_metrics.html",
        active="supervised",
        info=perceptron_model.INFO,
        metrics=perceptron_model.METRICS,
        logistic=logistic_model.METRICS,
    )

if __name__ == "__main__":
    app.run(debug=True)