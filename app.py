"""Data with Roots: a Flask web application about Machine Learning."""

from flask import Flask, abort, render_template, request

import content
import model

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


if __name__ == "__main__":
    app.run(debug=True)