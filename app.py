"""Data with Roots: a Flask web application about Machine Learning."""

from flask import Flask, abort, render_template

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
    if number not in (1, 2, 3, 4):
        abort(404)
    return render_template("use_case.html", active="use_cases", number=number)


@app.route("/linear-regression/concepts")
def lr_concepts():
    return render_template("lr_concepts.html", active="supervised")


@app.route("/linear-regression/application")
def lr_application():
    return render_template("lr_application.html", active="supervised")


if __name__ == "__main__":
    app.run(debug=True)