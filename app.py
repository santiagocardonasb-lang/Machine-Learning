from flask import Flask,render_template, request
import LinearRegression

app = Flask(__name__)


@app.route("/")
def home():
    return "Hello World"


@app.route("/templates")
def template():
    return render_template("index.html")


@app.route("/LinearRegression", methods = ["GET","POST"])
def calculate():
    calculateResult = None
    if request.method == "POST":
        hours = float (request.form["hours"])
        calculateResult = LinearRegression.calculateGrade(hours)
    return render_template("tempLinearRegression.html", result = calculateResult)   