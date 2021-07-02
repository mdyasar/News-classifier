from flask import Flask, render_template, request
from classify import classifier

app= Flask(__name__)

@app.route("/")
def home():
  return render_template("index.html")

@app.route("/predict", methods=["POST", "GET"])
def pred():
  news= request.form.get("news")
  res= classifier(news)
  return render_template("index.html", res=res)

if __name__=="__main__":
  app.run()