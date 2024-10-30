from flask import Flask, request,render_template
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
#vector = TfidfVectorizer(stop_words='english', max_df=0.7)
vector = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("finalized_model.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    print("Prediction route hit")  # Debug statement
    if request.method == "POST":
        news = str(request.form['news'])
        print(f"Received news: {news}")  # Debug statement
        predict = model.predict(vector.transform([news]))
        print(f"Prediction: {predict}")  # Debug statement
        return render_template("prediction.html", prediction_text=f"News headline is -> {predict}")
    else:
        return render_template("prediction.html")


if __name__ == '__main__':
    app.debug = True
    app.run()
