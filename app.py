from flask import Flask, render_template, request

import pickle

app = Flask(__name__)

# Loading the model outside the route to avoid loading it every time the route is called
with open("sentimentanalysis_mlflow.pkl", "rb") as f:
    model = pickle.load(f)

#request.method == 'POST'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST'])  # Changed the method to POST
def result():
    global model
    text = request.form['text']
    print(text,model)
    prediction = model.predict([text])[0]  # Predicting sentiment for the text
    print("***********",prediction)
    # Rendering the output template with the predicted sentiment
    if text == "":
        return render_template('home.html', text='Please Enter Text')
    else:
        return render_template('home.html', valid=prediction,text=text)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)