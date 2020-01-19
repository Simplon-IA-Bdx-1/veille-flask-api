from flask import Flask, render_template, request
import pickle

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict/')
def get_predict():
    return render_template('get-prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [[
        request.form['sepal-length'],
        request.form['sepal-width'],
        request.form['petal-length'],
        request.form['petal-width']
    ]]

    model = pickle.load(open('./models/classifier.sav', 'rb'))
    output = model.predict(input_data)

    if output == 0:
        classe = 'Setosa'
    if output == 1:
        classe = 'Versicolor'
    if output == 2:
        classe = 'Virginica'   

    return render_template('predict.html', classe = classe)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
