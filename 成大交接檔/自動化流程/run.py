# 載入必須套件
from flask import Flask, request,jsonify
from flask_cors import CORS
import predict
import renew
import predict_single



# 創建Flask app物件
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "hello!!"

@app.route('/test')
def test():
    return "hello!! test"

@app.route('/predict',methods=['POST'])
def get_information():
    intsertValue = request.get_json()
    object = intsertValue['目標']
    room = intsertValue['病房']
    part_no = intsertValue['料號']
    if object == '預測':
        predict.main(room)
        temp = "Predict OK"
    elif object == '更新使用特徵':
        renew.main(room)
        temp = "renew OK"
    elif object == '單一衛材預測':
        predict_single.main(room,part_no)
        temp = "renew OK"
    return jsonify({'return':str(temp)})
        
if __name__ == "__main__":
    app.run(port=5000, debug=True)