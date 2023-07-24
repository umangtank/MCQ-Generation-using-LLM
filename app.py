from flask import Flask, request
from lmqg import TransformersQG


# initialize model
model = TransformersQG(language='en', model='lmqg/t5-base-squad-qg-ae')

app = Flask(__name__)
@app.route('/qna',methods = ['POST'])
def QnA():
    data = request.get_json()
    context = data['context']
    question_answer = model.generate_qa(context)
    print(question_answer)
    return question_answer

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)