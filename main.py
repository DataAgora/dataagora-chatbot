from flask import Flask, render_template
from flask_cors import CORS
import requests, zipfile, io

application = Flask(__name__)
CORS(application)
application.config['CORS_HEADER'] = 'Content-Type'

@application.route('/')
def root():
    return render_template('index.html')

@application.route('/<path:path>')
def static_proxy(path):
    # send_static_file will guess the correct MIME type
    return application.send_static_file(path)

# @application.route('/download', methods=['POST', 'GET'])
# def download():
#     r = requests.get('https://dataagora-chatbot.s3-us-west-1.amazonaws.com/weights.zip')
#     z = zipfile.ZipFile(io.BytesIO(r.content))
#     z.extractall("static/weights")
#     print("DONE")
#     return "Success!"

if __name__ == '__main__':
    application.run()