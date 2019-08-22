from flask import Flask, render_template, request, redirect
from flask_cors import CORS
from flask_talisman import Talisman

application = Flask(__name__)
CORS(application)
Talisman(application, content_security_policy=None)
application.config['CORS_HEADER'] = 'Content-Type'

# @application.before_request
# def before_request():
#     if not request.is_secure:
#         url = request.url.replace('http://', 'https://', 1)
#         code = 301
#         return redirect(url, code=code)
#     else:
#         print('lol nope')

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