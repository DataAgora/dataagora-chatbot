from flask import Flask, render_template
application = Flask(__name__)



@application.route('/')
def root():
    return render_template('index.html')

@application.route('/<path:path>')
def static_proxy(path):
    # send_static_file will guess the correct MIME type
    return application.send_static_file(path)

if __name__ == '__main__':
    application.run()