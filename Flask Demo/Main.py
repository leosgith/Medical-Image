from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to the Future.'

@app.route('/rookies/<color>')
def rookies(color):
    if (color == "purple"):
        return "%s rookies" % color
    return "<h2>No it is PURPLE Rookies!</h2>"

if __name__ == "__main__":
    app.run()
