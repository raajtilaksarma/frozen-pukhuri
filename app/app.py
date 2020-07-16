import bot as my_bot
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=['POST','GET'])
def index():
    point_path = []
    if request.method == 'POST':
        attempts = int(request.form['attempts'])
        point_path = my_bot.attempting(attempts)
        print(point_path)
    return render_template('index.html', point_path = point_path)

if __name__=='__main__':
    app.run()
