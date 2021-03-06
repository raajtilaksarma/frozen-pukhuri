import bot as my_bot
from flask import Flask, render_template, request
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)

# @app.route("/", methods=['POST','GET'])
# def index():
#     point_path = []
#     if request.method == 'POST':
#         attempts = int(request.form['attempts'])
#         point_path = my_bot.attempting(attempts)
#         print(point_path)
#     return render_template('index.html', point_path = point_path)

@app.route('/')
def index():
    return render_template('index.html')

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))
    print(type(json))
    if 'attempts' in json:
        print(json['attempts'])
        attempts = int(json['attempts'])
        for point in my_bot.attempting(attempts):
            socketio.emit('my response', {'message':str(point)}, callback=messageReceived)

if __name__=='__main__':
    socketio.run(app, debug=True)

    
