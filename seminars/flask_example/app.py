from flask import Flask, request, render_template

app = Flask(__name__)

def search(query):
    pass
    return [('https://yandex.ru','Yandex'),
            ('https://vk.com/','VK')]
 
@app.route('/')
def index():
    if request.args:
        query = request.args['query']
        links = search(query)
        return render_template('index.html',links=links)
    return render_template('index.html',links=[])

if __name__ == '__main__':
    app.run(debug=True)
            
