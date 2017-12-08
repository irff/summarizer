from sanic import Sanic
from sanic.response import json
from scraper import Scraper

scraper = Scraper()
app = Sanic(__name__)

@app.get('/')
async def root(request):
    return json({'hello':'world'})

@app.post('/summarize')
async def summarize(request):
    data = request.json
    query = data.get('query', None)

    intro = scraper.get_intro(query)

    response = {
        'summary': intro
    }

    return json(response)

app.run(host='0.0.0.0', port=8000, debug=True)
