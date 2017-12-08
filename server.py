from sanic import Sanic
from sanic.response import json
from summarizer import Summarizer

summarizer = Summarizer()
app = Sanic(__name__)

@app.get('/')
async def root(request):
    return json({'hello':'world'})

@app.post('/summarize')
async def summarize(request):
    data = request.json
    query = data.get('query', None)
    type = data.get('type', None)
    language = data.get('language', None)
    size = data.get('size', 1)

    summary = summarizer.summarize(type=type, query=query, size=int(size))

    response = {
        'summary': summary
    }

    return json(response)

app.run(host='0.0.0.0', port=8000, debug=True)
