import random
import errno
import os
import sys
from summarizer import Summarizer
from googletrans import Translator
from settings import CHANNEL_SECRET, CHANNEL_ACCESS_TOKEN
from argparse import ArgumentParser
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    TemplateSendMessage, ConfirmTemplate, MessageTemplateAction,
    ButtonsTemplate, URITemplateAction, PostbackTemplateAction,
    CarouselTemplate, CarouselColumn, PostbackEvent,
    StickerMessage, StickerSendMessage, LocationMessage, LocationSendMessage,
    ImageMessage, VideoMessage, AudioMessage,
    UnfollowEvent, FollowEvent, JoinEvent, LeaveEvent, BeaconEvent
)

app = Flask(__name__)

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)
summarizer = Summarizer()
static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')
translator = Translator()

def make_static_tmp_dir():
    try:
        os.makedirs(static_tmp_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(static_tmp_path):
            pass
        else:
            raise


@app.route('/')
def index():
    return 'Bot is running'


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


def reply(event, message):
    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text=message))


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    text = event.message.text
    cmd = text.split()
    print(event.as_json_string())
    flag = False
    for ch in ['apa', 'ap', 'siapa', 'what', 'who', 'siapakah']:
        flag |= text.lower().startswith(ch)

    if cmd[0] == '/check' and len(cmd) > 1:
        reply(event,random.randint(0,int(cmd[1])))
    elif '?' in text and flag and text.startswith('mbok,'):
        print("ini detectednya :" + str(translator.detect(text).lang))
        language = 'indonesian' if (translator.detect(text).lang == 'id' or translator.detect(text).lang == 'msid' or translator.detect(text).lang == 'idms') else 'english'
        print(language)
        reply(event, str(summarizer.summarize(language=language, query=text, size=2)))


if __name__ == "__main__":
    # arg_parser = ArgumentParser(
    #     usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    # )
    # arg_parser.add_argument('-p', '--port', default=8000, help='port')
    # arg_parser.add_argument('-d', '--debug', default=False, help='debug')
    # options = arg_parser.parse_args()
    #
    # # create tmp dir for download content
    # make_static_tmp_dir()
    #
    # app.run(debug=options.debug, port=options.port)

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0')
