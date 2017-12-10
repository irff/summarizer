import wikipedia

ENGLISH = 'english'
INDONESIAN = 'indonesian'
EN = 'en'
ID = 'id'

class Scraper():
    def __init__(self, language):
        self.language = language

    def get_first_page(self, possible_page_titles):
        first_page = possible_page_titles[0]
        page_result = wikipedia.page(first_page)
        return page_result.content

    def get(self, query):
        if self.language == INDONESIAN:
            wikipedia.set_lang(ID)
        else:
            wikipedia.set_lang(EN)
        try:
            possible_page_titles = wikipedia.search(query)
            if len(possible_page_titles) > 0:
                return self.get_first_page(possible_page_titles)
            else:
                suggested_query = wikipedia.suggest(query)
                possible_page_titles = wikipedia.search(suggested_query)

                if len(possible_page_titles) > 0:
                    return self.get_first_page(possible_page_titles)

        except wikipedia.exceptions.DisambiguationError as e:
            possible_page_titles = e.options
            return self.get_first_page(possible_page_titles)

        return None

    def get_intro(self, query):
        content = self.get(query)

        if content is not None:
            intro = content.split('==', 1)[0]
            if intro is not None:
                return intro.rstrip('\n\r')

        return None

    def get_query(self, query, isInverse=False):
        if self.language == INDONESIAN and not isInverse:
            wikipedia.set_lang(ID)
            lang = INDONESIAN
        else:
            wikipedia.set_lang(EN)
            lang = ENGLISH
        try:
            possible_page_titles = wikipedia.search(query)
            if len(possible_page_titles) > 0:
                return possible_page_titles[0], 0, lang
            else:
                suggested_query = wikipedia.suggest(query)
                if len(suggested_query) < 1:
                    return [], -1, lang
                possible_page_titles = wikipedia.search(suggested_query)

                if len(possible_page_titles) > 0:
                    return possible_page_titles[0], -1, lang

        except wikipedia.exceptions.DisambiguationError as e:
            possible_page_titles = e.options
            return possible_page_titles[0], 0, lang
        else:
            return possible_page_titles[0], 0, lang

        return [], -1, lang
    def get_intro_lang(self, query, lang):
        if lang == INDONESIAN:
            wikipedia.set_lang(ID)
        else:
            wikipedia.set_lang(EN)
        try:
            content = self.get_first_page([query])
        except wikipedia.exceptions.DisambiguationError as e:
            possible_page_titles = e.options
            content = self.get_first_page(possible_page_titles)

        if content is not None:
            intro = content.split('==', 1)[0]
            if intro is not None:
                return intro.rstrip('\n\r\s')

        return None
