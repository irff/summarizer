# def-summarizer
Definition Lookup &amp; Summarizer

## Using Scraper class
For scraping wikipedia articles.
```python
scraper = Scraper()

intro = scraper.get_intro('dolly parton')
print(result)

full_content = scraper.get('dolly parton')
print(full_content)
```