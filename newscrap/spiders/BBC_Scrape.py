import scrapy
from ..items import ScrapenewsItem

class ScrapeNews(scrapy.Spider):
    name = 'news_bbc'
    start_urls = [
        #'https://www.bbc.com/news/us-canada' # this is politics 
        #'https://www.bbc.com/culture'
        'https://www.bbc.com/business'
        #'https://www.bbc.com/sports' #problem here
    ]

    def parse(self, response):
        items = ScrapenewsItem()

        if 'us-canada' in response.url:
            category = 'politics'
        else:
            category = response.url.split('/')[-1]

        all_news = response.css(".khCtOO")

        for news in all_news:
            title = news.css('.bvDsJq::text').extract()
            content = news.css('.cNPpME::text').extract()
            source = 'BBC'
            relative_link = response.css('div[data-testid="edinburgh-card"] a[data-testid="internal-link"]::attr(href)').get()
            link = response.urljoin(relative_link)

            items['title'] = title
            items['category'] = category
            items['content'] = content
            items['source'] = source
            items['link'] = link

            yield items
