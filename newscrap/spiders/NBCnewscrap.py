import scrapy
from ..items import ScrapenewsItem

class ScrapeNews(scrapy.Spider):
    name = 'nbcnews'
    start_urls = [

        'https://www.nbcnews.com/business'
        #'https://www.nbcnews.com/politics'
        #'https://www.nbcnews.com/sports'
        #'https://www.nbcnews.com/culture-matters'

        #'https://www.aljazeera.com/sports/'
        #'https://www.aljazeera.com/economy/'
        #'https://www.aljazeera.com/us-canada/'
        #'https://www.aljazeera.com/interactives/'


        #'https://www.bbc.com/news/us-canada'
        #'https://www.bbc.com/culture'
        #'https://www.bbc.com/business'
        #'https://www.bbc.com/sports'
    ]

    def parse(self, response):
        if 'culture-matters' in response.url:
            category = 'entertainment'
        else:
            category = response.url.split('/')[-1]  # Get last part of URL as category

        all_news = response.css("div.wide-tease-item__info-wrapper > a")

        for news in all_news:
            title = news.css('h2::text').get()
            link = news.css('::attr(href)').get()

            if link:
                yield scrapy.Request(link, callback=self.parse_article, meta={'title': title, 'category': category})

    def parse_article(self, response):
        title = response.meta['title']
        category = response.meta['category']
        source = 'NBC'  # Assuming fixed source as 'NBC'

        # Extract content from the article page
        content = self.extract_content(response)

        # Create ScrapenewsItem with extracted data
        item = ScrapenewsItem()
        item['title'] = title.strip() if title else None
        item['category'] = category
        item['content'] = content
        item['source'] = source
        item['link'] = response.url

        yield item

    def extract_content(self, response):
        # Extract content from
        content = ' '.join(response.css('div.article-body__content p::text').extract()).strip()
        return content