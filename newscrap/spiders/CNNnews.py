import scrapy
from ..items import ScrapenewsItem

class ScrapeNews(scrapy.Spider):
    name = 'newscnn'
    start_urls = [

        'https://edition.cnn.com/politics'
        #'https://edition.cnn.com/business'
        #'https://edition.cnn.com/entertainment'
        # 'https://edition.cnn.com/sport'

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
        # if 'culture-matters' in response.url:
        #     category = 'entertainment'
        # else:
        category = response.url.split('/')[-1]  # Get last part of URL as category

        all_news = response.css("div.container_list-headlines__field-wrapper a")
        for news in all_news:
            title = news.css('span.container__headline-text ::text').get()
            relative_link = news.css('::attr(href)').get()
            if relative_link:
                link = response.urljoin(relative_link) 
                yield scrapy.Request(link, callback=self.parse_article, meta={'title': title, 'category': category})

    def parse_article(self, response):
        title = response.meta['title']
        category = response.meta['category']
        source = 'CNN'  

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
        content = ' '.join(response.css('div.article__content p ::text').extract()).strip()
        return content