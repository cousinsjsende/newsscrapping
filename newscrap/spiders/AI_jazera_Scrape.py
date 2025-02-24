import scrapy
from ..items import ScrapenewsItem

class ScrapeNews(scrapy.Spider):
    name = 'news_aijazera'
    start_urls = [
        #'https://www.nbcnews.com/business'
        #'https://www.nbcnews.com/politics'
        #'https://www.nbcnews.com/sports'
        #'https://www.nbcnews.com/culture-matters'

        #'https://www.aljazeera.com/sports/'
        #'https://www.aljazeera.com/economy/' # this is for business
        'https://www.aljazeera.com/us-canada/' # this is politics 
        #'https://www.aljazeera.com/interactives/' # for intertainment


        #'https://www.bbc.com/news/us-canada'
        #'https://www.bbc.com/culture'
        #'https://www.bbc.com/business'
        #'https://www.bbc.com/sports'
    ]

    def parse(self, response):
        if 'us-canada' in response.url:
            category = 'politics'
        else:
            category = response.url.split('/')[-2]  # Get last part of URL as category

        all_news = response.css("h3.gc__title a")

        for news in all_news:
            title = news.css('span::text').get()
            relative_link = news.css('::attr(href)').get()  # Get relative link

            if relative_link:
                # Construct absolute link by appending base URL
                link = response.urljoin(relative_link)

                yield scrapy.Request(link, callback=self.parse_article, meta={'title': title, 'category': category})

    def parse_article(self, response):
        title = response.meta['title']
        category = response.meta['category']
        source = 'Al Jazeera'  # Fixed source as 'Al Jazeera'

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
        # Extract content from the article page
        content = ' '.join(response.css('div.wysiwyg--all-content p::text').extract()).strip()
        return content
