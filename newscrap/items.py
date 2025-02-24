
import scrapy

class ScrapenewsItem(scrapy.Item):
    # define the fields for your item here like:
    title = scrapy.Field()
    category = scrapy.Field()
    content = scrapy.Field()  
    source = scrapy.Field()
    link = scrapy.Field()