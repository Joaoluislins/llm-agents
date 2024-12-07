import scrapy
from scrapy.crawler import CrawlerProcess

class StevensSpider(scrapy.Spider):
    name = "stevens_spider"
    allowed_domains = ['stevens.edu']
    start_urls = ['https://www.stevens.edu/sitemap.xml']

    def parse(self, response):
    # Print the sitemap content to see its structure
        print(response.text)  # This will show you the raw XML content
    
    # Extract URLs from sitemap and follow them
        for url in response.xpath('//loc/text()').getall():
            yield scrapy.Request(url=url, callback=self.parse_page)

    def parse_page(self, response):
        # Extract content, clean HTML tags, and save content to text files
        page_content = ' '.join(response.xpath('//p//text()').getall())
        page_title = response.url.split('/')[-1] or "index"
        filename = f"data/{page_title}.txt"
        
        # Ensure 'data' directory exists
        import os
        os.makedirs('data', exist_ok=True)

        # Write content to file
        with open(filename, 'w') as f:
            f.write(page_content)

# Running the spider without FEEDS (since it's manually handled)
process = CrawlerProcess()
process.crawl(StevensSpider)
process.start()
