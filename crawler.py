import asyncio
import os
import time
import html
import json
from urllib.parse import urlparse
from pathlib import Path
from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext
from googlesearch import search  

# For scrapping
# pip install googlesearch-python
# python -m pip install 'crawlee[all]'
# playwright install
# pip install beautifulsoup4

# For labelling dataset
# pip install -U label-studio
# label-studio start

PHRASE = "Delhi Metro Rail Corporation vs Delhi Airport Metro Express arbitration case" 
TOP_N = 10 
output_directory = f"scraped_content/{PHRASE.replace(' ', '_')[:10]}"  

def get_top_urls(query, num_results):
    urls = []
    for url in search(query, num_results=num_results):
        urls.append(url)
        print(f"Found URL: {url}")
        time.sleep(0.1) 
    return urls

def sanitize_filename(url):
    parsed = urlparse(url)
    netloc = parsed.netloc
    if (netloc.startswith('www.')):
        netloc = netloc[4:]  
    filename = netloc + parsed.path.replace('/', '_')
    filename = ''.join(c if c.isalnum() or c in ['_', '-', '.'] else '_' for c in filename)
    return filename[:10] 

def create_label_studio_xml(html_content, url, title):
    json_content = json.dumps([{"html_content": html_content}])
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<View>
  <HyperText name="text" value="$html_content" valueType="text" />
  <Labels name="labels" toName="text">
    <Label value="Important" />
    <Label value="Review" />
  </Labels>
  <Meta>
    <Info name="url" value="{html.escape(url)}"/>
    <Info name="title" value="{html.escape(title)}"/>
  </Meta>
</View>
"""
    return xml_content, json_content

async def main() -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    html_output_path = output_path / "html"
    html_output_path.mkdir(exist_ok=True)
    
    json_output_path = output_path / "json"
    json_output_path.mkdir(exist_ok=True)

    print(f"Searching Google for: {PHRASE}")
    urls = get_top_urls(PHRASE, TOP_N)
    
    crawler = PlaywrightCrawler(
        headless=True, 
    )

    @crawler.router.default_handler
    async def request_handler(context: PlaywrightCrawlingContext) -> None:
        url = context.request.url
        context.log.info(f'Processing {url} ...')

        await context.page.wait_for_load_state("networkidle", timeout=30000)
        html_content = await context.page.content()
        title = await context.page.title()
        _, json_content = create_label_studio_xml(html_content, url, title)

        filename = sanitize_filename(url)

        json_filename = filename + '.json'
        json_file_path = json_output_path / json_filename
        with open(json_file_path, 'w', encoding='utf-8') as f:
            f.write(json_content)
        context.log.info(f'Saved JSON content to {json_file_path}')

        html_filename = filename + '.html'
        html_file_path = html_output_path / html_filename
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        context.log.info(f'Saved HTML content to {html_file_path}')

        data = {
            'url': url,
            'title': title,
            'saved_json': str(json_file_path),
            'saved_html': str(html_file_path),
        }
        await context.push_data(data)

    await crawler.run(urls)
    print(f"Scraping completed. Files saved to:")
    print(f"- JSON: {output_directory}/json/")
    print(f"- HTML: {output_directory}/html/")
    print(f"You can now import the JSON files into Label Studio for annotation.")

if __name__ == '__main__':
    asyncio.run(main())
