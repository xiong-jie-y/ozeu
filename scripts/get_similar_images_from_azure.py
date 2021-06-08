import os
import pathlib
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import click
from urllib.parse import urlparse
import requests, json

@click.command()
@click.argument('image_file')
@click.option("--max-count", default=1000, type=int)
@click.option("--start-offset", default=0, type=int)
@click.option("--output-dir", default="images")
@click.option("--subscription-key", required=True)
def main(image_file, max_count, start_offset, output_dir, subscription_key):
    BASE_URI = 'https://api.bing.microsoft.com/v7.0/images/visualsearch'
    # BASE_URI = "https://api.bing.microsoft.com/v7.0/images/details"
    SUBSCRIPTION_KEY = subscription_key
    imagePath = image_file

    os.makedirs(output_dir, exist_ok=True)

    img_extensions = ("jpg", "jpeg", "png", "bmp", "webp")

    HEADERS = {'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY}
    nextOffset = start_offset
    totalEstimate = max_count
    while nextOffset < totalEstimate:
        file = {
            'image' : ('myfile', open(imagePath, 'rb')),

        }
        body = {
            "knowledgeRequest": json.dumps({
                "knowledgeRequest" : {
                    "invokedSkills": ["SimilarImages"],
                    "offset":nextOffset,
                    "count":150
                }
            })
        }

        response = requests.post(BASE_URI, data=body, headers=HEADERS, files=file)
        response.raise_for_status()
        search_results = response.json()
        # import IPython; IPython.embed()
        images = search_results['tags'][0]['actions'][0]['data']['value']
        nextOffset = search_results['tags'][0]['actions'][0]['data']['nextOffset']
        totalEstimate = search_results['tags'][0]['actions'][0]['data']['totalEstimatedMatches']
        # thumbnail_urls = [img["thumbnailUrl"] for img in search_results["value"][:16]]

        

        for image_meta in images:
            image_url = image_meta["contentUrl"]
            image_id = image_meta["imageId"]
            ext = pathlib.Path(urlparse(image_url).path).name.split(".")[-1]
            if ext not in img_extensions:
                continue
            try:
                image_data = requests.get(image_url)
                image_data.raise_for_status()
            except:
                continue
            else:
                try:
                    image = Image.open(BytesIO(image_data.content))        
                    image.save(os.path.join(output_dir, image_id + "." + ext))
                except UnidentifiedImageError:
                    continue
                except OSError:
                    continue

        print(f"Next is {nextOffset}")

if __name__ == "__main__":
    main()