import requests
import os

class PixabayImageSearch:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://pixabay.com/api/"
        
    def search_videos(self, query, per_page=10, page=1):
        """
        Search for videos on Pixabay.

        :param query: The search term.
        :param per_page: Number of results per page. Default is 10.
        :param page: Page number for pagination. Default is 1.
        :return: A list of video URLs.
        """
        video_url = self.base_url + "videos/"
        params = {
            "key": self.api_key,
            "q": query,
            "per_page": per_page,
            "page": page
        }

        try:
            response = requests.get(video_url, params=params)
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()
            return [hit['videos']['medium']['url'] for hit in data.get('hits', [])]
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return []

    def search_images(self, query, image_type="photo", per_page=10, page=1):
        """
        Search for images on Pixabay.

        :param query: The search term.
        :param image_type: Type of image. Default is 'photo'. Options include 'photo', 'illustration', 'vector'.
        :param per_page: Number of results per page. Default is 10.
        :param page: Page number for pagination. Default is 1.
        :return: A list of image URLs.
        """
        params = {
            "key": self.api_key,
            "q": query,
            "image_type": image_type,
            "per_page": per_page,
            "page": page
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()
            return [hit['webformatURL'] for hit in data.get('hits', [])]
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return []
