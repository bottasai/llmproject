import requests
import csv
from bs4 import BeautifulSoup

def download_reviews(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0;Win64) AppleWebkit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'}
    print(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    reviews = soup.find_all('div', class_='styles_reviewContent__0Q2Tg')
    extracted_reviews = []
    for review in reviews:
        review_text = review.find('p', class_='typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn')
        if review_text:
            extracted_reviews.append(review_text.text.strip())

    return extracted_reviews

def save_reviews_to_csv(reviews, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Review'])
        writer.writerows([[review] for review in reviews])

def download_all_reviews(base_url, duration, num_pages):
    all_reviews = []

    for page in range(1, num_pages + 1):
        url = f'{base_url}?date={duration}&page={page}'
        reviews = download_reviews(url)
        all_reviews.extend(reviews)

    save_reviews_to_csv(all_reviews, 'trustpilot_reviews.csv')

# Example usage
base_url = 'https://www.trustpilot.com/review/bt.com'
duration = 'last6months'
num_pages = 5  # Specify the number of pages to scrape

download_all_reviews(base_url, duration, num_pages)
#download_reviews(base_url)
