#!/usr/bin/env python
# coding: utf-8

# # Imports and settings
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import os
import subprocess

scrape_dir = './wav/'
url = 'http://www.avisoft.com/animal-sounds/'
num_scrape = -1 # number of files to scrape. -1 to scrape all

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# div containing bird recordings and spectrograms
bird_div = soup.find_all('div',attrs={'class':'table-responsive'})[0]
bird_links = bird_div.find_all('a')

print('Beginning data download!')
counter = 0
for link in bird_links:
    href = link['href']
    filename = os.path.split(href)[-1]
    if os.path.splitext(filename)[1] == '.wav':
        counter +=1
        urllib.request.urlretrieve(href, scrape_dir + filename)
        print("Downloaded file " + str(counter) + ": " + filename)
        if counter == num_scrape:
            print("num_scrape reached")
            break

