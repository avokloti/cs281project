""" This script parses the Berlin nature archive and downloads a set of birdsong recordings."""
""" Irina Tolkova, Nov 2019."""

import requests
import urllib.request
import time
from bs4 import BeautifulSoup

# address of local folder where data should be saved
local_folder = '/Users/ira/Documents/bioacoustics/cs281/data/birdsong/'

# url to Berlin museum site
url = 'https://www.tierstimmenarchiv.de/RefSys/_TsaRefData/MP3s/'

# connect to the URL
response = requests.get(url)

# parse HTML and save to BeautifulSoup object
soup = BeautifulSoup(response.text, "html.parser")

# get all "a" tags
atags = soup.findAll('a')

# check total number of tags
numlinks = len(atags) - 6

# note on this data: There are 6490 recordings stored as tags in "atags". The first 6 tags refer to other (irrelevant) links. For the final dataset, we can download all 6490, but the code below starts with the first  20.

print('Beginning data download!')

# download the first 20 recordings (to download all, change '20' to 'numlinks'
for i in range(20):
    tag = atags[i + 6]
    download_url = 'https://www.tierstimmenarchiv.de/RefSys/_TsaRefData/MP3s/' + tag['href']
    urllib.request.urlretrieve(download_url, local_folder + tag['href'])
    print('-- Downloaded file ' + str(i) + ': ' + tag['href'])

