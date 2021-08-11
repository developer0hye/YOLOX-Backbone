from urllib import request

def download_from_url(url, filename):
    request.urlretrieve(url, filename)