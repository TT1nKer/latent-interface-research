"""Obtain public Assembly101 annotation files for the pre-registered validation screen.

No images/video. The analysis computes content hashes for the fetched inputs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
import re
import requests

FINE_VALIDATION = '11vLrFT8pYk7wPBes62KdFopv8q8zuMiD'
ASSEMBLY_SPLIT = '1rXNhSvm87cUxCBybLJAqpfjI-ncvuDRD'
COARSE_FOLDER = '11gOlHRkU6ppgs9CtMTkPw8Nkm0AiV5WH'


class PublicFolder(HTMLParser):
    def __init__(self):
        super().__init__()
        self.index = {}
        self.entry = None
        self.title = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag=='a':
            found=re.search(r'drive.google.com/file/d/([^/]+)/',attrs.get('href',''))
            if found:
                self.entry=found[1]
        if tag=='div' and attrs.get('class')=='flip-entry-title':
            self.title=True

    def handle_data(self, data):
        if self.title and self.entry:
            self.index[data.strip()]=self.entry

    def handle_endtag(self, tag):
        if tag=='div':
            self.title=False
        if tag=='a':
            self.entry=None


def fetch(item):
    path,file_id,limit=item
    with requests.get('https://drive.usercontent.google.com/download',params={'export':'download','confirm':'t','id':file_id},stream=True,timeout=45) as response:
        response.raise_for_status()
        if 'html' in response.headers.get('content-type','').lower():
            raise ValueError('Received an HTML page instead of a public annotation file')
        data=response.raw.read(limit+1)
    if not 0<len(data)<=limit:
        raise ValueError('Unexpected annotation file size')
    path.write_bytes(data)


def main(directory):
    directory.mkdir(exist_ok=True,parents=True)
    coarse=directory/'coarse'
    coarse.mkdir(exist_ok=True)
    split=directory/'val_coarse_assembly.txt'
    fetch((split,ASSEMBLY_SPLIT,12000))
    response=requests.get('https://drive.google.com/embeddedfolderview',params={'id':COARSE_FOLDER},timeout=30)
    response.raise_for_status()
    folder=PublicFolder()
    folder.feed(response.text)
    tasks=[(directory/'validation.csv',FINE_VALIDATION,36_000_000)]
    for line in split.read_text().splitlines():
        filename=line.split()[0]
        if not re.fullmatch(r'assembly_[A-Za-z0-9_-]+\.txt',filename):
            raise ValueError('Unexpected source filename')
        tasks.append((coarse/filename,folder.index[filename],100000))
    with ThreadPoolExecutor(max_workers=5) as threads:
        list(threads.map(fetch,tasks))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    main(parser.parse_args().directory)
