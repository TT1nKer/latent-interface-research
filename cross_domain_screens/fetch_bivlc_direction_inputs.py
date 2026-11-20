"""Fetch pinned public ranking tables and photo IDs without loading image pixels.

Requires requests and pyarrow. Writes third-party data to a directory of your choice.
The parquet reader fetches only file footers and six small metadata columns.
"""
import argparse
import csv
import json
import struct
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as parquet
import requests

DATA_REVISION = '618ad54047c6bc3807226ce15de6b5ca027c4255'
METADATA = ['image.path', 'negative_image.path', 'caption', 'negative_caption', 'type', 'subtype']


def fetch_range(url, first, last):
    byte_range = f'{first}-{last}'
    with requests.get(url + '?metadata_range=' + byte_range, headers={'Range': 'bytes=' + byte_range},
                      stream=True, timeout=30) as response:
        response.raise_for_status()
        if response.status_code != 206:
            raise ValueError('Server ignored the requested byte range')
        extent, total = response.headers['Content-Range'].removeprefix('bytes ').split('/')
        count = last if first == '' else last-first+1
        start = int(total)-count if first == '' else first
        if extent != f'{start}-{start+count-1}' or not 0 < count < 1_000_000:
            raise ValueError('Unexpected source range')
        data = response.raw.read(count)
        if len(data) != count:
            raise ValueError('Incomplete response')
        return data, int(total)


def photo_ids(shard, path):
    url = (f'https://huggingface.co/datasets/imirandam/BiVLC/resolve/{DATA_REVISION}/'
           f'data/test-{shard:05d}-of-00002.parquet')
    trailer, size = fetch_range(url, '', 8)
    if trailer[4:] != b'PAR1':
        raise ValueError('Parquet footer signature missing')
    footer_length = struct.unpack('<I', trailer[:4])[0] + 8
    footer, _ = fetch_range(url, size-footer_length, size-1)
    with path.open('wb') as scratch:
        scratch.write(b'PAR1')
        scratch.seek(size-footer_length)
        scratch.write(footer)
    file = parquet.ParquetFile(path)
    with path.open('r+b') as scratch:
        for i in range(file.metadata.num_row_groups):
            group = file.metadata.row_group(i)
            for j in range(group.num_columns):
                c = group.column(j)
                if c.path_in_schema in METADATA:
                    start = c.dictionary_page_offset if c.has_dictionary_page else c.data_page_offset
                    data, _ = fetch_range(url, start, start+c.total_compressed_size-1)
                    scratch.seek(start)
                    scratch.write(data)
    for row in parquet.ParquetFile(path).read(columns=METADATA).to_pylist():
        yield [row['image']['path'],row['negative_image']['path'],*(row[k] for k in METADATA[2:])]


def fetch(directory):
    protocol = json.loads(Path(__file__).with_name('bivlc_cycle_protocol_20261120.json').read_text())
    directory.mkdir(exist_ok=True, parents=True)
    source = protocol['source']
    for name in source['files']:
        response = requests.get(f"https://raw.githubusercontent.com/{source['project']}/{source['revision']}/results/{name}",timeout=25)
        response.raise_for_status()
        (directory/name).write_bytes(response.content)
    with TemporaryDirectory() as temporary, (directory/'source_photo_ids.csv').open('w',newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['filename', 'negative_filename', *METADATA[2:]])
        for shard in range(2):
            writer.writerows(photo_ids(shard, Path(temporary)/f'{shard}.sparse'))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    fetch(cli.parse_args().directory)
