"""Download the original Kaggle archive without inventing substitute records."""
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile
import sys

URL = 'https://www.kaggle.com/api/v1/datasets/download/fedesoriano/heart-failure-prediction'


def main():
    destination = Path(__file__).resolve().parents[1] / 'data' / 'heart.csv'
    if destination.exists():
        print(f'{destination} already exists; leaving it unchanged.')
        return
    try:
        request = Request(URL, headers={'User-Agent': 'NeuroFuzzyCollegeProject/1.0'})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        with ZipFile(BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if Path(n).name == 'heart.csv']
            if len(names) != 1:
                raise ValueError('Archive does not contain exactly one heart.csv')
            content = archive.read(names[0])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        print(f'Downloaded {destination}; training will validate its schema and counts.')
    except (OSError, BadZipFile, ValueError) as exc:
        print(f'Download failed: {exc}\nDownload heart.csv manually from '
              'https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction '
              f'and place it at {destination}.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
