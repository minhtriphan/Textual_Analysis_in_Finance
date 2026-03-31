
Due to the heavy size of the data, it can't be uploaded to github. Please download it here: https://drive.google.com/file/d/1CnadkUgrHYDwxepPhZMoD1UbR4XoTJnm/view?usp=sharing

To download it using Python, the `gdown` package is needed. This is the package allows us to download a shared file (not folder) from Google drive

Install the package by running this `pip install gdown` on the terminal, then run the following code in Python (e.g., a cell in Jupyter notebook)

```
import gdown

url = 'https://drive.google.com/file/d/1CnadkUgrHYDwxepPhZMoD1UbR4XoTJnm/view?usp=sharing'
file_id = url.split('/')[-2]
download_url = f'https://drive.google.com/uc?id={file_id}'
output = 'central_banker_speeches.csv'
gdown.download(download_url, output, quiet = False)
```
