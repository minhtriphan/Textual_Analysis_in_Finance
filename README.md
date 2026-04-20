
# Textual Analysis in Finance
This is the git repo for the Textual Analysis in Finance lecture at the University of Basel. It will contain course materials, including slides and code

# Lecture info
* Time: Every Thursday (Group 1) and Friday (Group 2), from 14:00 to 18:00
* Duration: from Apr 10, 2026 to May 28, 2027
* Location: PC lab S18 HG.37
* Course URL: https://vorlesungsverzeichnis.unibas.ch/de/vorlesungsverzeichnis?id=301098

# Lecture content
* Week 1 - Course overview, Bag-of-Words
* Week 2 - Sentiment analysis and topic modeling
* Week 3 - Neural embedding
* Week 4 - Language models
* Week 5 - Large language models
* Week 6 - Retrieval-augmented generation, feedback, guest lecture

# Data description
All necessary datasets are given in the Data folder, which includes:
* `Transcripts` - NVIDIA transcripts from 2020 to 2025 (fiscal) - Obtained from [FMP API](https://site.financialmodelingprep.com/developer/docs)
* `FinancialStatements` - TSLA financial statements - Obtained from [Loughran-McDonald shared folder](https://drive.google.com/drive/folders/1tZP9A0hrAj8ptNP3VE9weYZ3WDn9jHic?usp=drive_link)
* `CentralBankerSpeeches` - Central bankers' speeches - Downloaded from [BIS website](https://www.bis.org/cbspeeches/index.htm)

# Detailed contents and remarks
## Week 1 - Course overview, Bag-of-Words
* Code: Utility functions, currently include all cleaning functions
* Data: NVIDIA's earnings call transcripts from 2020 to 2025 (fiscal)
* Notice:

In Week 1, we need two additional packages named `contractions` and `wordcloud`. Install them before running any code by running the following code in the terminal

```
pip install contractions
pip install wordcloud
```

## Week 2 - Sentiment analysis and topic modeling
* Code: A class to vectorize texts using Bag-of-Words and TF-IDF
* Data: Financial Phrasebank and central bankers' speeches
* Notice: 

In this week, we need a package for dictionary-based sentiment analysis, so-called `pysentiment2`. Install it by `pip install pysentiment2`

## Week 3 - Neural embedding
* Code: Training/Inferring using ELMo
* Data: Tesla's 10-K files in from 2016 to 2025 for Word2Vec and Doc2Vec

## Week 4 - Language models
* Code: Fine-tuning FinBERT for sentiment analysis using Financial Phrasebank (to be updated)
* Data: Financial Phrasebank
