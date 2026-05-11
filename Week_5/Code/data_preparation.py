import os
from glob import glob
from typing import Union, List, Dict
from datasets import Dataset

def _read_transcript_contents(data_dir: str) -> List[Dict]:
    '''
    This function reads and loads the transcript content in a specific fiscal period, defined by year and quarter
    Args:
      - data_dir: the base directory containing all transcripts
    Return:
      The file name of the specified transcript (str). Return 0 if no transcript is found.
    '''
    filenames = glob(os.path.join(data_dir, f'transcript_NVDA_*_*_*.txt'))

    transcript_details = []

    for filename in filenames:
        # Read the file
        with open(filename, 'r') as f:
            content = f.read()
        # Parse the file name
        _, _, fiscal_year, fiscal_qtr, earning_date = filename.split('/')[-1].split('.')[0].split('_')
        transcript_details.append({
            'fiscal_year': fiscal_year,
            'fiscal_qtr': fiscal_qtr,
            'content': content,
        })
    return transcript_details

def _organize_dataset_each_transcript(transcript: str, fiscal_year: int, fiscal_qtr: str) -> List[Dict]:
    '''
    This function processes and organizes the transcript contents into a list of dictionaries that will be converted into a HuggingFace dataset.
    Each pair contains a question: 'What are people discussing during NVIDIA earnings call for the fiscal period {fiscal_year}{fiscal_qtr}?' and a paragraph
    Args:
      - transcript (str): The content of the transcript
      - fiscal_year (int): The fiscal year
      - fiscal_qtr (str): The fiscal quarter; in the form of 'Q<quarter>', such as 'Q1' or 'Q2'
    Return:
      A list of dictionaries, each of which has two keys, 'instruction' and 'output', the value corresponding the 'output' argument contains the paragraphs
    '''
    instruct = f'What are people discussing during NVIDIA earnings call for the fiscal period {fiscal_year}{fiscal_qtr}?'
    all_paragraphs = transcript.split('\n')

    items = []
    for i, para in enumerate(all_paragraphs):
        each_input_pair = {
            'instruction': instruct,
            'output': f'The following paragraph is the {i}-th paragraph of their discussion:\n{para}'
        }
        items.append(each_input_pair)
    return items

def _organize_dataset(data_dir: str):
    '''
    This function organizes the transcripts' contents in the HuggingFace dataset to suit the input for fine-tuning an LLM
    '''
    # Read the transcripts' contents and details
    transcript_details = _read_transcript_contents(data_dir)

    items = []
    for item in transcript_details:
        print(f'Processing the transcript in {item["fiscal_year"]}{item["fiscal_qtr"]}')
        # Organize the data using the _organize_dataset_each_transcript function
        items.extend(_organize_dataset_each_transcript(item['content'], item['fiscal_year'], item['fiscal_qtr']))
    return Dataset.from_list(items)
