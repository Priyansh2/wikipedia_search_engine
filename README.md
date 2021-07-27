# Wikipedia Search Engine
* Designed and developed a search system using Python and Spacy to process a user-given query within 1s
* Scaled the project with SAX parsing and multi-level indexing. Reduced page retrieval time by 40% with binary search
* Built a search module for retrieval and ranking (top-10) of relevant wiki titles using vector space model with TF-IDF weighting scheme



# Requirements
- Python >=3.7.3
- NLTK (```pip install --user -U nltk```)
- Spacy and pre-trained models for English Language (```pip install -U spacy && pip install -U spacy-lookups-data && python -m spacy download en_core_web_sm```)
- Sorted Containers (```pip install sortedcontainers```)
- FTFY (```pip install ftfy```)
- Unidecode (```pip install Unidecode```)
- Dill (```pip install dill```)

# Files
- ```index```: global index created by running ```index.py```
- ```index.py```: Index creation code which takes wikipedia dump (```./enwiki_sample``` for eg) and index folder (```./index``` for eg)
- ```search.py```: Search Operation which takes index folder (```./index``` for eg), queryfile (```./queryfile``` for eg) and outputfile (```./resultfile``` for eg)
- ```enwiki_sample```: Wiki dump sample (100MB). 
- ```queryfile```: Input query file
- ```resultfile```: Search outputs file

# Note
1. Index files for original wiki dump (75 GB) can be downloaded: https://iiitaphyd-my.sharepoint.com/:f:/g/personal/priyansh_agrawal_research_iiit_ac_in/EgpyrelWNKBFsysv_TMF_Y8BG34ltJgL6Vz-iJ4_97E53g?e=LTMwGu.  
