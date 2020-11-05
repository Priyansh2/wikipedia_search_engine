# Wikipedia Search Engine
Designed a scalable and efficient search engine in Python to query a Wikipedia corpus of ~75GB with a response time of 1s and outputs the top 10 relevant documents based on the search query.

# Problem Description
- The project involves building a search engine on the Wikipedia Data Dump without using any external index. For this project, data dump of size ~75 GB is used to create a global index upon which search operation is conducted based on given query. 
- The creation of index must be done in a efficient and scalable manner i.e, on a dump of size ~100 MB (~0.126% of target: 75 GB) the index creation time must be <= 150sec in Python environment. Moreover, index size should be less than 1/4 (< 25% of 75GB) of dump size.
- The search operation should return results in realtime with a max limit of 1-2 sec per query. 
- The search results which should be displayed are the top 10 relevant documents (wiki articles titles) for each input query.
- The input query can be either normal-query ('free word' query or query consists of only words) or field-query where fields include Title, Infobox, Body, Category, Links and References of a wikipedia page. 

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
