DynamicWord2Vec
===============

Paper title:
**Dynamic Word Embeddings for Evolving Semantic Discovery.**

Paper links:
https://dl.acm.org/citation.cfm?id=3159703
https://arxiv.org/abs/1703.00607

## Workflow

1. Build trainings data
   - input: raw sentences?
   - use `NLTK`/`gensim`? to compute cooccurrence counts for some window size
   - store those in `wordCoOccurByYear.csv` and word-freqencies in `wordIDHash.csv`
   - use `*.py` (**WIP**) to compute `PMI` (or other metric) values
   - use `prepare_static_embeddings.py` to generate eigenvalue embeddings

2. Training
   - use `train_time_CD_smallnyt` with correct params

3. Plot things
   - `find_extreme_words.py` and others to find most-changed words
   - `plot_norms.py` to check & `plot_trajectories.py` to show fancy things
   - use **WIP**`.py` to get rank word list differences for a word


## Files:

_NOTE: not all files have been genericified!_  
_NOTE: data will mostly be stored in MATLAB dictionary (embedding) files and CSV files! This will be rewritten to use numpy arrays._  
_NOTE: Much has been rewritten to use cmd parameters but not everything._  
_NOTE: currently best used for continuous yearly data, other time intervals require slight modifications of code!_  

`/embeddings`
 - embeddings in loadable MATLAB files. 0 corresponds to 1990, 1 to 1991, ..., 19 to 2009.
 To save space, each year's embedding is saved separately. When used in visualization code, first merge to 1 embedding file.
 
`/train_model`
 - contains code used for training our embeddings
 - data file download: https://www.dropbox.com/s/nifi5nj1oj0fu2i/data.zip?dl=0 (4,0 GB)
 
    `/train_model/train_timeCD.py`
     - main training script
     - needs `wordPairPMI_%d.csv` data files, optionally `emb_static.mat` with initial static eigenvalue embedding

    `/train_model/util_timeCD.py`
     - containing helper functions and update functions
       - update function, eq. 8 from paper
       - reading and writing of CSV PMI data files and embeddings

    `/train_model/util_shared.py`
     - containing helper functions for printing

    `/train_model/build_pmis.py`
     - currently private, uses preprocessed DB data to build data files
     - used to generate `wordlist.txt` & `wordIDHash.csv` and `wordCoOccurByYear.csv`
     - also generates `wordPairPMI_%d.csv`
     - TODO: rewrite PMI computation to use `wordCoOccurByYear.csv` data

    `/train_model/prepare_static_embeddings.py`
     - rewritten code of (following) `form_cum_coocur_matrix.py` and `get_best_static_eig.m`

    `/train_model/data/form_cum_coocur_matrix.py` (extracted from `data.zip`)
     - uses a magic `wordCoOccurByYear_min200_ws5.csv` file with cooccurrence values for word-pairs (probably minimum frequency of 200 and window size of 5), cooccurrence columns for each year
     - file `wordIDHash.csv` with w_id, word and probably total number of occurrences in text
     - script to transform data for static eigenvalue computation
     - `wordCoOccurByYear` may have also been used to compute **PMI** values for training

    `/train_model/data/get_best_static_eig.m` (extracted from `data.zip`)
     - MATLAB code
     - compute eigenvalue initial static embeddings for training
     - uses cumulative yearly cooccurrence matrix

`/other_embeddings`
 - contains code for training baseline embeddings
 - data file download: https://www.dropbox.com/s/tzkaoagzxuxtwqs/data.zip?dl=0 (2,3 GB)
 
   `/other_embeddings/staticw2v.py`
    - static word2vec (Mikolov et al 2013)
    - not really code for execution - lazy lazy...
    
   `/other_embeddings/aw2v.py`
    - aligned word2vec (Hamilton, Leskovec, Jufarsky 2016)
    - using procustes
    
   `/other_embeddings/tw2v.py`
    - transformed word2vec (Kulkarni, Al-Rfou, Perozzi, Skiena 2015)
    - KDTrees?
    
`/visualization`
 - scripts for visualizations in paper

   `/visualizations/find_extreme_words.py`
    - compute normalized embeddings and compute distance between each year for all words
    - output words with most/least distance
 
   `/visualization/plot_norms.py`
    - changepoint detection figures
    - compute norms for given words and plot
    
   `/visualization/plot_trajectories.py`
    - trajectory figures
    - for a given word, collect nearest neighbors (cosinus-distance) and project into 2D with t-SNE
    - for trajectory do some more magic to filter words by distance etc.
    
`/distorted_smallNYT`
 - code for robust experiment
 - data file download: https://www.dropbox.com/s/6q5jhhmxdmc8n1e/data.zip?dl=0 (8.7 GB)
 
`/misc`
 - contains general statistics and word hash file
 - `hash` as in number of occurrences
