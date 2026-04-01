## Implementation Notes
**Modified by**: Coco Sittardt

**Date**: April 2026

**Changes**:   
Replacing the Word-level Graph with a Sentence-Level Graph using sentence embeddings via CLIP. Goal is 
to adapt and extend the existing implementation, while keeping the rest of the pipeline unchanged.

**Based on original Work**:  
https://github.com/JaTrev/unsupervised_graph-based
and the paper "Unsupervised Graph-based Topic Modeling from Video Transcriptions"
by Jason Thies, Lukas Stappen, Gerhard Hagerer, Björn W. Schuller, and Georg Groh.

## Notes
I'm using CLIP `ViT-B/32` for the sentence embeddings, as I feel that the intended purpose of CLIP (encoding not only text but also images)
represents the original idea of the paper. The implementation of creating a topic extractor for video transcripts could be enhanced 
when connecting the modalities of text and image. 

### Issues
**Sentences like these are too logn for context length:**
RuntimeError: Input Absolute animal technology in this car , the cameras the reversing camera captures underneath the wing 
Burr in the front gives you a bird's eye point of view of this car , and you can watch yourself driving along like Grand Theft Auto in the olden days , 
Um , but this car then moves into the sporty loathe even that we have got a way that it just accelerated from the line . is too long for context length 77


---

### Unsupervised Graph-based Topic Modeling from Video Transcriptions
This is the repository of the paper
"Unsupervised Graph-based Topic Modeling from Video Transcriptions"
by Jason Thies, Lukas Stappen, Gerhard Hagerer, Björn W. Schuller, and Georg Groh.

In this paper,  we aim at developing a topic extractor on video transcriptions. 
The model improves coherence by exploiting neural word embeddings through a graph-based clustering method. 
Unlike typical topic models, this approach works without knowing the true number of topics. 
Experimental results on the real-life multimodal dataset MuSe-CaR demonstrates that our approach extracts coherent and 
meaningful topics, outperforming baseline methods. 
Furthermore, we successfully demonstrate the generalisability of our approach on a pure text review dataset.



Overview of this repository:
- visuals:
      This folder contains all graphs and scores from the topic models.

- src:
      This folder contains all the python source code for the study,
      use the requirements file to download all necessary libraries.

- data:
    This folder includes the training data set (including the labels) of MuSe - CaR 
    as well as the CitySearch Car Review data set (training and test set) from 
    ([http://www.cs.cmu.edu/~mehrbod/RR/][http://www.cs.cmu.edu/~mehrbod/RR/]). 
    All existing pre-calculated models are in this folder.
  


Installation Instructions:


1. Clone Repository:
    git clone ...


2. Create virtual environment (this project runs on Python 3.6):
    conda create --name unsupervised_graph-based python=3.6


3. Activate virtual environment:
    conda activate unsupervised_graph-based


3. Fetch requirements:
    pip3 install -r requirements.txt


4. run main.py:
    python main.py --data_set XX --tm YY


Arguments:
- (--data_set) is used to select the preprocessed data set:
    MuSe-CaR: MUSE
    Citysearch Corpus: CRR

- (--tm) is used to set the topic model:
    Clustering-Based Baselines: TVS
    Graph-based Clustering (using K-Components): k-components