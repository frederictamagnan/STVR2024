# Generation of Regression Tests From Logs with Clustering Guided by Usage Patterns

This is the repository of an STVR Journal paper
## Installation



```bash
pip install -r requirements.txt
```

# Data

## data > raw
It contains the raw data of three of the datasets used in this paper

## data > datasets
The json files are cleansed datasets put in [Agilkia format](https://github.com/utting/agilkia)

# Implementation

Two abstract classes define the clustering pipeline and the sampling method and the methods they should implements
## Clustering pipeline class
```python
from abc import ABC, abstractmethod

class ClusteringPipeline(ABC):

    

    @abstractmethod
    def preprocessor(self,execution_traces_agilkia_format,**kwargs):
        pass
    
    @abstractmethod
    def fit_predict(self,preprocessed_execution_traces,k):
        pass
```
## Sample Heuristic class
```python
from abc import ABC, abstractmethod
class SampleHeuristic(ABC):


    @abstractmethod
    def tests_extraction(self,execution_traces_agilkia_format,cluster_labels):
        pass
```

## Example of Kmeans + W2V
The implementations of all the clustering pipelines are in the folder stvr.
Their filename start with "cp_..". In those files, you will find as well some sampling methods.


```python
from .clustering_pipeline import ClusteringPipeline
from .utils_preprocessing import traceset_to_textset
from sklearn.cluster import KMeans


import gensim
import numpy as np

class KmeansW2v(ClusteringPipeline):
    
    def __init__(self,dataset_name,filepath,**kwargs):
        self.dataset_name=dataset_name
        self.filepath=filepath
    def preprocessor(self, execution_traces_agilkia_format,**kwargs):
        listset=traceset_to_textset(execution_traces_agilkia_format,format='lst')
        model = gensim.models.Word2Vec(sentences=listset,vector_size=10,window=5,min_count=1)
        means=[]
        for seq in listset:
            vecs=[model.wv[elt] for elt in seq]
            means.append(np.mean(vecs,axis=0))
        X=np.array(means)
        
        return X

    def fit_predict(self, preprocessed_execution_traces,k):
        model=KMeans(n_clusters=k)
        y=model.fit_predict(preprocessed_execution_traces)
        self.model=model
        return  y

```

## Training of Embeddings
In stvr, all the files whose filename start with "x_" are related with the training of transformers or AE embeddings.

# Running the experiments
To run all the experiments you should use the task_id_definition.py script.
This script is originally designed to be run on a HPC center, but you can run it locally.

Pipelines to be run are defined in the first part of the script.

```python
pipelines+=[("BOW|Kmeans|RS",KmeansPipeline,Sampling,vae_dict)]
pipelines+=[("BOW|Kmeans|BUC",KmeansPipeline,BUC,vae_dict)]
pipelines+=[("BOW+POH|Kmeans|BUC",BowPlus,BUC,vae_dict)]
pipelines+=[("BOW+POH|Kmeans|RS",BowPlus,Sampling,vae_dict)]
pipelines+=[("TermFreq|Kmeans|RS",TfIdf,Sampling,vae_dict)]
pipelines+=[("TermFreq|Kmeans|BUC",TfIdf,BUC,vae_dict)]
```
In the config>config.json file, you will find the boundaries of the number of clusters explored, in this case, a number of cluster of 3 to 150 will be explored with a step of 5 
```json
    "femto_booking_agilkia_v6": {
        "datapath": "./data/datasets/",
        "range":[3,150,5]
```

```json
{"name_exp": "VAE|Kmeans|BUC", "cluster_nb": 7, "clustering_pipeline": "VAEEncodingKmeans", "sample_heuristic": "BUC", "pattern_coverage": [0.5557128120486899, 0.0019129236527896285], "pattern_coverage_raw_data": [0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5621346537377072, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5621346537377072, 0.5621346537377072, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5621346537377072, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536, 0.5551543910322536], "distance_experiments": [2.3714328038803396, 0.011673340607174126], "time": 151.26760005950928, "coverage_freq": 0.005, "dataset_name": "teaming_execution"}

```
After running the script, you will obtain json lines files for all the clustering_pipelines/cluster number you have set before. It will give you the UPC for each.

## License

[MIT](https://choosealicense.com/licenses/mit/)