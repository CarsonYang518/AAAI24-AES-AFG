# Unveiling the Tapestry of Automated Essay Scoring: A Comprehensive Investigation of Accuracy, Fairness, and Generalizability
These are the implemented codes for our paper **Unveiling the Tapestry of Automated Essay Scoring: A Comprehensive Investigation of Accuracy, Fairness, and Generalizability** ([Full Paper Here](https://arxiv.org/abs/2401.05655)), which has been accepted as a full paper on AAAI 2024.  

# Dataset
The dataset we used can be access [here](https://github.com/scrosseye/persuade_corpus_2.0).  

# Description
<kbd>feature_generation.ipynb</kbd> contains functions for generating hand-crafted features.  
<kbd>Full_features_SVMipynb</kbd> contains methods in the [Paper](https://aclanthology.org/W15-0626/).  
<kbd>CNN_LSTM_ATT_FINAL.ipynb</kbd> contains methods in the [Paper](https://aclanthology.org/K17-1017/).  
<kbd>SKIPFLOW_LSTM.ipynb</kbd> contains methods in the [Paper](https://arxiv.org/abs/1711.04981).  
<kbd>BERT_Regression_Ranking.ipynb</kbd> contains methods in the [Paper](https://aclanthology.org/2020.findings-emnlp.141/).  
<kbd>BERT_3layer.ipynb</kbd> contains methods in the [Paper](https://arxiv.org/abs/1909.09482).  
<kbd>Reduced_features_SVM.ipynb</kbd> contains methods in the [Paper](https://aclanthology.org/W15-0626/).  
<kbd>PAES.ipynb</kbd> contains methods in the [Paper](https://arxiv.org/abs/2008.01441).  
<kbd>TDNN.ipynb</kbd> contains methods in the [Paper](https://aclanthology.org/P18-1100/).  
Note: For **RankSVM** (also the first step in **TDNN**), please refer to the implementation [here](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html). 

# Version
- Python 3.11
- PyTorch 2.1.0
- TensorFlow 2.15.0
- scikit-learn 1.4.1
- NLTK 3.8.1
- spaCy 3.7
- RSMTool 11.3.0
- Stanza 1.5.0

# Reference
If you use our code in a publication, we would appreciate citations:
```
@inproceedings{yang2024unveiling,
  title={Unveiling the Tapestry of Automated Essay Scoring: A Comprehensive Investigation of Accuracy, Fairness, and Generalizability},
  author={Yang, Kaixun and Rakovi{\'c}, Mladen and Li, Yuyang and Guan, Quanlong and Ga{\v{s}}evi{\'c}, Dragan and Chen, Guangliang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={20},
  pages={22466--22474},
  year={2024}
}

```


