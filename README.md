# Is-it-easy-to-be-multilingual

## Abstract
###### Multi-lingual language models have redefined natural language processing in low-resource languages through cross-lingual transfer. By pre-training on multiple languages and fine- tuning for specific tasks, these models show- case potent zero-shot transfer capabilities, re- quiring minimal human-labeled data for pro- ficient task performance. This study intro- duces an interpretable statistical framework, systematically evaluating the impact of model- related factors while highlighting the crucial role played by syntactic, morphological, lexi- cal, and phonological similarities in predicting cross-lingual transfer performance. Offering nuanced insights into the determinants of suc- cessful cross-lingual transfer, this research pro- vides valuable guidance for optimizing multi- lingual language models across diverse linguis- tic contexts, facilitating robust natural language processing in low-resource settings.
    
### Question Answering
###### TyDiQA_data_DL.sh can be used to download the TyDiQA secondary (GoldP task) dataset.
###### QA_script.py contains the code necessary to fine-tune and test the mBERT model.

### Named Entity Recognition
###### PANX_NER_data_DL.sh can be used to download the PANX dataset and extract necessary files.
###### PANX_mBERT_script.py contains the code to fine-tune and test the mBERT model.

### Language Similarity Tasks
###### Syn_Phon_Morph_Similarities.ipynb contains the code needed to compute Syntactic, Phonological, and Morphological similarity of languages
###### lexical_sim.py contains the code needed to compute the lexical similarity of languages

### Regression
###### Regression_prep.ipynb collates all the required features for running regression models.
###### The Regression.rmd contains the regression models.


## Conclusion
###### This study delves into the examination of a pre- trained mBERT model to delve deeper into the mechanics of cross-lingual transfer. By conducting comprehensive model interpretation experiments across various language pairs and tasks, we’ve un- earthed significant insights. Our findings highlight the possibility of statistically modeling transfer through a select set of linguistic and data-derived features. Notably, we’ve established that the syn- tax, morphology, and phonology of languages serve as robust predictors of cross-lingual transfer, sur- passing the predictive capacity of lexical similar- ity between languages. Moreover, our analysis underscores the relevance of language model per- formance as a crucial indicator of cross-lingual prowess, presenting a readily available metric to better understand and facilitate cross-lingual trans- fer processes.

