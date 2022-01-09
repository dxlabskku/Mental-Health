# A lexicon-based approach to examine depression detection in social media: the case of Twitter and university community

We provide the depression labeled dataset on Twitter posts consist of multiple languages(Korean, English and Japanese). The datasets were collected using Twitter APIs with community-based random sampling approaches and our datasets consist of 921k tweets from Korean users, 10M tweets from English users and 15M tweets from Japanese users. Each language depression dataset was labeld depression(1) or non-depression(0) with depression lexicon, which was collected from prior studies that related to the detection of depression on social media.
In addition, we applied our model to specific group(e.g. university community) to detect not only general social media posts but also specific groups posts.

## Workflow of our study
![new_model2](https://user-images.githubusercontent.com/96400041/148496605-79c7e029-bfcf-4df0-8986-88400a387ee3.jpg)
**Figure : From data collcetion to classification**


## Data
Our sampled twitter datasets are in **data** folder as gitHub limits the size of files allowed in repositories. We are only allowed to distribute the data for the research purpose, if you want to achieve full datasets, please complete the request form(update later).

## Experiment
We employed deep learning framework to classify depression posts in the social media and university community. Within the models folder we uploaded binary classification models for each language. We achived ranges form 99.39% to 99.66% f1-score for each language in detecting depression posts on general social media and 64.51% f1-score for university community in South Korea.

Within the **model** folder we uploaded classificatino models to detect depression posts.
