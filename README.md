# Emowild: Building an In-the-Wild Non-Cognizant Image Emotion Dataset

**This repository contains the code to reproduce the EmoWild experiments discussed in the paper**

Estimating the perceived emotion to visual stimuli has gained significant traction in recent years. However, the data curated through subjective studies for this purpose are typically

1. (a) limited in scale and diversity (of both images and the subjects); and
2. (b) conducted in a controlled environment making the subjects cognizant and self-aware---thus, consequently diluting the objective of capturing relatively spontaneous emotions evoked across a multitude of personalities. 

As a result, previous emotion estimation models built on such non-cognizant datasets fail to capture the entire image information and thus the emotional context----which is necessary for generalization and handling of cross-content and cross-subject variety. We address these key gaps in the emotion recognition domain by building: 

    (i) a unique dataset corpus of 40k in-the-wild images EmoWild with continuous valence and arousal values from relatively subliminal comments of the users on social media, while analyzing and equitably accounting for the user persona; and 
    
    (ii) Cognitive Contextual Summarization model that takes the captions generated from the images and feeds them to a BERT network----for contextual understanding, that typically evokes human emotion---thereby enabling continuous emotion estimation. The EmoWild dataset augments the model's ability to generalize better on a variety of images and datasets.
    
Link to Dataset - https://drive.google.com/drive/folders/1-lOnTw0ztmc-JvqhuJ6cKvJ5cJqaZLyN?usp=sharing


# 📝 Instruction

1. Setup condo environment with latest pytroch version and python 3.6 for smooth execution
2. The code uses certain modules from tensorflow - please refer to requirements.txt for the same
3. Compile OFA modules and downlaod latest OFA checkpoint following the instructions from their github repo
4. Upload images to test to imageSamples directory and execute test.py
5. Sample commmand to execute with deafult options 
```
python test.py
```
