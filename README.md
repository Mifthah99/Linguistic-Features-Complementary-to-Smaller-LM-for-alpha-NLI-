BASELINE CODE:

Training:

To run training and create a new model: 	python train_anli.py 

To run training to update existing model:	python train_anli.py model_name.pkl

if you want to change number of epochs please use epochs as cmd line arg, ex :python train_anli.py model_name.pkl epochs=20


Evaluation:

If you don't want to train just evaluate:	python evaluate.py

if you want to evaluate specific model:		python evaluate model_name.pkl




ADVANCED CODE:
Training: 
To run training and create new model:   python HYPO_LLM.py train

Evaluate:
If you don't want to train just evaluate: python HYPO_LLM.py



---

# CL_Lab_Project
## Repository for CL Team Lab Project

### Team Members:

Muhammmed Mifthah

Manuprasrad Mani

**Project: Natural Language Inference**
---
### Progress and Planned Schedule:

*__Week 1 (7/4-14/4):__*

* Choosing task (Natural Language Inference)

* Created Github Repository

*__Week 2 (14/04-21/04):__* 

* Created scripts for Splitting training data into train and test (Test data not provided and labels for test data not available online)(Mifthah)

* Created evaluation Scripts (Kept both accuracy and F1 Score)(Manuprasad)


*__Week 3 (21/4-28/4):__* 

* Data exploration, Preprocessing and Feature Extraction (Plan)

    * Implemented Perceptron (Mifthah)
    * Script to convert data to pandas format (Manuprasad)


*__Week 4 (28/4-05/5) (Plan):__* 

*  Feature extraction
  
   * Script to create BOW vector (Mifthah)
   * Script for additional features like similarity scores, sentiment polarity, word overlap etc. (Manuprasad)

* Perceptron Implementation and Testing
  
  * Made changes to perceptron code to make use of generator because of memory error (Mifthah)
  * Trained perceptron on just BOW vector and tested on dev (Accuracy 50.9%)


*__Week 5 (05/5-12/5) (Plan):__* 

* Parameter Tuning and Refining Model
* Creating presentation
* Planning Advanced Methods


*__Week 6 (12/5-19/5) (Plan):__* 

* Final Changes and Preparation for submission

*__Week 7 (19/5-26/5) :__*

    Finalized presentation and prepared next steps

    Submitted presentation and Pr

*__Week 8 (26/5-02/6) :__*

    Sentence Transformer for feature vector for perceptron (Mifthah)
    Word to Vector representation as feature vector for perceptron (Manuprasad)

*__Week 9-10 (02/6 - 16/6):__*

    LLM Implementation and fine-tuning (Mifthah)

*__Week 11-12 (16-6 - 30/6):__*

    LLM Feature Extraction and analysis (Mifthah)
        Used a slightly larger model got an improvement in dev result to 67.35%
    Trying out Sentence transformer to see if change in embeddings helps improve performance (Manupasad)

*__Week 13-14 (30/6 - 14/7):__*

    Update LLM and Improve Performance, tryout topicBert(mifthah)
    Report writing(Mifthah and Manuprasad)



