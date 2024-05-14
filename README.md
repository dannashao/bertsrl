# Checklisting the Fine-tured BERT for Semantic Role Labeling

- Thesis link: https://www.overleaf.com/read/gwnyxhfdnxtn#f462ee

This repo uses the Checklist approach to evaluate the fine-tuned BERT models for SRL. For details on how the BERT models are fine-tuned, see [this notebook](https://github.com/dannashao/portfolio-NLP/blob/main/SRL/Fine%20tune%20BERT%20for%20SRL.ipynb)

## Contents
### Folders
- data/: The original PropBank data
- tests/: The 5 Challenge sets
### Python files
- baseline_ds, advanced_ds, read_and_preprocess, utils: util functions for notebooks.
### Notebooks
- Evaluation: Actual evaluation using the challenge sets on the models.
- Test14_Generate: Demonstrate how to use [Checklist](https://github.com/marcotcr/checklist/tree/master) to generate test instances.

## Usage
Directly run the Evaluation.ipynb. No extra downloading required (the models are downloaded within the notebook using transformer's `AutoModelForTokenClassification.from_pretrained()` from [huggingface](https://huggingface.co/dannashao).)
