# Final project for Advanced NLP

- Paper link: https://www.overleaf.com/read/gwnyxhfdnxtn#f462ee


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
