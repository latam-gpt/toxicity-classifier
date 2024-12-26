# Toxicity Classifier
When training LLMs, it is very important to make sure that you don't have toxic content within your dataset. For that reason, we developed this repo for fine tuning toxicity classifiers with your custom datasets.

This repo has 3 main modules:

- A data labeling script, that uses the GCP Cloud Natural Language API to rate a subsample of the dataset
- A fine tuning script that takes the annotated data and fine tunes a classifier
- An inference script, that takes a trained model and performs distributed inference over a dataset

## Installation
Create a python environment and run
```bash
pip install -r requirements/base.txt
```

## Usage
As mentioned before, this repo has 3 main modules.

### GCP-based data labeling
While trying diferent toxicity classifiers, we realized that the one provided by GCP was among the bests. For that reason, the script `gcp_data_labeling.py` uses the `Cloud Natural Language API`. In order to use it, you have to enable that API in your GCP project, and create a service account with a JSON credentials file. Once you've done that, you can run the script like this:

```bash
python gcp_data_labeling.py --service_account_file <path_to_service_account.json> \
                            --dataset_path <path_to_dataset> \
                            --output_file <output_file_path> \
                            --text_column <text_column_name>
```

That will output a JSON file with annotated data, that you can later use as input for the fine tuning script.

### Fine tuning
Once you've got your data labeled, it's time for the fine tuning of the base model. To do that, you can run the `finetune.py` file with the following command:

```bash
python finetune.py --json_path <path_to_labeled_data.json> \
                   --output_dir <output_directory> \
                   --logging_dir <logging_directory> \
                   --threshold <quelity_threshold> \
                   --model_name <hf_model_name>
```

### Inference
Finally, for distributed inference over your entire dataset, you have the `inference.py` file, which you can run like this:

```bash
python inference.py --model_path <path_to_model> \
                    --dataset_path <path_to_dataset> \
                    --output_path <output_file_path> \
                    --text_column <text_column_name> \
                    --batch_size <batch_size> \
                    --procs_per_gpu <procs_per_gpu>
```

Getting a new dataset with your model's predictions.
