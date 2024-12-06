import argparse
import json
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_data(
    json_path: str, threshold: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga y procesa los datos desde un archivo JSON para preparar
    los DataFrames de entrenamiento, validación y prueba.

    Args:
        json_path (str): Ruta al archivo JSON con los datos.
        threshold (int): Threshold para considerar un texto como tóxico.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames de
        entrenamiento, validación y prueba.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = list(map(lambda x: x["original_text"], data))
    labels = list(
        map(
            lambda x: 1
            if x["moderation_result"]["moderationCategories"][0]["confidence"]
            > threshold
            else 0,
            data,
        )
    )

    df = pd.DataFrame({"text": texts, "label": labels})

    train_df, test_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        test_df, test_size=0.5, stratify=test_df["label"], random_state=42
    )

    return train_df, val_df, test_df


def undersample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Submuestrea para balancear las clases en un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con las columnas `text` y `label`.

    Returns:
        pd.DataFrame: DataFrame balanceado.
    """
    class_0 = df[df["label"] == 0]
    class_1 = df[df["label"] == 1]

    if len(class_0) > len(class_1):
        class_0 = resample(
            class_0, replace=False, n_samples=len(class_1), random_state=42
        )
    else:
        class_1 = resample(
            class_1, replace=False, n_samples=len(class_0), random_state=42
        )

    balanced_df = (
        pd.concat([class_0, class_1])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    return balanced_df


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """
    Calcula métricas de evaluación para el modelo.

    Args:
        eval_pred (Any): Predicciones y etiquetas verdaderas.

    Returns:
        Dict[str, float]: Métricas calculadas (accuracy, precision, recall, f1).
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    """
    Función principal del script.
    """
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelo de clasificación de toxicidad."
    )
    parser.add_argument(
        "--json_path", required=True, help="Ruta al archivo JSON con los datos."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Ruta para guardar los resultados del modelo.",
    )
    parser.add_argument(
        "--logging_dir",
        required=True,
        help="Ruta para guardar los logs del entrenamiento.",
    )
    parser.add_argument(
        "--threshold",
        required=True,
        help="Threshold para considerar un texto como tóxico.",
    )
    parser.add_argument("--model_name", required=True, help="Modelo a finetunear.")
    args = parser.parse_args()

    json_path = args.json_path
    output_dir = args.output_dir
    logging_dir = args.logging_dir
    threshold = float(args.threshold)
    model_name = args.model_name

    train_df, val_df, test_df = load_data(json_path, threshold)
    balanced_train_df = undersample_dataframe(train_df)
    balanced_val_df = undersample_dataframe(val_df)
    balanced_test_df = undersample_dataframe(test_df)

    train_dataset = Dataset.from_pandas(balanced_train_df)
    val_dataset = Dataset.from_pandas(balanced_val_df)
    test_dataset = Dataset.from_pandas(balanced_test_df)

    dataset = DatasetDict(
        {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(
        examples: Dict[str, str]
    ) -> Dict[str, Union[List[int], List[str]]]:
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate()
    print(results)


if __name__ == "__main__":
    main()
