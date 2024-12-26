import argparse

import torch
from datasets import load_from_disk
from transformers import pipeline


def classify_toxicity(
    batch: dict, model_path: str, device_id: int, text_column: str, batch_size: int
) -> dict:
    """
    Realiza inferencia sobre un batch de datos utilizando una pipeline de Hugging Face.

    Args:
        batch (dict): Diccionario que contiene un batch del dataset.
        model_path (str): Ruta al modelo preentrenado.
        device_id (int): ID de la GPU a usar.
        text_column (str): Nombre de la columna que contiene los textos.
        batch_size (int): Tamaño del batch.

    Returns:
        dict: Diccionario con las etiquetas de toxicidad y las puntuaciones.
    """
    toxicity_classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=device_id,
        truncation=True,
        max_length=512,
    )
    texts = batch[text_column]
    results = toxicity_classifier(texts, batch_size=batch_size)
    labels = [res["label"] for res in results]
    scores = [res["score"] for res in results]
    return {"toxicity_label": labels, "toxicity_score": scores}


def process_batch(
    batch: dict,
    model_path: str,
    text_column: str,
    batch_size: int,
    process_index: int,
    num_gpus: int,
) -> dict:
    """
    Procesa un batch de datos asignando una GPU en función del índice del proceso.

    Args:
        batch (dict): Diccionario que contiene un batch del dataset.
        model_path (str): Ruta al modelo preentrenado.
        text_column (str): Nombre de la columna que contiene los textos.
        batch_size (int): Tamaño del batch.
        process_index (int): Índice del proceso actual.
        num_gpus (int): Número de GPUs disponibles.

    Returns:
        dict: Diccionario con las etiquetas de toxicidad y las puntuaciones.
    """
    device_id = process_index % num_gpus
    return classify_toxicity(batch, model_path, device_id, text_column, batch_size)


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Clasificación de toxicidad en paralelo usando GPUs."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Ruta al modelo de clasificación de toxicidad.",
    )
    parser.add_argument(
        "--dataset_path", required=True, help="Ruta al dataset por etiquetar."
    )
    parser.add_argument(
        "--output_path", required=True, help="Ruta para guardar el dataset etiquetado."
    )
    parser.add_argument(
        "--text_column",
        required=True,
        help="Nombre de la columna de texto del dataset.",
    )
    parser.add_argument(
        "--batch_size",
        required=True,
        help="Tamaño del batch.",
    )
    parser.add_argument(
        "--procs_per_gpu",
        type=int,
        default=2,
        help="Número de procesos por GPU.",
    )
    args = parser.parse_args()

    model_path = args.model_path
    dataset_path = args.dataset_path
    output_path = args.output_path
    text_column = args.text_column
    batch_size = int(args.batch_size)
    procs_per_gpu = int(args.procs_per_gpu)

    dataset = load_from_disk(dataset_path)
    print("Dataset cargado")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No se encontraron GPUs disponibles.")

    total_procs = num_gpus * procs_per_gpu

    results = dataset.map(
        lambda batch, process_index: process_batch(
            batch, model_path, text_column, batch_size, process_index, num_gpus
        ),
        batched=True,
        batch_size=batch_size,
        num_proc=total_procs,
        with_rank=True,
    )

    results.save_to_disk(output_path)
    print(f"Dataset etiquetado guardado en: {output_path}")


if __name__ == "__main__":
    main()
