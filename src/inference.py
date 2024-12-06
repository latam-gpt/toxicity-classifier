import argparse
from multiprocessing import Manager, Process
from multiprocessing.managers import DictProxy

import torch
from datasets import Dataset, load_from_disk
from transformers import pipeline


def classify_toxicity(
    data_slice: Dataset,
    device_id: int,
    results_dict: DictProxy,
    idx: int,
    model_path: str,
    text_column: str,
) -> None:
    """
    Realiza inferencia de toxicidad en una partición del dataset usando una GPU
    específica.

    Args:
        data_slice (Dataset): Subconjunto del dataset a procesar.
        device_id (int): ID de la GPU asignada.
        results_dict (DictProxy): Diccionario compartido para almacenar los resultados.
        idx (int): Índice del proceso.
        model_path (str): Ruta del modelo de clasificación de toxicidad.
        text_column (str): Nombre de la columna de texto del dataset.
    """
    print(f"Inicializando proceso {idx} en dispositivo {device_id}")
    toxicity_classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=device_id,
        truncation=True,
        max_length=512,
    )
    textos = [
        text.replace("_usr", "").replace("_url", "") for text in data_slice[text_column]
    ]

    results = toxicity_classifier(textos, batch_size=16)
    labels = [res for res in results]

    results_dict[idx] = labels


def main() -> None:
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Clasificación de toxicidad en paralelo usando GPUs."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Ruta al modelo de clasificación de toxicidad.",
    )
    parser.add_argument(
        "--dataset_path", required=True, help="Ruta al dataset de entrada."
    )
    parser.add_argument(
        "--output_path", required=True, help="Ruta para guardar el dataset etiquetado."
    )
    parser.add_argument(
        "--text_column",
        required=True,
        help="Nombre de la columna de texto del dataset.",
    )
    args = parser.parse_args()

    model_path = args.model_path
    dataset_path = args.dataset_path
    output_path = args.output_path
    text_column = args.text_column

    dataset = load_from_disk(dataset_path)
    print("Dataset cargado")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No se encontraron GPUs disponibles.")

    shard_size = len(dataset) // num_gpus
    dataset_slices = [
        dataset.select(range(i * shard_size, (i + 1) * shard_size))
        for i in range(num_gpus)
    ]
    if len(dataset) % num_gpus != 0:
        dataset_slices[-1] = dataset.select(
            range((num_gpus - 1) * shard_size, len(dataset))
        )

    with Manager() as manager:
        results_dict = manager.dict()
        processes = []

        for i in range(num_gpus):
            process = Process(
                target=classify_toxicity,
                args=(dataset_slices[i], i, results_dict, i, model_path, text_column),
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        final_labels = []
        for i in range(num_gpus):
            final_labels.extend(results_dict[i])

        final_dataset = Dataset.from_dict(
            {"texto": dataset["texto"], "toxicity_label": final_labels}
        )

        final_dataset.save_to_disk(output_path)
        print(f"Dataset etiquetado guardado en: {output_path}")


if __name__ == "__main__":
    main()
