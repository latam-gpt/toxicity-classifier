import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

from datasets import Dataset, load_from_disk
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
API_URL = "https://language.googleapis.com/v1/documents:moderateText"
BATCH_SIZE = 100


def create_authorized_session(
    service_account_file: str, scopes: List[str]
) -> AuthorizedSession:
    """Crea una sesión autorizada con las credenciales proporcionadas.

    Args:
        service_account_file (str): Ruta al archivo de credenciales del servicio.
        scopes (List[str]): Lista de alcances para la sesión.

    Returns:
        AuthorizedSession: Sesión autorizada para realizar solicitudes a la API.
    """
    credentials = Credentials.from_service_account_file(
        service_account_file, scopes=scopes
    )
    return AuthorizedSession(credentials)


def load_results(
    output_file: str,
) -> Tuple[List[Dict[str, Union[str, Dict]]], Set[str]]:
    """Carga resultados previamente procesados desde un archivo JSON.

    Args:
        output_file (str): Ruta al archivo JSON con los resultados.

    Returns:
        Tuple[List[Dict[str, Union[str, Dict]]], Set[str]]:
        Resultados cargados y textos ya procesados.
    """
    if Path(output_file).exists():
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        processed_texts = {r["original_text"] for r in results}
    else:
        results = []
        processed_texts = set()
    return results, processed_texts


def clean_text(example: Dict[str, str], text_column: str) -> Dict[str, str]:
    """Limpia el texto eliminando patrones específicos.

    Args:
        example (Dict[str, str]): Diccionario con el texto a limpiar.
        text_column (str): Nombre de la columna de texto.

    Returns:
        Dict[str, str]: Diccionario con el texto limpio.
    """
    example[text_column] = example[text_column].replace("_url", "").replace("_usr", "")
    return example


def infer_text(session: AuthorizedSession, text: str) -> Union[Dict, None]:
    """Envía una solicitud de inferencia a GCP.

    Args:
        session (AuthorizedSession): Sesión autorizada para realizar la solicitud.
        text (str): Texto a analizar.

    Returns:
        Union[Dict, None]: Resultado de la inferencia o `None` en caso de error.
    """
    payload = {
        "document": {
            "type": "PLAIN_TEXT",
            "content": text,
            "language": "es",
        }
    }
    response = session.post(
        API_URL,
        headers={"Content-Type": "application/json; charset=utf-8"},
        json=payload,
    )
    if response.status_code == 200:
        return response.json()
    else:
        print(
            f"Error en la solicitud (status: {response.status_code}): {response.json()}"
        )
        return None


def process_batch(
    batch: Dataset,
    processed_texts: Set[str],
    session: AuthorizedSession,
    text_column: str,
) -> List[Dict[str, Union[str, Dict]]]:
    """Procesa un batch de textos, realizando inferencias y almacenando los resultados.

    Args:
        batch (Dataset): Batch de textos a procesar.
        processed_texts (Set[str]): Conjunto de textos ya procesados.
        session (AuthorizedSession): Sesión autorizada para realizar la solicitud.
        text_column (str): Nombre de la columna de texto.

    Returns:
        List[Dict[str, Union[str, Dict]]]: Resultados procesados del batch.
    """
    batch_results = []
    for text in batch[text_column]:
        if text in processed_texts:
            print(f"Texto ya procesado: {text}")
            continue
        result = infer_text(session, text)
        if result:
            batch_results.append({"original_text": text, "moderation_result": result})
            processed_texts.add(text)
    return batch_results


def save_results(output_file: str, results: List[Dict[str, Union[str, Dict]]]) -> None:
    """Guarda los resultados en un archivo JSON.

    Args:
        output_file (str): Ruta al archivo de salida.
        results (List[Dict[str, Union[str, Dict]]]): Resultados a guardar.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Procesa inferencias de texto utilizando la API de Google."
    )
    parser.add_argument(
        "--service_account_file",
        required=True,
        help="Ruta al archivo de credenciales del servicio.",
    )
    parser.add_argument(
        "--dataset_path", required=True, help="Ruta al dataset de entrada."
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Ruta al archivo donde se guardarán los resultados.",
    )
    parser.add_argument(
        "--text_column",
        required=True,
        help="Nombre de la columna de texto en el dataset.",
    )
    args = parser.parse_args()

    session = create_authorized_session(args.service_account_file, SCOPES)

    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.map(lambda x: clean_text(x, args.text_column))

    results, processed_texts = load_results(args.output_file)

    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i : i + BATCH_SIZE]
        print(
            f"Procesando batch {i // BATCH_SIZE + 1} "
            f"de {len(dataset) // BATCH_SIZE + 1}..."
        )
        batch_results = process_batch(batch, processed_texts, session, args.text_column)

        if batch_results:
            results.extend(batch_results)
            save_results(args.output_file, results)
            print(f"{len(batch_results)} textos procesados y guardados.")

    print(
        f"Inferencia completada. Resultados finales guardados en '{args.output_file}'."
    )


if __name__ == "__main__":
    main()
