import os
import glob
import pandas as pd
from fitparse import FitFile
from fitparse.utils import FitCRCError
from tqdm import tqdm

RECORDS_FILE = "workouts_per_second.csv"
SESSIONS_FILE = "session_records.csv"
PROCESSED_FILES_LOG = "processed_files.csv"


def load_processed_files(log_path: str) -> pd.DataFrame:
    """
    Load the log of already processed files.

    Parameters
    ----------
    log_path : str
        Path to the processed-files log CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns "file_name" and "crc_error".
        Empty if the log does not exist yet.
    """
    if os.path.exists(log_path):
        return pd.read_csv(log_path)
    return pd.DataFrame(columns=["file_name", "crc_error"])


def parse_fit_file(file_path: str) -> tuple[list[dict], list[dict], bool]:
    """
    Parse a single FIT file, tolerating CRC errors.

    The file is first read with CRC validation. If a CRC error occurs,
    the file is read again without validation so its data can still be used.

    Parameters
    ----------
    file_path : str
        Path to the .fit file.

    Returns
    -------
    tuple[list[dict], list[dict], bool]
        (record_rows, session_rows, had_crc_error)

    Raises
    ------
    Exception
        If the file cannot be parsed even without CRC validation.
    """
    try:
        records, sessions = _extract_messages(file_path, check_crc=True)
        return records, sessions, False
    except FitCRCError:
        records, sessions = _extract_messages(file_path, check_crc=False)
        return records, sessions, True


def _extract_messages(file_path: str, check_crc: bool) -> tuple[list[dict], list[dict]]:
    """
    Extract "record" and "session" messages from a FIT file.

    Parameters
    ----------
    file_path : str
        Path to the .fit file.
    check_crc : bool
        Whether to validate the CRC checksum.

    Returns
    -------
    tuple[list[dict], list[dict]]
        (record_rows, session_rows)
    """
    file_name = os.path.basename(file_path)
    record_rows: list[dict] = []
    session_rows: list[dict] = []

    fitfile = FitFile(file_path, check_crc=check_crc)

    for message in fitfile.get_messages():
        row = {"source_file": file_name, "message_type": message.name}
        row.update({field.name: field.value for field in message})

        if message.name == "record":
            record_rows.append(row)
        elif message.name == "session":
            session_rows.append(row)

    return record_rows, session_rows


def append_to_csv(rows: list[dict], csv_path: str) -> None:
    """
    Append rows to a CSV file, aligning columns with existing data.

    Parameters
    ----------
    rows : list[dict]
        Rows to append. Nothing happens if the list is empty.
    csv_path : str
        Path to the target CSV file.
    """
    if not rows:
        return

    new_df = pd.DataFrame(rows)

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path, low_memory=False)
        new_df = pd.concat([existing_df, new_df], ignore_index=True)

    new_df.to_csv(csv_path, index=False)


def convert_fit_dir_to_csv(dir_path: str, output_dir: str = "./Dataframes") -> None:
    """
    Read new .fit files in a directory and append them to two CSV files.

    Two types of data are distinguished:

    - "record" messages: second-by-second workout data
      (e.g. heart rate, speed, position at each timestamp).
    - "session" messages: summary information about the whole activity.

    Files that were already processed in a previous run (tracked in
    processed_files.csv) are skipped. Files with a CRC error are still
    loaded, but listed in a summary at the end.

    Parameters
    ----------
    dir_path : str
        Path to the directory containing the .fit files.
    output_dir : str, optional
        Directory for the resulting CSV files. Created if missing.
        Default: "./Dataframes".

    Returns
    -------
    None
        Results are saved as CSV files.

    Examples
    --------
    >>> convert_fit_dir_to_csv("./Source_Data/Fitnessdaten")
    """
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, PROCESSED_FILES_LOG)
    processed_df = load_processed_files(log_path)
    processed_names = set(processed_df["file_name"])

    all_files = glob.glob(os.path.join(dir_path, "*.fit"))
    new_files = [f for f in all_files if os.path.basename(f) not in processed_names]

    print(f"{len(all_files)} files found, {len(new_files)} new, "
          f"{len(all_files) - len(new_files)} already processed.")

    new_records: list[dict] = []
    new_sessions: list[dict] = []
    new_log_entries: list[dict] = []
    crc_error_files: list[str] = []
    failed_files: list[tuple[str, str]] = []

    for file_path in tqdm(new_files, desc="Processing FIT files"):
        file_name = os.path.basename(file_path)
        try:
            records, sessions, had_crc_error = parse_fit_file(file_path)
        except Exception as e:
            # Not logged as processed, so it is retried on the next run
            failed_files.append((file_name, str(e)))
            continue

        new_records.extend(records)
        new_sessions.extend(sessions)
        new_log_entries.append({"file_name": file_name, "crc_error": had_crc_error})
        if had_crc_error:
            crc_error_files.append(file_name)

    append_to_csv(new_records, os.path.join(output_dir, RECORDS_FILE))
    append_to_csv(new_sessions, os.path.join(output_dir, SESSIONS_FILE))

    if new_log_entries:
        updated_log = pd.concat(
            [processed_df, pd.DataFrame(new_log_entries)], ignore_index=True
        )
        updated_log.to_csv(log_path, index=False)

    _print_summary(len(new_log_entries), crc_error_files, failed_files)


def _print_summary(
    n_loaded: int,
    crc_error_files: list[str],
    failed_files: list[tuple[str, str]],
) -> None:
    """Print a summary of the loading process."""
    print(f"\nNewly loaded files: {n_loaded}")

    if crc_error_files:
        print(f"\nFiles with CRC error (loaded anyway): {len(crc_error_files)}")
        for name in crc_error_files:
            print(f"  - {name}")

    if failed_files:
        print(f"\nFiles that could not be read (skipped): {len(failed_files)}")
        for name, error in failed_files:
            print(f"  - {name}: {error}")


convert_fit_dir_to_csv("./Source_Data/Fitnessdaten")