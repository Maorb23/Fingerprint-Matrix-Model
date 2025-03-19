import pandas as pd
import os
import argparse
from pathlib import Path
import logging
import json
import numpy as np
from scipy.ndimage import uniform_filter

class SRM_preprocess:
    def __init__(self, base_dir, columns, filtered=False):
        self.base_dir = base_dir
        self.columns = columns
        self.filtered = filtered
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


    def read_and_process_file(self,file_path, columns):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return None
        
        df.columns = df.columns.str.strip()  # Strip whitespace from column names
        return df[columns] if all(col in df.columns for col in columns) else None  # Ensure all columns exist

    def process_and_save(self, output_path, save_as_pickle = False):
        # Recursively get all files that match conditions
        files_boring = []
        files_interesting = []
        files_knock = []
        files_standup = []
        self.logger.info(f"Processing files in {self.base_dir}")
        for root, dirs, filenames in os.walk(self.base_dir):
            for filename in filenames:
                if filename.startswith("Argaman") and "boring" in filename:
                    files_boring.append(os.path.join(root, filename))
                elif filename.startswith("Argaman") and "interesting" in filename:
                    files_interesting.append(os.path.join(root, filename))
                elif filename.startswith("Argaman") and "knock" in filename:
                    files_knock.append(os.path.join(root, filename))
                elif filename.startswith("Argaman") and "standup" in filename:
                    files_standup.append(os.path.join(root, filename))
        self.logger.warning(f"Found {len(files_boring)} boring files")
        # Read and process all files
        dataframes_boring = [self.read_and_process_file(file, self.columns) for file in files_boring]
        dataframes_interesting = [self.read_and_process_file(file, self.columns) for file in files_interesting]
        dataframes_knock = [self.read_and_process_file(file, self.columns) for file in files_knock]
        dataframes_standup = [self.read_and_process_file(file, self.columns) for file in files_standup]
        
        # Drop any None values in case some files were not processed
        dataframes_boring = [df for df in dataframes_boring if df is not None]
        dataframes_interesting = [df for df in dataframes_interesting if df is not None]
        dataframes_knock = [df for df in dataframes_knock if df is not None]
        dataframes_standup = [df for df in dataframes_standup if df is not None]

        # Find the minimum length of all dataframes (across all categories)
        min_length = min(
            min(len(df) for df in dataframes_boring),
            min(len(df) for df in dataframes_interesting),
            min(len(df) for df in dataframes_knock),
            min(len(df) for df in dataframes_standup)
        )

        # Trim all dataframes to the minimum length
        dataframes_boring = [df.iloc[:min_length, :] for df in dataframes_boring]
        dataframes_interesting = [df.iloc[:min_length, :] for df in dataframes_interesting]
        dataframes_knock = [df.iloc[:min_length, :] for df in dataframes_knock]
        dataframes_standup = [df.iloc[:min_length, :] for df in dataframes_standup]

        dataframes1_trans_boring = [df.T.to_numpy() for df in dataframes_boring]
        dataframes1_trans_interesting = [df.T.to_numpy() for df in dataframes_interesting]
        dataframes1_trans_standup = [df.T.to_numpy() for df in dataframes_standup]
        dataframes1_trans_knock = [df.T.to_numpy() for df in dataframes_knock]

        if self.filtered:
            self.logger.warning("Applying filtering to the data")
            dataframes1_trans_boring = [uniform_filter(df, size=5) for df in dataframes1_trans_boring]
            dataframes1_trans_interesting = [uniform_filter(df, size=5) for df in dataframes1_trans_interesting]
            dataframes1_trans_standup = [uniform_filter(df, size=5) for df in dataframes1_trans_standup]
            dataframes1_trans_knock = [uniform_filter(df, size=5) for df in dataframes1_trans_knock]

        

        # Concatenate all dataframes for each category
        df_boring = pd.concat(dataframes_boring, ignore_index=True)
        df_interesting = pd.concat(dataframes_interesting, ignore_index=True)
        df_knock = pd.concat(dataframes_knock, ignore_index=True)
        df_standup = pd.concat(dataframes_standup, ignore_index=True)

        # Save the dataframes and the list of files
        self.logger.info(f"Saving data to {output_path}")


        # Save the transposed numpy arrays
        np.savez(output_path / "boring_arrays.npz", *dataframes1_trans_boring)
        np.savez(output_path / "interesting_arrays.npz", *dataframes1_trans_interesting)
        np.savez(output_path / "standup_arrays.npz", *dataframes1_trans_standup)
        np.savez(output_path / "knock_arrays.npz", *dataframes1_trans_knock)


        if save_as_pickle:
            df_boring.to_pickle(output_path / "boring.pkl")
            df_interesting.to_pickle(output_path / "interesting.pkl")
            df_knock.to_pickle(output_path / "knock.pkl")
            df_standup.to_pickle(output_path / "standup.pkl")

        else:
            df_boring.to_csv(output_path / "boring.csv", index=False)
            df_interesting.to_csv(output_path / "interesting.csv", index=False)
            df_knock.to_csv(output_path / "knock.csv", index=False)
            df_standup.to_csv(output_path / "standup.csv", index=False)
        
        self.logger.warning("Data saved successfully")
        return dataframes1_trans_boring, dataframes1_trans_interesting,dataframes1_trans_knock, dataframes1_trans_standup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=r"C:\Users\maorb\Classes\Arg_Liron\CSV_OpenFace - Main", help="Path to input data")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Relative path to save output data")
    parser.add_argument("--save_as_pickle", action="store_true", help="Save as pickle if set, otherwise CSV")
    parser.add_argument("--filtered", action="store_true", default=False, help="Apply filtering to the data")
    

    args = parser.parse_args()

    columns = [
        'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
        'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r',
        'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
    ]

    processor = SRM_preprocess(args.base_dir, columns, filtered=args.filtered)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    processor.process_and_save(output_path, save_as_pickle=args.save_as_pickle)
