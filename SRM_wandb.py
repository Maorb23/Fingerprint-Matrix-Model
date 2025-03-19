import wandb
from prefect import task, flow
from pathlib import Path
from srm_preprocess import SRM_preprocess
from srm_train import SRM_train
import pandas as pd
import numpy as np
import json
import argparse
import holoviews as hv
from catboost import CatBoostClassifier
import panel as pn
from bokeh.resources import INLINE
hv.extension("bokeh", logo=False)
from error_analysis import error_analysis

"""
+-+-+-+-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+
|P|r|e|p|r|o|c|e|s|s|i|n|g| |T|a|s|k|
+-+-+-+-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+
"""

@task(name="preprocess_data")
def preprocess_data(csv_path: str, output_path: str, filtered):
    # Create output directory and convert to Path
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    preprocessor = SRM_preprocess(
        base_dir=Path(csv_path),
        columns=[
            'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
            'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r',
            'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
        ],
        filtered=filtered
    )
    data_boring, data_interesting, data_knock, data_standup = preprocessor.process_and_save(output_path, save_as_pickle=False)
    return data_boring, data_interesting, data_knock, data_standup


"""
+-+-+-+-+-+-+-+-+ +-+-+-+-+
|T|r|a|i|n|i|n|g| |T|a|s|k|
+-+-+-+-+-+-+-+-+ +-+-+-+-+
"""

@task(name="train_model")
def train_model(data, tol: float, max_iter: int, verbose: bool):
    """
    Load best hyperparams from JSON (assuming it was saved by the tuner), then train and evaluate the model.
    """
    trainer = SRM_train(tol=tol, max_iter=max_iter, verbose=verbose)
    iter_count, W_i_new_group, S, dist_vec, delta_S_list, delta_W_list = trainer.SRM(data)
    return iter_count, W_i_new_group, S, dist_vec, delta_S_list, delta_W_list

    
"""
+-+-+-+-+ +-+-+-+-+
|M|a|i|n| |F|l|o|w|
+-+-+-+-+ +-+-+-+-+
"""

@flow(name="preprocess_and_train_flow")
def preprocess_and_train_flow(
    csv_path: str,
    output_path: str,
    tol: float,
    max_iter: int,
    verbose: bool,
    model_name: str,
    run_id: int,
    filtered: bool
):
    wandb.init(
        project="SRM_ArgLiron",
        settings=wandb.Settings(start_method="thread"),
        config={"model_name": model_name, "run_id": run_id}
    )
    data_boring, data_interesting, data_knock, data_standup = preprocess_data(csv_path, output_path, filtered)
    # join all the data horizontally on same row, meaning each row will have the columns of boring, interesting, knock, standup
    descriptive_stats = []
    for i in range(len(data_boring)):
        row_stats = pd.DataFrame({
            'Boring': data_boring[i].flatten(),
            'Interesting': data_interesting[i].flatten(),
            'Knock': data_knock[i].flatten(),
            'Standup': data_standup[i].flatten()
        }).describe().T

        # Flatten the DataFrame so each participant's stats become one row.
        flattened = {}
        for category in row_stats.index:
            for metric in row_stats.columns:
                flattened[f"{category}_{metric}"] = row_stats.loc[category, metric]
        descriptive_stats.append(flattened)

    descriptive_table = pd.DataFrame(descriptive_stats)

    random_integer = np.random.randint(0, 28)
    boring_hist = np.histogram(np.array(data_boring[random_integer]).flatten())
    interesting_hist = np.histogram(np.array(data_interesting[random_integer]).flatten())
    knock_hist = np.histogram(np.array(data_knock[random_integer]).flatten())
    standup_hist = np.histogram(np.array(data_standup[random_integer]).flatten())

    # Log to wandb
    
    wandb.log({
        "descriptive_table": wandb.Table(dataframe=descriptive_table),
        "data_boring": wandb.Histogram(np_histogram=boring_hist),
        "data_interesting": wandb.Histogram(np_histogram=interesting_hist),
        "data_knock": wandb.Histogram(np_histogram=knock_hist),
        "data_standup": wandb.Histogram(np_histogram=standup_hist)
    })
    
    




    iter_count, \
    W_list_boring, \
    S_boring, \
    dist_vec_boring, \
    delta_S_list_boring,delta_W_boring = train_model(data_boring, tol, max_iter, verbose)

    table = wandb.Table(dataframe=pd.DataFrame({"iter_count": np.arange(len(delta_S_list_boring)), "dist_vec": dist_vec_boring, "delta_S_list": delta_S_list_boring, "delta_W_list": delta_W_boring}))
    wandb.log({"boring_lineplot": wandb.plot.line_series(
        xs = np.arange(len(table.get_column("delta_W_list"))),
        ys = [table.get_column("delta_W_list"), table.get_column("delta_S_list"), table.get_column("dist_vec")],
        title="metrics vs iter_count"),
               "boring_table": table})
               
    """
    iter_count, \
    W_list_interesting, \
    S_interesting, \
    dist_vec_interesting, \
    delta_S_list_interesting, \
    delta_W_interesting = train_model(data_interesting, tol, max_iter, verbose)

    iter_count, \
    W_list_knock, \
    S_knock, \
    dist_vec_knock, \
    delta_S_list_knock, \
    delta_W_knock = train_model(data_knock)

    iter_count, \
    W_list_standup, \
    S_standup, \
    dist_vec_standup, \
    delta_S_list_standup, \
    delta_W_standup = train_model(data_standup)"
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess and train SRM model')
    parser.add_argument('--csv_path', type=str, default = r"C:\Users\maorb\Classes\Arg_Liron\CSV_OpenFace - Main", help='Path to input data')
    parser.add_argument('--output_path', type=str, default='data/processed', help='Path to save the output data')
    parser.add_argument('--tol', type=float, default=1e-3, help='Tolerance for convergence')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--model_name', type=str, default='SRM', help='Name of the model')
    parser.add_argument('--run_id', type=int, default=1, help='ID of the run')
    parser.add_argument('--filtered', action='store_true', default=False, help='Apply filtering to data')
    args = parser.parse_args()

    flow = preprocess_and_train_flow(
        csv_path=args.csv_path,
        output_path=args.output_path,
        tol=args.tol,
        max_iter=args.max_iter,
        verbose=args.verbose,
        model_name=args.model_name,
        run_id=args.run_id,
        filtered=args.filtered
    )



    