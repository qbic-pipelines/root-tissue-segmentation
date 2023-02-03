import os
from argparse import ArgumentParser
from pprint import pformat

import joblib as joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import sys
import yaml
from contextlib import redirect_stdout
import io
from optuna.integration import MLflowCallback
from optuna.samplers import TPESampler
from optuna.visualization import plot_edf, plot_contour, plot_intermediate_values, plot_optimization_history, \
    plot_parallel_coordinate, plot_param_importances


def show_optuna_results(filepath) -> None:
    """
    Prints results of the hyperparameter optimization into command line.
    :param filepath: The filename of the optuna file to be loaded.
    :return: None.
    """
    study = joblib.load(filepath)
    print('Best trial until now:', study.best_trial.number)
    print(' Value: ', study.best_trial.value)
    print(' Params: ')
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')


def plot_optuna_results(log_dir: str, study: optuna.study.Study) -> None:
    """
    Creates several plots obtained during a Optuna study, most importantly hyperparameter importance.
    :param log_dir: Directory where plots are saved.
    :param study: Optuna study object to create plots.
    """
    os.makedirs(log_dir, exist_ok=True)
    plots = []
    names = ['edf', 'contour', 'intermediate', 'history', 'parallel', 'importance']
    plots.append(plot_edf(study))
    plots.append(plot_contour(study))
    plots.append(plot_intermediate_values(study))
    plots.append(plot_optimization_history(study))
    plots.append(plot_parallel_coordinate(study))
    plots.append(plot_param_importances(study))
    for idx, name in enumerate(names):
        plots[idx].write_html(f"{log_dir}/{name}.html")


def suggest_hyperparameters(trial: optuna.Trial):
    """
    Suggests hyperparameters for an Optuna trial. For further details see the Optuna documentation.
    :param trial: Optuna Trial object.
    :return: Dictionary containing the suggested hyperparameters.
    """
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight-decay", 1e-4, 5e-1, log=True)
    epsilon = trial.suggest_float("epsilon", 1e-16, 1e-8, log=True)  # 3.170393250650158e-12 
    gamma = trial.suggest_uniform("gamma-factor", 1.4, 3)

    max_epochs = trial.suggest_int('max_epochs', low=50, high=100)
    hps = {'lr': lr, 'weight-decay': weight_decay, 'gamma-factor': gamma,
           'epsilon': epsilon, 'max_epochs': max_epochs}
    for i in range(5):
        hps[f'alpha-{i}'] = trial.suggest_float(f"alpha-{i}", 1e-2, 1)
    print(f"Suggested hyperparameters: \n{pformat(trial.params)}")
    return hps


def objective(
    trial: optuna.Trial,
) -> float:
    hparams = suggest_hyperparameters(trial)
    mlflow_path = "mlruns"
    #remove_previous_model(mlflow_path)
    
    val_iou = run_mlflow_project(hparams, "HPO Optimization")
    os.system("rm -r /tmp/tmp*")
    os.system("docker system prune -a -f")
    return val_iou


def run_mlflow_project(
    #hparams, 
    experiment_name):
    try:
        mlflow.projects.run('.', 
        #parameters=hparams, 
        experiment_name=experiment_name,docker_args={"gpus":"all"})
        with open('data/best.txt') as file:
            val_iou = float(file.readline())
    except RuntimeError:
        val_iou = 0
    return val_iou


def test_reproducibility(#hparams: dict, 
                        n_trials_reproducibility: int = 10,
                        csv_path: str = os.path.join("deterministic", "ious.csv"), mlflow_path: str = "mlruns"
                          ):
    """

    :param hparams: Hyperparameters to test the reproducibility of the experiment.
    :param n_trials_reproducibility: Number of trials to test the reproducibility of the experiment.
    :return: Best validation IoUs achieved during each training run.
    """
    val_ious = []
    os.makedirs(csv_path, exist_ok=True)
    for i in range(n_trials_reproducibility):
        val_ious.append(run_mlflow_project(
            #hparams, 
            "reproducibility"))
    val_ious = np.array(val_ious)
    pd.DataFrame(val_ious, columns=['Run']).to_csv(os.path.join(csv_path, "determinism_test_ious.csv"))
    if np.var(val_ious, axis=0) == 0.0:
        print("Model is reproducible")
    else:
        print("Model is not reproducible")


def remove_previous_model(mlflow_path):
    try:
        os.remove(mlflow_path + "/best.ckpt")
        os.remove(mlflow_path + "/lr_find_temp_model.ckpt")
    except OSError as e:
        print("Error: %s : %s" % (mlflow_path, e.strerror))


def optimize_hyperparameters(seed=0, n_startup_trials=10, n_trials=100, sampler_name: str = 'MultivariateTPE',
                             pruner_name: str = None, log_dir: str = os.path.join(os.getcwd(), 'optuna'),
                             joblib_filename: str = 'optuna.pkl', plot_dir="plots"):
    mlflow_cb = MLflowCallback(
        #tracking_uri='file:///mlflow/tmp/mlruns',
        metric_name='val_avg_iou'
    )

    sampler = get_sampler(n_startup_trials, sampler_name, seed)
    pruner = get_pruner(pruner_name)
    study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="HPO Optimization", direction='maximize')
    study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_cb])
    os.makedirs(log_dir, exist_ok=True)
    optuna_filepath = os.path.join(log_dir, joblib_filename)
    joblib.dump(value=study, filename=optuna_filepath)
    show_optuna_results(filepath=optuna_filepath)
    plot_dir = os.path.join(log_dir, plot_dir)
    plot_optuna_results(log_dir=plot_dir, study=study)
    return study.best_trial.params


def get_sampler(n_startup_trials, sampler_name, seed):
    if sampler_name == "MultivariateTPE":
        sampler = TPESampler(seed=seed, multivariate=True, n_startup_trials=n_startup_trials)
    else:
        print("Currently only MultivariateTPE is supported, switching to MultivariateTPE.")
        sampler = TPESampler(seed=seed, multivariate=True, n_startup_trials=n_startup_trials)
    return sampler


def get_pruner(pruner_name):
    if pruner_name == "Hyperband":
        print("Hyperband not working correctly.")
        pruner = optuna.pruners.HyperbandPruner()
    else:
        print("Currently only Hyperband is supported, deactivating pruning.")
        pruner = None
    return pruner


if __name__ == "__main__":
    """
    Conducts Hyperparameter optimization for the rsphd project.
    """
    parser = ArgumentParser(description='Optuna argument parser.')
    parser.add_argument(
        '--n-trials',
        type=int,
        default=300,
        help='Number of trials to optimize',
    )
    parser.add_argument(
        '--n-startup-trials',
        type=int,
        default=10,
        help='Number of trials for startup',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed of the Optuna Sampler',
    )
    parser.add_argument(
        '--sampler',
        type=str,
        default='MultivariateTPE',
        help='Name of the Optuna sampler',
        choices=['MultivariateTPE']
    )
    parser.add_argument(
        '--pruner',
        type=str,
        default='Hyperband',
        help='Name of the Optuna pruner',
        choices=[None, 'Hyperband']
    )
    args = parser.parse_args()
    #optuna_hparams = vars(args)
    #optuna_log_dir = os.path.join(os.getcwd(), 'optuna')
    #final_hps = optimize_hyperparameters(optuna_hparams['seed'], optuna_hparams['n_startup_trials'],
    #                                     optuna_hparams['n_trials'],
    #                                     optuna_hparams['sampler'], optuna_hparams['pruner'], log_dir=optuna_log_dir)
    #with open(f'{optuna_log_dir}/best_hparams.yml', 'w') as outfile:
    #    yaml.dump(final_hps, outfile)
    test_reproducibility()#(final_hps, csv_path=os.path.join(optuna_log_dir))
