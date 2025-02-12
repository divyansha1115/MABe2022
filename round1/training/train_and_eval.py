import json
import os
import numpy as np
import time
import pandas as pd
from tqdm.auto import tqdm

from round1.training.dataloader import MABeDataSplitter
from round1.training.trainer import MABeSingleTaskTrainer
import ray
from tqdm import tqdm

@ray.remote(num_cpus=8)
def train_single_task(data_splitter, seed, task_id, Constants, test_size, alpha):
    
    trainer = MABeSingleTaskTrainer(data_splitter=data_splitter)
    trainer.split_data(seed=seed, split_keys=['SubmissionTrain'], test_size=test_size)
    trainer.data_splitter.load_labels(task_id=task_id)
    trainer.setup_logging(log_path=Constants.LOG_PATH, train_prefix=f'alpha-{alpha}-')
    trainer.setup_neural_net(alpha=alpha)
    trainer.train()
    return {f'{task_id}-alpha-{alpha}' :trainer.model_path}
    
def train_multiple_tasks(data_splitter, seed, task_id_list, Constants, test_size, alpha_vals):
    start = time.time()
    
    # Create a reference to the data splitter that can be shared across tasks
    data_splitter_ref = ray.put(data_splitter)
    constants_ref = ray.put(Constants)
    
    results = []
    for task_id in task_id_list:
        for alpha in alpha_vals:
            results.append(train_single_task.remote(data_splitter_ref, seed, task_id, constants_ref, test_size, alpha))
            
    # Create a progress bar for tracking
    pbar = tqdm(total=len(results), desc="Training tasks")
    
    # Collect results as they complete
    model_paths = {}
    while results:
        done_id, results = ray.wait(results)
        result = ray.get(done_id[0])
        model_paths.update(result)
        pbar.update(1)
    
    pbar.close()
    print(f'Seed {seed}: All tasks train time', time.time() - start)
    
    return model_paths

def predict_single_task_multiseed(data_splitter, task_id, model_paths, Constants, split_keys):
    start = time.time()

    trainer = MABeSingleTaskTrainer(data_splitter=data_splitter)
    trainer.split_data(seed=0, # Doesn't matter when test size is 0
                       split_keys=split_keys, 
                       test_size=0.0)
    
    trainer.data_splitter.load_labels(task_id=task_id)

    all_y_preds = []
    y_true = data_splitter.y_train
    X = data_splitter.X_train
    for mp in model_paths:
        trainer.load_model(mp)
        y_pred = trainer.model.predict(X)
        all_y_preds.append(y_pred)
    
    agg_fn, metrics_fn, metric_name = trainer.get_agg_and_metric()
    
    print(f'Task {task_id}: {trainer.data_splitter.split_keys} Predict time for all seeds', time.time() - start)

    return all_y_preds, agg_fn, metrics_fn, y_true, metric_name

@ray.remote(num_cpus=4)
def eval_single_task_multiseed(data_splitter, task_id, alpha, model_paths, Constants, task_is_sequence_level):

    def rem_nan_idx(y):
        y_notnan_idx = ~np.isnan(y)
        y_new = y[y_notnan_idx]
        return y_new

    res = predict_single_task_multiseed(data_splitter, task_id, model_paths, Constants, split_keys=['publicTest'])
    public_y_preds, agg_fn, metrics_fn, public_y_true, metric_name = res

    public_agg = agg_fn(public_y_preds)
    public_score = metrics_fn(public_y_true, public_agg)

    # Average Pooling Sequences
    # load each sequence with frame number map
    # regenerate not nan indexes by adding frame indexes back to back
    remnan_start = 0
    y_pooled_true, y_pooled_pred = [], []
    for sk in data_splitter.train_snippets:
        start, end = data_splitter.frame_number_map[sk]
        y_orig = data_splitter.labels['label_array'][data_splitter.task_idx, start:end]
        y_remnan = rem_nan_idx(y_orig)
        if len(y_remnan) > 0:
            avg_pool_gt = np.mean(y_remnan)
            if metric_name == 'f1_score':
                avg_pool_gt = int(avg_pool_gt >= 0.5)
            y_pooled_true.append(avg_pool_gt)
            avg_pool_pred = np.mean(public_agg[remnan_start:remnan_start+len(y_remnan)])
            if metric_name == 'f1_score':
                avg_pool_pred = int(avg_pool_pred >= 0.5)
            y_pooled_pred.append(avg_pool_pred)
            remnan_start += len(y_remnan)

    res = predict_single_task_multiseed(data_splitter, task_id, model_paths, Constants, split_keys=['privateTest'])
    private_y_preds, agg_fn, metrics_fn, private_y_true, metric_name = res

    pri_pub_y_preds = [np.concatenate([pr, pb]) for pr, pb in zip(private_y_preds, public_y_preds)]
    pri_pub_y_agg = agg_fn(pri_pub_y_preds)
    pri_pub_y_true = np.concatenate([private_y_true, public_y_true])
    private_score = metrics_fn(pri_pub_y_true, pri_pub_y_agg)

    remnan_start = 0
    private_agg = agg_fn(private_y_preds)
    for sk in data_splitter.train_snippets:
        start, end = data_splitter.frame_number_map[sk]
        y_orig = data_splitter.labels['label_array'][data_splitter.task_idx, start:end]
        y_remnan = rem_nan_idx(y_orig)
        if len(y_remnan) > 0:
            avg_pool_gt = np.mean(y_remnan)
            if metric_name == 'f1_score':
                avg_pool_gt = int(avg_pool_gt >= 0.5)
            y_pooled_true.append(avg_pool_gt)
            avg_pool_pred = np.mean(private_agg[remnan_start:remnan_start+len(y_remnan)])
            if metric_name == 'f1_score':
                avg_pool_pred = int(avg_pool_pred >= 0.5)
            y_pooled_pred.append(avg_pool_pred)
            remnan_start += len(y_remnan)
    
    single_seed_scores = [metrics_fn(pri_pub_y_true, pred_single_seed) for pred_single_seed in pri_pub_y_preds]
    no_ensemble_score = np.mean(single_seed_scores)

    if task_is_sequence_level:
        pooled_score = metrics_fn(y_pooled_true, y_pooled_pred)
    else:
        pooled_score = -1

    return {f'alpha_{alpha}_taskid_{task_id}': [private_score, public_score,  metric_name, no_ensemble_score, pooled_score, task_is_sequence_level]}

def run_all_tasks(Constants, test_size):
    with open(Constants.SPLIT_INFO_FILE, 'r') as fp:
        split_info = json.load(fp)

    with open(Constants.TASK_INFO_FILE, 'r') as fp:
        tasks_info = json.load(fp)

    task_id_list = tasks_info['task_id_list']
    sequence_level_tasks = tasks_info['sequence_level_tasks']
    seeds = tasks_info['seeds']
    alpha_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
    if Constants.SHORT_RUN:
        seeds = [42, 43]
        task_id_list = task_id_list[1:3]
        alpha_vals = [0.1, 1.0]
   
    start = time.time()
    data_splitter = MABeDataSplitter(submission_data_path=Constants.SUBMISSION_DATA_PATH,              
                                     split_info=split_info,
                                     labels_path=Constants.LABELS_PATH,
                                     frame_number_map_file=Constants.FRAME_NUMBER_MAP)
    print(f'Submission data load time', time.time() - start)

    model_paths_all = {}
    for seed in seeds:
        model_paths_all[seed] = train_multiple_tasks(data_splitter, seed, task_id_list, Constants, test_size, alpha_vals)

    ray.put(model_paths_all)
    ray.put(data_splitter)
    
    results = []
    model_paths = {}
    future_results = []
    for task_id in task_id_list:
        alpha_scores = []
        for alpha in alpha_vals:
            model_paths[alpha] = []
            for seed in seeds:
                model_paths[alpha].append(model_paths_all[seed][task_id + '-alpha-' + str(alpha)])
        
            task_is_sequence_level = task_id in sequence_level_tasks
            future_results.append(eval_single_task_multiseed.remote(data_splitter, task_id, alpha, model_paths[alpha], Constants, task_is_sequence_level))
            
            
    pbar = tqdm(total=len(future_results), desc="Evaluating tasks")
    all_results = {}
    while future_results:
        done_id, future_results = ray.wait(future_results)
        res = ray.get(done_id[0])
        all_results.update(res)
        pbar.update(1)
    
    pbar.close()   
    results = []
    for task_id in task_id_list:
        alpha_scores = []
        for alpha in alpha_vals:
            alpha_scores.append(all_results[f'alpha_{alpha}_taskid_{task_id}'][0].copy())
            if alpha == 1.0:
                private_score, public_score, metric_name, no_ensemble_score, pooled_score, task_is_sequence_level = all_results[f'alpha_{alpha}_taskid_{task_id}']
                task_results = [task_id, private_score, public_score, metric_name, no_ensemble_score, pooled_score, task_is_sequence_level]
        print("Results: Task", task_id, "| Metric", metric_name, "| Public", public_score, "| Private", private_score, '\n')
        task_results.extend(alpha_scores)
        results.append(task_results)
   
    columns=['Task ID', 'Private Score', 'Public Score', 'Metric', 
             'No Ensemble Score', 'Pooled Score', 'Sequence Level Task']
    for alpha in alpha_vals:
        columns.append(f"Score Alpha {alpha}")
    
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(os.path.join(Constants.LOG_PATH, 'results.csv'), index=False)
    
    
if __name__ == '__main__':
    class Constants:
        SUBMISSION_DATA_PATH = os.getenv('SUBMISSION_DATA_PATH', './example_data/example_embeddings.npy')
        LABELS_PATH = os.getenv('LABELS_PATH', './example_data/example_labels.npy')
        SPLIT_INFO_FILE = os.getenv('SPLIT_INFO_FILE', './example_data/example_split.json')
        TASK_INFO_FILE = os.getenv('TASK_INFO_FILE', '../metadata/jax_tasks.json')
        LOG_PATH = os.getenv('TRAINING_LOG_PATH', './temp')

        # SUBMISSION_DATA_PATH = '/home/dipam/aicrowd/mabe2022/data/round1_upload/mouse_triplets/sample_submission.npy' 
        # LABELS_PATH = '/home/dipam/aicrowd/mabe2022/data/round1_upload/mouse_triplets/submission_labels.npy'
        # SPLIT_INFO_FILE = "../metadata/jax_split.json"
        # TASK_INFO_FILE = "../metadata/jax_tasks.json"

        # SUBMISSION_DATA_PATH = '/home/dipam/aicrowd/mabe2022/data/round1_upload/fruit_flies/sample_submission.npy' 
        # LABELS_PATH = '/home/dipam/aicrowd/mabe2022/data/round1_upload/fruit_flies/submission_labels.npy'
        # SPLIT_INFO_FILE = "../metadata/flies_split.json"
        # TASK_INFO_FILE = "../metadata/fly_tasks.json"

    run_all_tasks(Constants)
    
