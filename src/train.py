import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from model import TaskEncoder, FiLMGenerator, FiLMRegressor
from utils import create_meta_tasks, prepare_task_relevance, run_task_embedding_visualization, run_k_shot_analysis
import config

try:
    df_real_main = pd.read_csv(config.PATH_REAL_MAIN)
    df_sim = pd.read_csv(config.PATH_SIMULATED)
    df_real_test = pd.read_csv(config.PATH_REAL_TEST)
    print("Data loaded successfully")
except FileNotFoundError as e:
    print(f"Error - {e}")

features = ['pH', 'temperature', 'sensor_value']
target = 'concentration_delta'

df_real_main_cleaned = df_real_main.dropna(subset=features + ['real_concentration']).copy()
df_real_main_cleaned['enzyme_type'] = 'real_enzyme_main_train'
df_real_main_cleaned[target] = df_real_main_cleaned['real_concentration'] - df_real_main_cleaned['sensor_value']

df_sim_cleaned = df_sim.dropna(subset=features + ['real_concentration']).copy()

if 'enzyme_id' in df_sim_cleaned.columns:
    df_sim_cleaned['enzyme_type'] = 'sim_' + df_sim_cleaned['enzyme_id'].astype(str)
else:
    df_sim_cleaned['enzyme_type'] = 'sim_generic'
df_sim_cleaned[target] = df_sim_cleaned['real_concentration'] - df_sim_cleaned['sensor_value']

df_real_test_cleaned = df_real_test.dropna(subset=features + ['real_concentration']).copy()
if 'enzyme_id' in df_real_test_cleaned.columns:
    df_real_test_cleaned['enzyme_type'] = 'real_test_' + df_real_test_cleaned['enzyme_id'].astype(str)
else:
    df_real_test_cleaned['enzyme_type'] = 'real_test_generic'
df_real_test_cleaned[target] = df_real_test_cleaned['real_concentration'] - df_real_test_cleaned['sensor_value']

df_meta_train = pd.concat([df_sim_cleaned, df_real_main_cleaned], ignore_index=True)
meta_train_task_names = df_meta_train['enzyme_type'].unique()
print(f"Total number of training tasks: {len(meta_train_task_names)}")


df_sim_cleaned = df_meta_train[df_meta_train['enzyme_type'].str.startswith('sim_')]
sim_task_names = df_sim_cleaned['enzyme_type'].unique()

df_real_train_cleaned = df_meta_train[df_meta_train['enzyme_type'].str.startswith('real_')]

features = ['pH', 'temperature', 'sensor_value']
target = 'concentration_delta'
features_and_target = features + [target]
task_relevance = prepare_task_relevance(df_sim_cleaned, df_real_train_cleaned, sim_task_names, features_and_target)

sorted_sim_tasks = sorted(task_relevance.items(), key=lambda item: item[1]['score'], reverse=True)
sorted_sim_task_names = [name for name, data in sorted_sim_tasks]

features = ['pH', 'temperature', 'sensor_value']
target = 'concentration_delta'
task_encoder = TaskEncoder(input_feature_dim=len(features), output_dim=config.TASK_EMBEDDING_DIM)
film_regressor = FiLMRegressor(input_size=len(features), hidden_size=config.HIDDEN_SIZE)
film_generator = FiLMGenerator(task_embedding_dim=config.TASK_EMBEDDING_DIM, num_film_layers=2, hidden_size=config.HIDDEN_SIZE)

all_params = list(film_regressor.parameters()) + list(film_generator.parameters()) + list(task_encoder.parameters())
optimizer = optim.Adam(all_params, lr=config.LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
loss_fn = nn.MSELoss()

task_encoder.train()
film_regressor.train()
film_generator.train()

for iteration in range(config.N_ITERATIONS):
    optimizer.zero_grad()

    if iteration < config.STAGE1_ITERATIONS:
        stage = 1
        pool_size = int(len(sorted_sim_task_names) * config.STAGE1_POOL_RATIO)
        if pool_size == 0: pool_size = 1
        sampling_pool = sorted_sim_task_names[:pool_size]
    else:
        stage = 2
        sampling_pool = meta_train_task_names

    task_name = np.random.choice(sampling_pool, 1)

    task_batch = create_meta_tasks(df_meta_train, task_name, config.K_SHOT, config.Q_QUERY, features, target)
    if not task_batch: continue

    (support_x, support_y), (query_x, query_y) = list(task_batch.values())[0]

    mean = support_x.mean(0, keepdim=True)
    std = support_x.std(0, keepdim=True) + 1e-8
    support_x_norm = (support_x - mean) / std
    query_x_norm = (query_x - mean) / std

    task_embedding = task_encoder(support_x_norm, support_y)

    gammas, betas = film_generator(task_embedding)

    query_pred = film_regressor(query_x_norm, gammas, betas)

    loss = loss_fn(query_pred, query_y)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if iteration % 1000 == 0:
        print(
            f"Iter {iteration}/{config.N_ITERATIONS} [Stage {stage}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.10f}")

print("Training completed!")

print("\n" + "="*60)
task_encoder.eval()
film_regressor.eval()
film_generator.eval()

test_task_names = df_real_test_cleaned['enzyme_type'].unique()
all_results = {}
final_predictions_df = pd.DataFrame()

for task_name in test_task_names:
    task_df = df_real_test_cleaned[df_real_test_cleaned['enzyme_type'] == task_name]
    print(f"\nEvaluating new tasks: {task_name} (Number of samples: {len(task_df)})")

    n_splits = min(5, len(task_df))
    if n_splits < 2: continue
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    task_real_concentrations = []
    task_calibrated_concentrations = []

    with torch.no_grad():
        for support_idx, query_idx in kf.split(task_df):
            support_df = task_df.iloc[support_idx]
            query_df = task_df.iloc[query_idx]

            support_x_raw = torch.tensor(support_df[features].values, dtype=torch.float32)
            support_y = torch.tensor(support_df[target].values, dtype=torch.float32).view(-1, 1)

            mean, std = support_x_raw.mean(0, keepdim=True), support_x_raw.std(0, keepdim=True) + 1e-8
            support_x_norm = (support_x_raw - mean) / std

            query_x_raw = torch.tensor(query_df[features].values, dtype=torch.float32)
            query_x_norm = (query_x_raw - mean) / std

            task_embedding = task_encoder(support_x_norm, support_y)
            gammas, betas = film_generator(task_embedding)
            test_pred_deltas = film_regressor(query_x_norm, gammas, betas)

            calibrated_values = query_df['sensor_value'].values + test_pred_deltas.numpy().flatten()

            fold_results_df = query_df.copy()
            fold_results_df['calibrated_concentration'] = calibrated_values
            fold_results_df['predicted_delta'] = test_pred_deltas.numpy().flatten()
            fold_results_df['kfold_split'] = f"Task_{task_name}_Fold"
            final_predictions_df = pd.concat([final_predictions_df, fold_results_df], ignore_index=True)

            task_real_concentrations.extend(query_df['real_concentration'].values)
            task_calibrated_concentrations.extend(
                query_df['sensor_value'].values + test_pred_deltas.numpy().flatten())

    if len(task_real_concentrations) > 0:
        task_rmse = np.sqrt(mean_squared_error(task_real_concentrations, task_calibrated_concentrations))
        task_r2 = r2_score(task_real_concentrations, task_calibrated_concentrations)
        task_mae = mean_absolute_error(task_real_concentrations, task_calibrated_concentrations)
        all_results[task_name] = {'RMSE': task_rmse, 'R2': task_r2, 'MAE': task_mae}

print("=" * 60)
if not all_results:
    print("Evaluation failed.")
else:
    avg_rmse = np.mean([metrics['RMSE'] for metrics in all_results.values()])
    avg_r2 = np.mean([metrics['R2'] for metrics in all_results.values()])
    avg_mae = np.mean([metrics['MAE'] for metrics in all_results.values()])

    for task_name, metrics in all_results.items():
        print(f"Task: {task_name}")
        print(f"RMSE:{metrics['RMSE']:.4f}")
        print(f"MAE:{metrics['MAE']:.4f}")
        print(f"R2:(R-squared): {metrics['R2']:.4f}")

    if len(all_results) > 1:
        print("\n--- The average performance of all independent test tasks ---")
        print(f"RMSE:{avg_rmse:.4f}")
        print(f"MAE:{avg_mae:.4f}")
        print(f"R2(R-squared):{avg_r2:.4f}")
print("=" * 60)

trained_models = (task_encoder, film_generator, film_regressor)

run_task_embedding_visualization(df_meta_train, trained_models, features, target, k_shot=10, task_relevance=task_relevance)
run_k_shot_analysis(df_real_test_cleaned, trained_models, features, target)
