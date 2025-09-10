import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def create_meta_tasks(df, task_names, k_shot, q_query, features, target):
    tasks = {}
    for task_name in task_names:
        task_df = df[df['enzyme_type'] == task_name]
        if len(task_df) < k_shot + q_query: continue
        support_df = task_df.sample(n=k_shot)
        remaining_df = task_df.drop(support_df.index)
        query_df = remaining_df.sample(n=q_query)
        support_x = torch.tensor(support_df[features].values, dtype=torch.float32)
        support_y = torch.tensor(support_df[target].values, dtype=torch.float32).view(-1, 1)
        query_x = torch.tensor(query_df[features].values, dtype=torch.float32)
        query_y = torch.tensor(query_df[target].values, dtype=torch.float32).view(-1, 1)
        tasks[task_name] = ((support_x, support_y), (query_x, query_y))
    return tasks


def rbf_kernel(x, y, gamma=1.0):
    x_size, y_size, dim = x.size(0), y.size(0), x.size(1)
    x = x.unsqueeze(1).expand(x_size, y_size, dim)
    y = y.unsqueeze(0).expand(x_size, y_size, dim)
    return torch.exp(-gamma * ((x - y) ** 2).sum(dim=2))

def mmd_loss_func(source_samples, target_samples, gamma=1.0):
    xx = rbf_kernel(source_samples, source_samples, gamma)
    yy = rbf_kernel(target_samples, target_samples, gamma)
    xy = rbf_kernel(source_samples, target_samples, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

def prepare_task_relevance(df_sim, df_real, sim_task_names, features_and_target):
    print("Task relevance score(based MMD)...")
    task_relevance = {}
    real_samples = torch.tensor(df_real[features_and_target].values, dtype=torch.float32)

    for sim_name in sim_task_names:
        sim_df = df_sim[df_sim['enzyme_type'] == sim_name]
        sim_samples = torch.tensor(sim_df[features_and_target].values, dtype=torch.float32)

        real_subset = real_samples[np.random.choice(len(real_samples), min(200, len(real_samples)), replace=False)]
        sim_subset = sim_samples[np.random.choice(len(sim_samples), min(200, len(sim_samples)), replace=False)]

        distance = mmd_loss_func(sim_subset, real_subset)
        relevance_score = torch.exp(-distance).item()
        task_relevance[sim_name] = {'score': relevance_score, 'distance': distance.item()}

    print("done")
    return task_relevance


def evaluate_on_task(task_df, task_encoder, film_generator, film_regressor, features, target, k_shot_for_adaptation):

    if len(task_df) <= k_shot_for_adaptation:
        return float('nan'), float('nan'), float('nan')

    support_df = task_df.sample(n=k_shot_for_adaptation)
    query_df = task_df

    with torch.no_grad():
        support_x_raw = torch.tensor(support_df[features].values, dtype=torch.float32)
        support_y = torch.tensor(support_df[target].values, dtype=torch.float32).view(-1, 1)

        mean = support_x_raw.mean(0, keepdim=True)
        if k_shot_for_adaptation > 1:
            std = support_x_raw.std(0, keepdim=True) + 1e-8
        else:
            std = torch.ones_like(mean)

        support_x_norm = (support_x_raw - mean) / std

        query_x_raw = torch.tensor(query_df[features].values, dtype=torch.float32)
        query_x_norm = (query_x_raw - mean) / std


        task_embedding = task_encoder(support_x_norm, support_y)
        gammas, betas = film_generator(task_embedding)
        test_pred_deltas = film_regressor(query_x_norm, gammas, betas)

        if torch.isnan(test_pred_deltas).any() or torch.isinf(test_pred_deltas).any():
            print(f"error: NaN/Inf predicted when k={k_shot_for_adaptation}, skipping this evaluation point.")
            return float('nan'), float('nan'), float('nan')

        test_pred_deltas_clipped = torch.clamp(test_pred_deltas, -1000, 1000)

        real_concentrations = query_df['real_concentration'].values
        calibrated_concentrations = query_df['sensor_value'].values + test_pred_deltas_clipped.numpy().flatten()

        rmse = np.sqrt(mean_squared_error(real_concentrations, calibrated_concentrations))
        r2 = r2_score(real_concentrations, calibrated_concentrations)
        mae = mean_absolute_error(real_concentrations, calibrated_concentrations)

    return rmse, r2, mae


def run_k_shot_analysis(df_test, models, features, target):
    print("\n" + "=" * 60)
    print("K-Shot analyze")
    print("=" * 60)

    task_encoder, film_generator, film_regressor = models
    task_encoder.eval(), film_generator.eval(), film_regressor.eval()

    k_shot_values = [1, 5, 10, 15, 20, 30]
    results_rmse, results_r2, results_mae = [], [], []

    task_df = df_test[df_test['enzyme_type'] == 'real_test_generic']
    max_k = len(task_df) - 2

    for k in k_shot_values:
        if k >= max_k: continue
        print(f"test K-Shot: {k}")

        rmses, r2s, maes = [], [], []
        for _ in range(10):
            rmse, r2, mae = evaluate_on_task(task_df, task_encoder, film_generator, film_regressor, features, target,
                                             k_shot_for_adaptation=k)
            if not np.isnan(rmse):
                rmses.append(rmse)
                r2s.append(r2)
                maes.append(mae)

        if rmses:
            results_rmse.append(np.mean(rmses))
            results_r2.append(np.mean(r2s))
            results_mae.append(np.mean(maes))
        else:
            results_rmse.append(float('nan'))

    valid_k_shots = [k for k in k_shot_values if k < max_k]
    print('valid_k_shots', valid_k_shots)
    print('results_rmse', results_rmse)
    print('results_r2', results_r2)
    print('results_mae', results_mae)

    plt.figure(figsize=(10, 6))
    plt.plot(valid_k_shots, results_rmse, marker='o', linestyle='-')

    plt.xlabel("K-Shot", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.title("K-Shot Analysis", fontsize=16)
    plt.xticks(valid_k_shots)
    plt.grid(True)
    plt.show()


def run_task_embedding_visualization(df_meta_train, models, features, target, k_shot, task_relevance):

    print("\n" + "=" * 60)
    print("Visualization of task embedding space")
    print("=" * 60)

    task_encoder, _, _ = models
    task_encoder.eval()

    task_embeddings = []
    task_labels = []
    task_types = []
    relevance_scores = []

    all_task_names = df_meta_train['enzyme_type'].unique()

    with torch.no_grad():
        for task_name in all_task_names:
            task_df = df_meta_train[df_meta_train['enzyme_type'] == task_name]
            if len(task_df) < k_shot: continue

            support_df = task_df.sample(n=k_shot)
            support_x_raw = torch.tensor(support_df[features].values, dtype=torch.float32)
            support_y = torch.tensor(support_df[target].values, dtype=torch.float32).view(-1, 1)
            mean, std = support_x_raw.mean(0, keepdim=True), support_x_raw.std(0, keepdim=True) + 1e-8
            support_x_norm = (support_x_raw - mean) / std

            embedding = task_encoder(support_x_norm, support_y).numpy()
            task_embeddings.append(embedding)

            task_labels.append(task_name)
            task_type = 'real' if 'real_' in task_name else 'sim'
            task_types.append(task_type)
            relevance_scores.append(task_relevance.get(task_name, {}).get('score', 1.0))

    print("Using t-SNE for dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(task_embeddings) - 1), random_state=42)
    embeddings_2d = tsne.fit_transform(np.array(task_embeddings))

    plt.figure(figsize=(14, 10))

    sim_indices = [i for i, t in enumerate(task_types) if t == 'sim']
    real_indices = [i for i, t in enumerate(task_types) if t == 'real']

    sc = plt.scatter(embeddings_2d[sim_indices, 0], embeddings_2d[sim_indices, 1],
                     c=np.array(relevance_scores)[sim_indices], cmap='viridis',
                     alpha=0.7, s=80, label='Simulated Tasks')


    plt.scatter(embeddings_2d[real_indices, 0], embeddings_2d[real_indices, 1],
                marker='*', c='red', edgecolor='black', s=400, label='Real Task (Training)')

    plt.title("t-SNE", fontsize=18)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(fontsize=12)
    cbar = plt.colorbar(sc)
    cbar.set_label('Relevance Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('t_SNE.svg')
    plt.show()

