import json

def reset_results_file(save_results):
    # Initialize the dictionaries storing the results.
    
    data = {
        'MLP': {},
        'GNN': {},
        'Conv1d': {}    
    }

    num_features = {}
    num_features['Conv1d'] = [4, 8, 16, 32, 64, 128, 256, 360, 512, 1024, 2048]
    num_features['MLP'] = [256, 360, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    num_features['GNN'] = [256, 360, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    for model_name in data.keys():
        for depth in [1, 2, 3]:
            data[model_name][depth] = {}

            for feature in num_features[model_name]:
                data[model_name][depth][feature] = -1

    with open(f'{save_results}/results.json', 'w') as file:  # 'IBC_maml++-IBC.json'
        json.dump(data, file)

    data = {
        'MLP': {},
        'GNN': {},
        'Conv1d': {}   
    }
    for model_name in data.keys():
        for item in ['best_val_acc', 'depth', 'num_features', 'test_acc', 'test_std']:
            data[model_name][item] = 0

    with open(f'{save_results}/best_results.json', 'w') as file:  # 'IBC_maml++-IBC.json'
        json.dump(data, file)

def update_results_file(save_results, model_name, depth, num_features, best_val_acc, test_acc, test_std):
    with open(f'{save_results}/results.json') as file:
        results = json.load(file)
        results[model_name][str(depth)][str(num_features)] = best_val_acc

    with open(f'{save_results}/results.json', 'w') as file:
        json.dump(results, file)

    with open(f'{save_results}/best_results.json') as file:  # 'IBC_maml++-IBC.json'
        results = json.load(file)
        if results[model_name]['best_val_acc'] <= best_val_acc:
            results[model_name]['best_val_acc'] = best_val_acc
            results[model_name]['depth'] = depth
            results[model_name]['num_features'] = num_features
            results[model_name]['test_acc'] = test_acc
            results[model_name]['test_std'] = test_std
    with open(f'{save_results}/best_results.json', 'w') as file:
        json.dump(results, file)