import pandas as pd

def analyze_mnist_results(csv_file_path):
    """
    Analyze MNIST results by averaging test accuracy across seeds
    for the same model type and number of parameters.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Filter for MNIST dataset only
    mnist_df = df[df['dataset'] == 'MNIST'].copy()
    
    # Rename model types
    mnist_df['model_type'] = mnist_df['model_type'].replace({
        'tt_regural': 'MPO_type_II',
        'tt_standard': 'MPO_type_II', 
        'tt_type1': 'MPO_type_I'
    })
    
    # Group by model_type, rank, N, CB, and num_parameters, then calculate statistics
    grouped_results = mnist_df.groupby(['model_type', 'r', 'N', 'CB', 'num_parameters']).agg({
        'test_accuracy': ['mean', 'std', 'count'],
        'seed': 'nunique'
    }).round(4)
    
    # Flatten column names
    grouped_results.columns = ['avg_test_accuracy', 'std_test_accuracy', 'num_experiments', 'num_seeds']
    
    # Reset index to make grouping columns regural columns
    grouped_results = grouped_results.reset_index()
    
    # Rename 'r' to 'rank' for clarity
    grouped_results = grouped_results.rename(columns={'r': 'rank'})
    
    # Filter out configurations with less than 5 seeds
    filtered_results = grouped_results[grouped_results['num_seeds'] >= 4].copy()
    
    # Show which configurations were filtered out
    filtered_out = grouped_results[grouped_results['num_seeds'] < 5].copy()
    
    return filtered_results, mnist_df, filtered_out

def get_seed_summary(mnist_df):
    """
    Get summary of seeds used for each model configuration.
    """
    # Group by model configuration and get seed information
    seed_summary = mnist_df.groupby(['model_type', 'r', 'N', 'CB']).agg({
        'seed': ['count', 'nunique', lambda x: sorted(list(x))]
    }).reset_index()
    
    # Flatten column names
    seed_summary.columns = ['model_type', 'rank', 'N', 'CB', 'total_runs', 'unique_seeds', 'seeds_used']
    
    return seed_summary

# Usage:
if __name__ == "__main__":
    # Replace 'your_dataset.csv' with your actual CSV file path
    results_df, mnist_data, filtered_out_df = analyze_mnist_results('amnistres.csv')
    
    print("MNIST Results Analysis (>= 4 seeds):")
    print("=" * 50)
    print(results_df)
    
    if len(filtered_out_df) > 0:
        print(f"\n\nFiltered out configurations (< 5 seeds):")
        print("=" * 50)
        print(filtered_out_df[['model_type', 'rank', 'N', 'CB', 'num_parameters', 'num_seeds']])
    else:
        print("\nNo configurations were filtered out (all have >= 4 seeds)")
    
    # Get seed summary
    seed_summary = get_seed_summary(mnist_data)
    print("\n\nSeed Summary by Model Configuration:")
    print("=" * 60)
    print(seed_summary)
    
    # Save results
    results_df.to_csv('MnistRes.csv', index=False)
    # seed_summary.to_csv('mnist_seed_summary.csv', index=False)
    print(f"\nResults (>= 5 seeds) saved to 'mnist_results_summary.csv'")
    print(f"Seed summary saved to 'mnist_seed_summary.csv'")
    
    # The results_df and seed_summary are pandas DataFrames ready for further analysis
