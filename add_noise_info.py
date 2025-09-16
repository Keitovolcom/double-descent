import os
import pandas as pd
import warnings
import argparse

# --- 既存のプロジェクトモジュールをインポート ---
from datasets import load_or_create_noisy_dataset 

def add_noise_info_to_csv(args):
    """
    既存のグループ1のCSVファイルに、データセットのノイズ情報を後付けで追加する。
    v3: テンソル形式のノイズ情報を正しく処理する。
    """
    print("--- Starting: Add Noise Info to CSV (v3) ---")
    
    # --- 1. 引数に基づいてCSVファイルのパスを構築 ---
    print("\nConstructing file path from arguments...")
    experiment_name = (
        f'test_seed_{args.fix_seed}width{args.model_width}_{args.model}_'
        f'{args.dataset}_variance{args.variance}_{args.target}_lr{args.lr}_'
        f'batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_'
        f'Optim{args.optimizer}_Momentum{args.momentum}'
    )
    base_save_dir = f"save_model/{args.dataset}/noise_{args.label_noise_rate}/{experiment_name}"
    
    # ユーザーが前回ソートして作成したファイル名を優先的に探す
    # このファイルはスクリプトと同じ階層にあることを想定
    sorted_filename = f"sorted_group1_a{args.epoch_a}_t{args.epoch_t}k{args.epoch_k}_indices_with_counts.csv"
    original_filename = f"group1_a{args.epoch_a}_t{args.epoch_t}k{args.epoch_k}_indices_with_counts.csv"
    input_csv_path_original = os.path.join(base_save_dir, "grouped_data_indices_csv", original_filename)

    input_csv_path = None
    if os.path.exists(sorted_filename):
        input_csv_path = sorted_filename
        print(f"Found sorted input file: '{input_csv_path}'")
    elif os.path.exists(input_csv_path_original):
        input_csv_path = input_csv_path_original
        print(f"Found original input file: '{input_csv_path}'")
    else:
        print(f"Error: Input CSV file not found.")
        print(f"Checked for sorted file at: '{sorted_filename}'")
        print(f"Checked for original file at: '{input_csv_path_original}'")
        return
        
    output_csv_filename = os.path.basename(input_csv_path).replace('.csv', '_with_noise_info.csv')

    print(f"Reading existing data from: {input_csv_path}")
    try:
        # ご提示いただいたCSVのヘッダーは 'sample_index' と 'count' のようでしたので、それに合わせます
        df = pd.read_csv(input_csv_path)
        if 'sample_index' not in df.columns:
            # もしヘッダーがない場合は、列名を指定して読み直す
            df = pd.read_csv(input_csv_path, names=['sample_index', 'count'], header=None)

    except Exception as e:
        print(f"Failed to read CSV file. Error: {e}")
        return

    # --- 2. データセットをロードしてノイズ情報を取得 ---
    print("\nLoading dataset to retrieve noise information...")
    try:
        train_dataset, _, _ = load_or_create_noisy_dataset(
            args.dataset, args.target, args.gray_scale, args, return_type="torch"
        )
        # ★修正点 1/2: ご提示のNoisyDatasetクラスに合わせて `noise_info` 属性から取得
        noise_info_list = train_dataset.noise_info
        print("Successfully retrieved noise information list.")
    except AttributeError:
        print("\nError: Could not find 'noise_info' attribute in the dataset.")
        print("Please ensure your 'load_or_create_noisy_dataset' function returns a dataset object")
        print("with a 'noise_info' attribute as defined in your 'NoisyDataset' class.")
        return
    except Exception as e:
        print(f"\nAn error occurred while loading the dataset: {e}")
        return

    # --- 3. 取得したノイズ情報をCSVデータに結合 ---
    print("\nAdding 'is_noisy' column to the data...")
    
    # ★修正点 2/2: テンソルから数値を取り出し(.item())、それをbool値に変換する
    # dfの各行の 'sample_index' に対応するノイズ情報を取得し、新しい列 'is_noisy' を作成
    try:
        df['is_noisy'] = df['sample_index'].apply(lambda index: bool(noise_info_list[index].item()))
    except AttributeError:
         # .item() が失敗した場合（中身がテンソルでない場合）のフォールバック
         df['is_noisy'] = df['sample_index'].apply(lambda index: bool(noise_info_list[index]))


    # --- 4. 結果を新しいCSVファイルに保存 ---
    try:
        # index=False をつけて、DataFrameのインデックスがCSVに保存されないようにする
        df.to_csv(output_csv_filename, index=False)
        print(f"\nSuccessfully created new CSV file: '{output_csv_filename}'")
        print("\nPlease check the new file to see if the 'is_noisy' column is now correct (True/False).")
    except Exception as e:
        print(f"Failed to save the new CSV file. Error: {e}")

    print("\n--- Process Complete ---")

def main():
    """
    スクリプト実行のためのメイン関数
    """
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Add noise information to an existing CSV file (v3).')
    
    # パス構築とデータセットロードに必要な全ての引数を追加
    parser.add_argument('--fix_seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='cnn_5layers')
    parser.add_argument('--model_width', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='emnist_digits')
    parser.add_argument('--target', type=str, default='combined')
    parser.add_argument('--label_noise_rate', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--variance', type=float, default=0.0)
    parser.add_argument('--gray_scale', action='store_true')
    parser.add_argument('--epoch_a', type=int, default=27)
    parser.add_argument('--epoch_t', type=int, default=28)
    parser.add_argument('--epoch_k', type=int, default=120)
    parser.add_argument('--weight_noisy', type=float, default=1.0)
    parser.add_argument('--weight_clean', type=float, default=1.0)
    
    args = parser.parse_args()
    add_noise_info_to_csv(args)

if __name__ == '__main__':
    main()