import argparse
from pathlib import Path

import pandas as pd


SLOWDOWN_FILE = Path('input/slowdown_stats_output_device_0.xlsx')
FEATURES_FILE = Path('input/kernel_metric_output.csv')
SLOWDOWN_KERNEL_COL = 'kShortName'
FEATURES_KERNEL_COL = 'Kernel Name'
_HELPER_CANONICAL_COL = '_canonical_kernel_name'
_HELPER_OCCURRENCE_COL = '_kernel_occurrence_index'
_NORMALIZATION_SUFFIXES = (
    '_vectorized',
    '_aligned16_contig',
    '_with_index',
)


def normalize_kernel_name(name: object) -> str:
    kernel_name = str(name)
    for suffix in _NORMALIZATION_SUFFIXES:
        if kernel_name.endswith(suffix):
            kernel_name = kernel_name[: -len(suffix)]
    return kernel_name


def _with_join_keys(dataframe: pd.DataFrame, kernel_name_col: str) -> pd.DataFrame:
    keyed_dataframe = dataframe.copy()
    keyed_dataframe[_HELPER_CANONICAL_COL] = keyed_dataframe[kernel_name_col].map(normalize_kernel_name)
    keyed_dataframe[_HELPER_OCCURRENCE_COL] = keyed_dataframe.groupby(_HELPER_CANONICAL_COL).cumcount()
    return keyed_dataframe


def merge_features(slowdown_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    keyed_slowdown_df = _with_join_keys(slowdown_df, SLOWDOWN_KERNEL_COL)
    keyed_features_df = _with_join_keys(features_df, FEATURES_KERNEL_COL)

    merged_df = keyed_slowdown_df.merge(
        keyed_features_df,
        on=[_HELPER_CANONICAL_COL, _HELPER_OCCURRENCE_COL],
        how='inner',
        suffixes=('', '_feature'),
        sort=False,
    )

    if merged_df.empty:
        raise RuntimeError(
            'No matched kernels were found between slowdown stats and kernel metrics. '
            'Please verify that the collected slowdown kernels and kernel metric CSV come from compatible runs.'
        )

    return merged_df.drop(columns=[_HELPER_CANONICAL_COL, _HELPER_OCCURRENCE_COL])


def main() -> None:
    parser = argparse.ArgumentParser(description='Merge slowdown Excel and kernel metrics CSV based on normalized kernel names.')
    parser.add_argument('--output', type=str, default='merged_file.csv', help='Path to save the merged CSV file.')
    args = parser.parse_args()

    slowdown_df = pd.read_excel(SLOWDOWN_FILE)
    features_df = pd.read_csv(FEATURES_FILE)

    print('slowdown_df.head()', slowdown_df.head())
    print('features_df.head()', features_df.head())

    merged_df = merge_features(slowdown_df, features_df)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    print(f'Merged rows: {len(merged_df)}')
    print(f'Merged kernels: {merged_df[SLOWDOWN_KERNEL_COL].nunique()} slowdown names, {merged_df[FEATURES_KERNEL_COL].nunique()} feature names')
    print(f'Merged file saved to: {output_path}')


if __name__ == '__main__':
    main()
