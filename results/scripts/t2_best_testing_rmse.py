#!/usr/bin/env python
"""
For each of our models, return the best testing rmse value.
"""

import os
import numpy as np
import pandas as pd

def get_best_test_rmse(csv_file:str):
    """ Calculate the minimum test RMSE from given .csv file. """
    df = pd.read_csv(csv_file, sep=',', header=0)

    # Check which column is present and calculate the minimum RMSE
    if 'test_rmse' in df.columns:
        return df['test_rmse'].min()
    elif 'test_blstm_rmse' in df.columns:
        return df['test_blstm_rmse'].min()
    else:
        raise ValueError("Neither 'test_rmse' nor 'test_blstm_rmse' found in the CSV file.")


if __name__=='__main__':
    results_dir = os.path.join(os.path.dirname(__file__), f'../run_results')

    # Binding
    print("Binding")
    print(f"FCN w/ ESM, no embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'fcn/fcn-esm_dms_binding-2023-10-17_15-38/fcn-esm_dms_binding-2023-10-17_15-38_train_84420_test_21105_metrics_per.csv'))}")
    print(f"FCN w/ RBD Learned, no embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'fcn/fcn-rbd_learned_320_dms_binding-2023-12-21_09-44/fcn-rbd_learned_320_dms_binding-2023-12-21_09-44_train_84420_test_21105_metrics_per.csv'))}")

    print(f"GCN w/ ESM, no embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'graphsage/graphsage-esm_dms_binding-2023-12-13_18-14/graphsage-esm_dms_binding-2023-12-13_18-14_train_84420_test_21105_metrics_per.csv'))}")
    print(f"GCN w/ RBD Learned, no embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'graphsage/graphsage-rbd_learned_320_dms_binding-2024-01-02_14-50/graphsage-rbd_learned_320_dms_binding-2024-01-02_14-50_train_84420_test_21105_metrics_per.csv'))}")

    print(f"BLSTM w/ ESM, no embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'blstm/blstm-esm_dms_binding-2023-12-07_13-53/blstm-esm_dms_binding-2023-12-07_13-53_train_84420_test_21105_metrics_per.csv'))}")
    print(f"BLSTM w/ RBD Learned, no embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'blstm/blstm-rbd_learned_320_dms_binding-2024-01-04_14-12/blstm-rbd_learned_320_dms_binding-2024-01-04_14-12_train_84420_test_21105_metrics_per.csv'))}")

    print(f"ESM-BLSTM, embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'esm-blstm/esm-blstm-esm_dms_binding-2023-12-12_17-02/esm-blstm-esm_dms_binding-2023-12-12_17-02_train_84420_test_21105_metrics_per.csv'))}")
    print(f"BERT-BLSTM w/ ESM Init, embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'bert_blstm_esm/bert_blstm_esm-dms_binding-2023-12-23_21-54/bert_blstm_esm-dms_binding-2023-12-23_21-54_train_84420_test_21105_metrics_per.csv'))}")    
    print(f"BERT-BLST w/ RBD Learned Init, embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'bert_blstm/bert_blstm-dms_binding-2023-11-22_23-03/bert_blstm-dms_binding-2023-11-22_23-03_train_84420_test_21105_metrics_per.csv'))}")

    # Expression
    print("\nExpression")
    print(f"FCN w/ ESM, no embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'fcn/fcn-esm_dms_expression-2023-12-19_15-53/fcn-esm_dms_expression-2023-12-19_15-53_train_93005_test_23252_metrics_per.csv'))}")
    print(f"FCN w/ RBD Learned, no embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'fcn/fcn-rbd_learned_320_dms_expression-2023-12-20_12-02/fcn-rbd_learned_320_dms_expression-2023-12-20_12-02_train_93005_test_23252_metrics_per.csv'))}")

    print(f"GCN w/ ESM, no embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'graphsage/graphsage-esm_dms_expression-2023-12-20_08-48/graphsage-esm_dms_expression-2023-12-20_08-48_train_93005_test_23252_metrics_per.csv'))}")
    print(f"GCN w/ RBD Learned, no embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'graphsage/graphsage-rbd_learned_320_dms_expression-2023-12-25_00-54/graphsage-rbd_learned_320_dms_expression-2023-12-25_00-54_train_93005_test_23252_metrics_per.csv'))}")

    print(f"BLSTM w/ ESM, no embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'blstm/blstm-esm_dms_expression-2023-12-07_13-58/blstm-esm_dms_expression-2023-12-07_13-58_train_93005_test_23252_metrics_per.csv'))}")
    print(f"BLSTM w/ RBD Learned, no embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'blstm/blstm-rbd_learned_320_dms_expression-2024-01-04_14-15/blstm-rbd_learned_320_dms_expression-2024-01-04_14-15_train_93005_test_23252_metrics_per.csv'))}")

    print(f"ESM-BLSTM, embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'esm-blstm/esm-blstm-esm_dms_expression-2023-12-12_16-58/esm-blstm-esm_dms_expression-2023-12-12_16-58_train_93005_test_23252_metrics_per.csv'))}")
    print(f"BERT-BLSTM w/ ESM Init, embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'bert_blstm_esm/bert_blstm_esm-dms_expression-2023-12-23_21-45/bert_blstm_esm-dms_expression-2023-12-23_21-45_train_93005_test_23252_metrics_per.csv'))}")    
    print(f"BERT-BLST w/ RBD Learned Init, embedding updated: {get_best_test_rmse(os.path.join(results_dir, 'bert_blstm/bert_blstm-dms_expression-2023-11-22_23-05/bert_blstm-dms_expression-2023-11-22_23-05_train_93005_test_23252_metrics_per.csv'))}")