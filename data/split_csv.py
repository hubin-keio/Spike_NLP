"""Randomly split a fasta file into two files"""

import os
import random


def split_csv(rnd_seed: int, input_csv: str, train_csv: str, test_csv: str):

    train_count = 0
    test_count = 0
    total_count = 0

    data = {}

    with open(input_csv, "r") as input:
        header = input.readline()
        counter = 0

        # entry = {"seq_id": "",
        #          "seq": "",
        #          "embedding": "",
        #          "log10_ka": ""}

        for row in input:
            counter += 1
            if counter == 10:
                exit()
            row = row.strip()

            print(row)
 





    # random.seed(rnd_seed)
    # random.shuffle(input_records)

    # with open(train_csv, 'w') as ft, open(test_csv, 'w') as fv:

    #     for record in input_records:
    #         rnd = random.random()
    #         if rnd > 0.2:
    #             SeqIO.write(record, ft, 'csv')
    #             train_count += 1
    #         else:
    #             SeqIO.write(record, fv, 'csv')
    #             test_count += 1
    #         total_count += 1
    # print(f'Total: {total_count}, Train: {train_count}, Test: {test_count}')


if __name__ == '__main__':
    root_dir = os.path.dirname(__file__)
    input_csv = os.path.join(root_dir, 'mutation_binding_Kds_embedded_first.csv')
    train_csv = input_csv.replace('.csv', '_train.csv')
    test_csv  = input_csv.replace('.csv', '_test.csv')

    rnd_seed = 0
    split_csv(rnd_seed, input_csv, train_csv, test_csv)
