#!/usr/bin/env python
import sqlite3
from Bio import SeqIO
import os
import csv

def extract_csv(train_file:str, test_file:str):
    def read_csv(file_path:str):
        with open(train_file, "r") as input:
            reader = csv.reader(input)
            header = next(reader)
            return list(reader)

    return read_csv(train_file), read_csv(test_file)

def extract_fasta(train_file:str, test_file:str):
    training_seqs = list(SeqIO.parse(open(train_file),'fasta'))
    testing_seqs = list(SeqIO.parse(open(test_file), 'fasta'))
    return training_seqs , testing_seqs

def make_alphaseq_db(db_name:str, train_file:str, test_file:str):
    training_seqs, testing_seqs = extract_csv(train_file, test_file)
    conn = sqlite3.connect(db_name)
    db_cursor = conn.cursor()

    # Create train, test sequences table
    create_table_sql = '''CREATE TABLE {} (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                           POI TEXT,
                                           Sequence TEXT,
                                           Target TEXT,
                                           Assay TEXT,
                                           Replicate TEXT,
                                           Pred_affinity TEXT,
                                           HC TEXT,
                                           LC TEXT,
                                           CDRH1 TEXT,
                                           CDRH2 TEXT,
                                           CDRH3 TEXT,
                                           CDRL1 TEXT,
                                           CDRL2 TEXT,
                                           CDRL3 TEXT,
                                           Mean_Affinity TEXT)'''

    db_cursor.execute(create_table_sql.format("train"))
    db_cursor.execute(create_table_sql.format("test"))

    for mode, table_name in zip([training_seqs, testing_seqs], ["train", "test"]):
        for record in mode:
            db_cursor.execute(f"INSERT INTO {table_name} (POI, Sequence, Target, Assay, Replicate, Pred_affinity, HC, LC, CDRH1, CDRH2, CDRH3, CDRL1, CDRL2, CDRL3, Mean_Affinity) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", record)
        conn.commit()

    conn.close()
    print(f'Database {db_name} initialized.')

def make_fasta_db(db_name:str, train_file:str, test_file:str):
    training_seqs, testing_seqs = extract_fasta(train_file, test_file)
    conn = sqlite3.connect(db_name)
    db_cursor = conn.cursor()

    # Create train, test sequences table
    db_cursor.execute('''CREATE TABLE train (id INTEGER PRIMARY KEY AUTOINCREMENT, header TEXT, sequence TEXT)''')
    db_cursor.execute('''CREATE TABLE test (id INTEGER PRIMARY KEY AUTOINCREMENT, header TEXT, sequence TEXT)''')

    for mode, table_name in zip([training_seqs, testing_seqs], ["train", "test"]):
        for fasta in mode:
            header, seq = fasta.id, str(fasta.seq)
            db_cursor.execute(f"INSERT INTO {table_name} (header, sequence) VALUES (?,?)", (header, seq))
        conn.commit()

    conn.close()
    print(f'Database {db_name} initialized.')

if __name__ == '__main__':
    data_dir = os.path.dirname(__file__)

    # AlphaSeq
    # train_csv = os.path.join(data_dir, 'alphaseq/clean_avg_alpha_seq_test.csv')
    # test_csv = os.path.join(data_dir, 'alphaseq/clean_avg_alpha_seq_train.csv')
    # db_name = 'AlphaSeq.db'
    # make_alphaseq_db(db_name, train_csv, test_csv)

    # RBD
    # train_fasta = os.path.join(data_dir, 'spike/spikeprot0528.clean.uniq.noX.RBD_train.fasta')
    # test_fasta = os.path.join(data_dir, 'spike/spikeprot0528.clean.uniq.noX.RBD_test.fasta')
    # db_name = 'SARS_CoV_2_spike_noX_RBD.db'
    # make_fasta_db(db_name, train_fasta, test_fasta)