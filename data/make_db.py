#!/usr/bin/env python
import sqlite3
from Bio import SeqIO
import os

db_name = "SARS_CoV_2_spike_noX_RBD.db"
conn = sqlite3.connect(db_name)
db_cursor = conn.cursor()

#create train, test sequences table
db_cursor.execute('''CREATE TABLE train (id INTEGER PRIMARY KEY AUTOINCREMENT, header TEXT, sequence TEXT)''')
db_cursor.execute('''CREATE TABLE test (id INTEGER PRIMARY KEY AUTOINCREMENT, header TEXT, sequence TEXT)''')

training_seqs = SeqIO.parse(open('spikeprot0528.clean.uniq.noX.RBD_train.fasta'),'fasta')

for i, fasta in enumerate(training_seqs):
    header, seq = fasta.id, str(fasta.seq)
    db_cursor.execute("INSERT INTO train (header, sequence) VALUES (?,?)", (header, seq))
conn.commit()

testing_seqs = SeqIO.parse(open(os.path.abspath('spikeprot0528.clean.uniq.noX.RBD_test.fasta')), 'fasta')

for i, fasta in enumerate(testing_seqs):
    header, seq = fasta.id, str(fasta.seq)
    db_cursor.execute("INSERT INTO test (header, sequence) VALUES (?,?)", (header, seq))
conn.commit()

conn.close()
print(f'Database {db_name} initialized.')