import sqlite3
from Bio import SeqIO
import os
from torch.utils.data import Dataset,DataLoader

conn = sqlite3.connect("demo.db")
db_cursor = conn.cursor()#create train sequences table

db_cursor.execute('''CREATE TABLE train_sequences
             (id INTEGER PRIMARY KEY,
              header TEXT,
              sequence TEXT)''')

#create test sequences table
db_cursor.execute('''CREATE TABLE test_sequences
             (id INTEGER PRIMARY KEY,
              header TEXT,
              sequence TEXT)''')



training_seqs = SeqIO.parse(open('spikeprot0203.clean.uniq.training.fasta'),'fasta')

for i, fasta in enumerate(training_seqs):
    header, seq = fasta.id, str(fasta.seq)
    db_cursor.execute("INSERT INTO train_sequences (header, sequence) VALUES (?,?)", (header,seq))

conn.commit()
conn.close()

conn = sqlite3.connect("demo.db")
db_cursor = conn.cursor()

testing_seqs = SeqIO.parse(open(os.path.abspath('../data/spikeprot0203.clean.uniq.testing.fasta')), 'fasta')

for i, fasta in enumerate(testing_seqs):
    header, seq = fasta.id, str(fasta.seq)
    db_cursor.execute("INSERT INTO test_sequences (header, sequence) VALUES (?,?)", (header,seq))
    
conn.commit()
conn.close()
