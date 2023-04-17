"""Dataset"""


import sqlite3
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader


def initialize_db(db_file_path: str, train_fasta: str, test_fasta: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_file_path)
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE train (id INTEGER PRIMARY KEY AUTOINCREMENT, header TEXT, sequence TEXT)''')
    cur.execute('''CREATE TABLE test (id INTEGER PRIMARY KEY AUTOINCREMENT, header TEXT, sequence TEXT)''')
    
    training_seqs = SeqIO.parse(open(train_fasta),'fasta')

    for i, fasta in enumerate(training_seqs):
        header, seq = fasta.id, str(fasta.seq)
        cur.execute("INSERT INTO train (header, sequence) VALUES (?,?)", (header, seq))
    conn.commit()

    test_seqs = SeqIO.parse(open(test_fasta), "fasta")

    for i, fasta in enumerate(test_seqs):
        header, seq = fasta.id, str(fasta.seq)
        cur.execute("INSERT INTO test (header, sequence) VALUES (?,?)", (header, seq))
    conn.commit()    

    print(f'Database {db_file_path} initialized.')
    return conn


class SeqDataset(Dataset):
    """
    Create Dataset compatible indexing of fasta file

    db_file: sqlite3 database file
    table_name: table name inside the sqlite database

    """
    def __init__(self, db_file: str, table_name: str) -> None:
        #TODO: check if db_file exists and suggests initilaizae_db if db does not exists.
        self.db_file = db_file
        self.table = table_name
        self.conn = None    # Use lazy loading
                
    def __getitem__(self, idx):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_file, isolation_level=None)  # Read only operations in sqlite connection.

        cur = self.conn.cursor()
        _, header, sequence = cur.execute(f'''SELECT * FROM {self.table} LIMIT 1 OFFSET {idx}''').fetchone()

        
        return header, sequence

    def __len__(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_file, isolation_level=None)  # Read only operations in sqlite connection.

        cur = self.conn.cursor()
        total_seq = cur.execute(f'''SELECT COUNT(*) as total_seq FROM {self.table}''').fetchone()[0]
        return total_seq
