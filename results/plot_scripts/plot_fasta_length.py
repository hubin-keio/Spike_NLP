'''Plot sequence length distribution with density function'''

import matplotlib.pyplot as plt

def read_fasta_file(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return sequences

def calculate_sequence_lengths(sequences):
    sequence_lengths = [len(seq) for seq in sequences]
    return sequence_lengths

def plot_sequence_length_distribution(sequence_lengths):
    plt.hist(sequence_lengths, bins=30, density=True, alpha=0.7)
    plt.xlabel('Sequence Length')
    plt.ylabel('Density')
    plt.title('Sequence Length Distribution')
    plt.show()

if __name__ == '__main__':
    import os
    root_dir = os.path.join(os.path.dirname(__file__), '../../../')
    fasta_file = 'data/spike/spikeprot0203.clean.uniq.noX.RBD.fasta'
    fasta_file = os.path.join(root_dir, fasta_file)
    sequences = read_fasta_file(fasta_file)
    sequence_lengths = calculate_sequence_lengths(sequences)
    plot_sequence_length_distribution(sequence_lengths)
