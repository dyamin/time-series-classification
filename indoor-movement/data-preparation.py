# load mapping files
from os import listdir
from pandas import read_csv
from numpy import vstack
from numpy import savetxt
from numpy import array
from matplotlib import pyplot
from numpy import pad

# return list of traces, and arrays for targets, groups and paths


def load_dataset(prefix='assets/'):
    grps_dir, data_dir = prefix+'groups/', prefix+'dataset/'
    # load mapping files
    targets = read_csv(data_dir + 'MovementAAL_target.csv', header=0)
    groups = read_csv(grps_dir + 'MovementAAL_DatasetGroup.csv', header=0)
    paths = read_csv(grps_dir + 'MovementAAL_Paths.csv', header=0)
    # load traces
    sequences = list()
    target_mapping = None
    for name in listdir(data_dir):
        filename = data_dir + name
        if filename.endswith('_target.csv'):
            continue
        df = read_csv(filename, header=0)
        values = df.values
        sequences.append(values)
    return sequences, targets.values[:, 1], groups.values[:, 1], paths.values[:, 1]


# load dataset
sequences, targets, groups, paths = load_dataset()
# summarize shape of the loaded data
print(len(sequences), targets.shape, groups.shape, paths.shape)

# summarize class breakdown
class1, class2 = len(targets[targets == -1]), len(targets[targets == 1])
print('Class=-1: %d %.3f%%' % (class1, class1/len(targets)*100))
print('Class=+1: %d %.3f%%' % (class2, class2/len(targets)*100))

# group sequences by paths
paths = [1, 2, 3, 4, 5, 6]
seq_paths = dict()
for path in paths:
    seq_paths[path] = [sequences[j]
                       for j in range(len(paths)) if paths[j] == path]
# plot one example of a trace for each path
pyplot.figure()
for i in paths:
    pyplot.subplot(len(paths), 1, i)
    # line plot each variable
    for j in [0, 1, 2, 3]:
        pyplot.plot(seq_paths[i][0][:, j], label='Anchor ' + str(j+1))
    pyplot.title('Path ' + str(i), y=0, loc='left')
pyplot.show()


# create a fixed 1d vector for each trace with output variable
def create_dataset(sequences, targets):
        # create the transformed dataset
    transformed = list()
    n_vars, n_steps, max_length = 4, 25, 200
    # process each trace in turn
    for i in range(len(sequences)):
        seq = sequences[i]
        # pad sequences
        seq = pad(seq, ((max_length-len(seq), 0), (0, 0)),
                  'constant', constant_values=(0.0))
        vector = list()
        # last n observations
        for row in range(1, n_steps+1):
            for col in range(n_vars):
                vector.append(seq[-row, col])
        # add output
        vector.append(targets[i])
        # store
        transformed.append(vector)
    # prepare array
    transformed = array(transformed)
    transformed = transformed.astype('float32')
    return transformed


# load dataset
sequences, targets, groups, paths = load_dataset()
# separate traces
seq1 = [sequences[i] for i in range(len(groups)) if groups[i] == 1]
seq2 = [sequences[i] for i in range(len(groups)) if groups[i] == 2]
seq3 = [sequences[i] for i in range(len(groups)) if groups[i] == 3]
# separate target
targets1 = [targets[i] for i in range(len(groups)) if groups[i] == 1]
targets2 = [targets[i] for i in range(len(groups)) if groups[i] == 2]
targets3 = [targets[i] for i in range(len(groups)) if groups[i] == 3]
# create ES1 dataset
es1 = create_dataset(seq1+seq2, targets1+targets2)
print('ES1: %s' % str(es1.shape))
savetxt('assets/es1.csv', es1, delimiter=',')
# create ES2 dataset
es2_train = create_dataset(seq1+seq2, targets1+targets2)
es2_test = create_dataset(seq3, targets3)
print('ES2 Train: %s' % str(es2_train.shape))
print('ES2 Test: %s' % str(es2_test.shape))
savetxt('assets/es2_train.csv', es2_train, delimiter=',')
savetxt('assets/es2_test.csv', es2_test, delimiter=',')
