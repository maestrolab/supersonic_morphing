import pickle

# if "_pickle.UnpicklingError: the STRING opcode argument must be quoted" error,
# convert outputs pickle file to unix file endings using dos2unix.py in data folder
f = open('../data/abaqus_outputs/outputs_big_front-2_ux.p', 'rb')
data = pickle.load(f, encoding='latin1')

print(max(data['U']['Step-3'][0][:,1]))
print(min(data['U']['Step-3'][0][:,1]))