import pickle

f = open('outputs_test.p', 'rb')
data = pickle.load(f)

print(max(data['U']['Step-3'][0][:,1]))
print(min(data['U']['Step-3'][0][:,1]))