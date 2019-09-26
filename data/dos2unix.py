#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py
"""
import os

def reformat(filename):
    content = ''
    outsize = 0
    with open(filename, 'rb') as infile:
        content = infile.read()
    os.remove(filename)
    with open(filename, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line + str.encode('\n'))

for dir in os.listdir():
    if dir[-3:] != '.py':
        files = os.listdir(dir+'/')
        for file in files:
            if file[-2:] == '.p':
                print(file)
                reformat(dir+'/'+file)

print("Done")
