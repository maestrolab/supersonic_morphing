#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py
"""
original = "./abaqus_outputs/outputs_big_front-2.p"
destination = "./abaqus_outputs/outputs_big_front-2_ux.p"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s by    tes." % (len(content)-outsize))
