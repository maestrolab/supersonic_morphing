#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py
"""
original = "./loudness/loudness_small_simple_noTE_Heat2_fix1.p"
destination = "./loudness_small_simple_noTE_Heat2_fix1.p"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s by    tes." % (len(content)-outsize))
