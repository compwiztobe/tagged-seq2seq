import sys

sep = sys.argv[1]

for line in sys.stdin:
  print(' '.join(pair.split(sep)[0] for pair in line.strip().split()))
