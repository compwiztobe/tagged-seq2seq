import sys

with open(sys.argv[1]) as f:
  lines = f.readlines()

for line in sys.stdin:
  print(lines[int(line.strip())], end='')
