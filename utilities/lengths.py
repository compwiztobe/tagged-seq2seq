import sys,json

lengths = {}

for i, line in enumerate(sys.stdin):
  length = len(line.strip().split())
  if length in lengths:
    lengths[length].append(i+1)
  else:
    lengths[length] = [i+1]
total_lines = i + 1

print("Example lines (to 10 for each length)")
print(json.dumps({l: str(lengths[l][:10]) for l in sorted(lengths)}, indent=2))
print()

cum = {0:0}
for l in sorted(lengths):
  cum[l] = list(cum.values())[-1] + len(lengths[l])

print("Cumulative length distribution")
print(json.dumps({l: str((count, count*10000//total_lines/100)) for l, count in cum.items()}, indent=2))
