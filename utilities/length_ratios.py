import sys,json,itertools

ratios = {}

#for i in itertools.count(start=1):
#  try:
#    src = next(sys.stdin)
#    trg = next(sys.stdin)
#  except StopIteration:
#    break
with open(sys.argv[1]) as f_src, open(sys.argv[2]) as f_trg:
 for i,src,trg in zip(itertools.count(start=1),f_src,f_trg):
  src_len = len(src.strip().split())
  trg_len = len(trg.strip().split())
  try:
    ratio = src_len/trg_len
  except ZeroDivisionError:
    ratio = 0
#  if ratio < 1 and ratio != 0:
#    ratio = 1/ratio
  if ratio in ratios:
    ratios[ratio].append(i)
  else:
    ratios[ratio] = [i]
total_pairs = i

print("Example lines (to 10 for each ratio)")
print(json.dumps({r: str(ratios[r][:10]) for r in sorted(ratios)}, indent=2))
print()

cum = {0:0}
for r in sorted(ratios):
  cum[r] = list(cum.values())[-1] + len(ratios[r])

print("Cumulative length distribution")
print(json.dumps([str(r if r>1 or r==0 else 1/r)+": "+str((count, count*10000//total_pairs/100)) for r, count in cum.items()], indent=2))
