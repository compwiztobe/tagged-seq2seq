import sys

MIN_LEN = int(sys.argv[1])
MAX_LEN = int(sys.argv[2])
MAX_RATIO = float(sys.argv[3])

#INPUT_PREFIX = sys.argv[4]
#OUTPUT_PREFIX = sys.argv[7]
#SRC_LANG = sys.argv[5]
#TRG_LANG = sys.argv[6]

while True:
  try:
    src = next(sys.stdin)
    trg = next(sys.stdin)
  except StopIteration:
    break
#with open(INPUT_PREFIX+'.'+SRC_LANG) as f_src, open(INPUT_PREFIX+'.'+TRG_LANG) as f_trg, open(OUTPUT_PREFIX+'.'+SRC_LANG,'w') as out_src, open(OUTPUT_PREFIX+'.'+TRG_LANG,'w') as out_trg):
# for src,trg in zip(f_src,f_trg):
  src_len = len(src.strip().split())
  trg_len = len(trg.strip().split())

  # filter conditions
  if src_len > MAX_LEN or trg_len > MAX_LEN:
    continue
  if src_len < MIN_LEN or trg_len < MIN_LEN:
    continue
  if src_len == 0 or trg_len == 0:
    continue
  len_ratio = src_len / trg_len
  if len_ratio > MAX_RATIO or 1/len_ratio > MAX_RATIO:
    continue

  # finally
#  out_src.write(src)
#  out_trg.write(trg)
  print(src, end='')
  print(trg, end='')
