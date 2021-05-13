import sys

counts = {}
vocab_size = {}
last_freq = -1
for i, line in enumerate(sys.stdin):
    freq = int(line.strip().split()[1])
    if freq == last_freq:
      counts[freq] += 1
    else:
      counts[freq] = 1
    vocab_size[last_freq] = i
    last_freq = freq
vocab_size[last_freq] = i + 1

freqs = sorted(counts.keys())
cum = [0]
for freq in freqs:
  cum.append(cum[-1] + freq * counts[freq])

#  lines = f.readlines()

#counts = {word: int(freq) for word, freq in [line.strip().split() for line in lines]}
#freqs = sorted(set(counts.values()))

#cum = [0]
#for freq in freqs:
#  if freq == 0:
#    continue
#  cum.append(cum[-1] + freq * len([word for word, word_f in counts.items() if freq == f]))

print("%d total words" % cum[-1])
for freq, c in zip(freqs,cum[1:1001]):
  print("<%d frequency catches %d words (%f) leaving vocab size %d" % (freq+1, c, c*1000//cum[-1]/10, vocab_size[freq]))
print("<%d frequency catches %d words (%f) leaving vocab size %d" % (freqs[-1]+1, cum[-1], cum[-1]*1000//cum[-1]/10, 0))
