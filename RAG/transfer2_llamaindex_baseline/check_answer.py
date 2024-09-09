import sys
import json

ref = sys.argv[1]
pred = sys.argv[2]

with open(ref,encoding='utf8') as f:
    ref_data = [json.loads(l) for l in f.readlines()]

with open(pred,encoding='utf8') as f:
    pred_data = [json.loads(l) for l in f.readlines()]

p_qid = {}
for i, p in enumerate(pred_data):
    p_qid[p['qid']] = i

correct = 0
not_found = 0
total = 0

for i,r in enumerate(ref_data):
    if r['qid'] not in p_qid:
        not_found += 1
        continue

    p = pred_data[p_qid[r['qid']]]

    for ans in r['answers']:
        if ans == p['prediction']:
            correct += 1
            break
    total += 1

print('correct rate = {} ({}/{})'.format(correct/total,correct,total))
if not_found > 0:
    print('WARN: not_found = {}'.format(not_found))
