accs = {}

for seed in range(300):
    try:
        with open(f'/scratch/acd14245px/sst5_1200/{seed}.txt') as f:
            lines = f.readlines()
            l = lines[-6]
            accs[seed] = l.split('=')[1].strip()
    except:
        pass

accs = sorted(accs.items(),key=lambda x:x[1],reverse=True)
with open('sst5_1200_accs.txt','w') as f:
    for seed,acc in accs:
        f.write(f'seed {seed}: {acc}\n')
