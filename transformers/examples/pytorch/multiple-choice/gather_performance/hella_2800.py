accs = {}

for seed in range(300):
    try:
        with open(f'/scratch/acd14245px/hella_2800/{seed}.txt') as f:
            l = f.readlines()[-6]
            accs[seed] = float(l.split('=')[1].strip())
    except:
        pass

accs = sorted(accs.items(),key=lambda x:x[1],reverse=True)
with open('hella_2800_accs.txt','w') as f:
    for seed,acc in accs:
        f.write(f'seed {seed}: {acc}\n')

