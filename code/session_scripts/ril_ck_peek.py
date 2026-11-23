import torch, sys
for p in sys.argv[1:]:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    print(p)
    for k, v in ck.items():
        if k.endswith('state_dict') or k in ('snp_projector', 'optimizer_state_dict', 'ema_state_dict'):
            continue
        s = repr(v)
        print('  ', k, ':', s[:300])
    for k in ('val_genotypes', 'test_genotypes', 'train_genotypes'):
        if k in ck:
            print('  len', k, len(ck[k]))
