#!/usr/bin/env python
import numpy as np

tcpu = np.loadtxt('output/timelost.txt')
tgpu = np.loadtxt('output_gpu/timelost.txt')

lost_count = 0
differ_count = 0
for i, (tc, tg) in enumerate(zip(tcpu, tgpu)):
    if tc < 1e-3:
        lost_count += 1
        if tc != tg: 
            differ_count += 1
            print(f'Tlost {tc-tg=} differ between CPU and GPU times')

print('=== Matching results: ===')
print(f'{differ_count=} of {lost_count=} between CPU vs GPU')


lost_count = 0
differ_count = 0
for i, (tc, tg) in enumerate(zip(tcpu, tgpu)):
    if tg < 1e-3:
        lost_count += 1
        if tc != tg: 
            differ_count += 1

print(f'{differ_count=} of {lost_count=} between GPU vs CPU')
