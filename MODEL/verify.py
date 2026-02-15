#!/usr/bin/env python3
'''Verify all required files exist'''
import os

required_files = [
    'config.py',
    'main.py',
    'README.md',
    'requirements.txt',
    'demo.py',
    'data/generate_data.py',
    'models/tft_model.py',
    'models/health_model.py',
    'models/maintenance_model.py',
    'training/train_tft.py',
    'training/train_health.py',
    'training/train_maintenance.py',
    'utils/utils.py',
    'utils/datasets.py',
]

print("Checking required files...")
missing = []
for f in required_files:
    if os.path.exists(f):
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ {f} [MISSING]")
        missing.append(f)

if missing:
    print(f"\n❌ {len(missing)} files missing")
else:
    print(f"\n✅ All {len(required_files)} files present!")
