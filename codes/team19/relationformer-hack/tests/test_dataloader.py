from absl.testing import absltest

import os
import sys
import matplotlib.pyplot as plt
import time

sys.path.append('..')
from dataloading import data_loading, preprocessing

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def afm_collate_fn(batch):
    images = torch.stack([torch.from_numpy(item['x']) for item in batch])
    points = [torch.from_numpy(item['nodes']) for item in batch]
    edges = [torch.from_numpy(item['edges']) for item in batch]
    #points = [item_ for item in batch for item_ in item[2]]
    #edges = [item_ for item in batch for item_ in item[3]]
    return [images, points, edges]


class TestDataloader(absltest.TestCase):
    def test_afm_dataloader(self):
        tfs = transforms.Compose([
            preprocessing.NormalizeImage(),
        ])
        
        ds = data_loading.AFMData(
            '/scratch/phys/project/sin/hackathon/data/afm.h5',
            transform=tfs
        )
        dl = DataLoader(ds, batch_size=8, collate_fn=afm_collate_fn, num_workers=8)
        dl = iter(dl)
        
        for i in range(100):
            t0 = time.time()
            batch = next(dl)
            print(f"time to load batch: {(time.time()-t0)*1000:.2f} ms")
            print(f"images: {batch[0].shape}, points: {len(batch[1])} - {batch[1][0].shape}, edges: {len(batch[2])} - {batch[2][0].shape}")
            time.sleep(0.1)
                

if __name__ == '__main__':
    absltest.main()
