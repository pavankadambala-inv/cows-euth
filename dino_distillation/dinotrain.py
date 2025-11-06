import torch
from ultralytics import YOLO
import lightly_train
import os
import torch
from PIL import Image
import numpy as np
import yaml

# Stage 1: Pretraining on small weapons dataset for Distillation
lightly_train.train(
    out="out/my_experiment1",
    data="/home/pavan_kadambala_invisible_email/Dev/cows/models/detector/datasets/weapon_dataset_small_weapons/train/images/",
    model=YOLO("yolo12n.pt"),
    method="distillationv1",
    overwrite=True,
    epochs=1000,
    batch_size=128,
    method_args={
        "teacher": "dinov3/vitb16",
        "teacher_url": "https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ2l3amh2Z2kxaDR0aGpyaGs0bWpjaXY3IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjE3Nzk3OTB9fX1dfQ__&Signature=eb0MnwGM5L6X0lSCr0PmuHryvPAtYkdgRycGJDYj1nSS4H6vsZAgCNuk4bbgP7tW%7EinmabqRXYAyuBhOrzSlM2lJwreRnXyFMSHLvizCut38p3sYZY5XOvm-8iD5oyUxT4H%7Ee59jmfoTKPuEPPica0gvfjNyGaJQpgQSXPegH1ELTnEWsEec%7EzuX3pYo23c0sGrNv1Bt6JE6MT-m7qmbPOvPUEEPpwW3DpGtxqIcejtwhWo4zMhysJwHsnYFv9oCI6%7ErKeKzA-RiHSsE4DS%7EfNKU15IJ2w02wjCGoD5Xn5M70ER9yYiNQYppVA70VrC9LrHiDxm4cWshJ-adV6hyqQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1547222399732669",
    }
)

# Stage 2: Fine-tuning on v4.7.0 dataset for Weapon Detection
if __name__ == "__main__":
    #load exported model
    model = YOLO("/home/pavan_kadambala_invisible_email/Dev/cows/models/detector/out/my_experiment1/exported_models/exported_last.pt")
    #train the model
    model.train(data='/home/pavan_kadambala_invisible_email/Dev/cows/models/detector/datasets/v4.7.0/data.yaml', epochs=100, batch=96, device='cuda', freeze="backbone")