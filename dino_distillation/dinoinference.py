from ultralytics import YOLO
# model = YOLO("/home/pavan_kadambala_invisible_email/Dev/cows/runs/detect/train3/weights/best.pt")
# model = YOLO("/home/pavan_kadambala_invisible_email/Dev/cows/runs/detect/train5/weights/last.pt")
model = YOLO("/home/pavan_kadambala_invisible_email/Dev/cows/runs/detect/train4/weights/last.pt")

results = model.predict(source="/home/pavan_kadambala_invisible_email/Dev/cows/block31.mp4",
save=True,
device="cuda",
conf= 0.5,
iou=0.5,
vid_stride=1)

print("Saved to:", results[0].save_dir)