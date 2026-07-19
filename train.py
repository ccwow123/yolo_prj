import os

from ultralytics import YOLO

os.environ["ULTRALYTICS_API_KEY"] = "ul_b025fc5cf79530ba37866c20e89b545fc3631e86"

model = YOLO("ul://ultralytics/yolo26/yolo26n")
model.train(
    data="ul://lemon/datasets/cbook",
    epochs=100,
    project="username/my-project",
    name="lemon/v2i",
)
# Metrics stream to Platform automatically