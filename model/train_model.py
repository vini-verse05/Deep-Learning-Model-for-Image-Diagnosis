import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cnn_model import build_model
from utils.preprocessing import get_data_generators

model = build_model()

train_gen, test_gen = get_data_generators()

model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=30
)

model.save("model/saved_model.h5")

print("Model Training Complete")