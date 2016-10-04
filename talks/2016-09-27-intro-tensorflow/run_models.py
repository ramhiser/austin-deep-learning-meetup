from models.base_convnet import basic_model
from models.inception_model import inception_model
from models.semantic_segmentation import semantic_segmentation

# This program will run all the models with their tuning parameters.

print "=============================Running Basic Convnet Model===================================="
basic_model(lr=0.003, num_epochs=10)
basic_model(lr=0.001, num_epochs=10)

print "===============================Running Inception Model======================================"
inception_model(lr=0.0003, num_epochs=10)
inception_model(lr=0.0001, num_epochs=10)

print "=========================Running Semantic Segmentation Model================================"
semantic_segmentation(lr=0.002, num_epochs=20)
