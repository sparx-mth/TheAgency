from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import torch

# Load model & processor
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

# Load image
image = Image.open("image.jpg")

# Define text queries (object descriptions)
texts = [["a chair", "a table", "a dog", "a monitor"]]

# Prepare inputs
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# Inference
with torch.no_grad():
    outputs = model(**inputs)

# Postprocess
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.3)[0]

# Print results
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f">>> {texts[0][label]}: {round(score.item(), 3)} at {box.tolist()}")
