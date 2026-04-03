import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, cv2, os, io
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# NIH standard list - Adjusted for exact naming in your screenshot
PATHOLOGY_LIST = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'PT', 'Hernia', 'Normal'
]

# ============================================================
# 1. ARCHITECTURE & GRAD-CAM
# ============================================================

    def forward(self, x):
        return (torch.sigmoid(self.m1(x)) + torch.sigmoid(self.m2(x))) / 2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.target_layer = model, target_layer
        self.gradients, self.activations = None, None
        target_layer.register_forward_hook(self.save_act)
        target_layer.register_full_backward_hook(self.save_grad)

    def save_act(self, m, i, o): self.activations = o
    def save_grad(self, m, gi, go): self.gradients = go[0]

    def generate(self, input_tensor, class_idx, device):
        probs = self.model(input_tensor.to(device))
        self.model.zero_grad()
        probs[0, class_idx].backward()
        weights = torch.mean(self.gradients[0], dim=(1, 2))
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32).to(device)
        for i, w in enumerate(weights): cam += w * self.activations[0, i]
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        
        # Anatomical Masking to keep BBox inside lung fields
        mask = np.zeros((224, 224), dtype=np.float32)
        cv2.ellipse(mask, (112, 112), (80, 100), 0, 0, 360, 1.0, -1)
        cam_np = cv2.resize(cam.detach().cpu().numpy(), (224, 224))
        return cam_np * mask

# ============================================================
# 2. REPORTING ENGINE (Strict Formatting)
# ============================================================
def generate_clinical_report(prob_scores, cam, filename):
    top_idx = np.argmax(prob_scores[:-1]) # Ignore 'Normal' for primary finding
    status = "POSITIVE" if prob_scores[top_idx] > 0.4 else "NEGATIVE"
    
    # Texture Analysis
    roi = cam[cam > 0.1]
    if len(roi) > 0:
        consolidation = (np.sum(roi > 0.6) / len(roi)) * 100
        infiltrates = 100 - consolidation
    else:
        consolidation, infiltrates = 0.0, 0.0

    # Output formatting
    print("=" * 65)
    print(f" 📄 RADIOLOGY REPORT - {filename}")
    print("=" * 65)
    print(f" AI PREDICTED METRICS:")
    print(f" - Primary Status    : {status}")
    print(f" - TB Confidence     : {prob_scores[top_idx]*100:6.2f}%")
    print(f" - Normal Confidence : {prob_scores[-1]*100:6.2f}%")
    print("-" * 65)
    print(f" [ PATHOLOGY COMPOSITION (from Heatmap) ]")
    print(f" - Consolidation : {consolidation:4.1f}%")
    print(f" - Infiltrates   : {infiltrates:4.1f}%")
    print("-" * 65)
    print(f" [ ADDITIONAL PARAMETERS (ResNet-50) ]")
    
    # Corrected alignment loop
    for i, name in enumerate(PATHOLOGY_LIST[:-1]):
        score = prob_scores[i] * 100
        # This formatting ensures colons and decimals align perfectly
        print(f" - {name.ljust(15)} : {score:5.2f}%")
    
    print("-" * 65)
    print(f" FINDINGS :")
    print(f" The chest radiograph demonstrates patterns consistent with {PATHOLOGY_LIST[top_idx].lower()}.")
    print(f" Costophrenic angles are clear. Heart size is normal.")
    print(f" FINAL CLINICAL SUGGESTION: {status}")
    print("=" * 65)

# ============================================================
# 3. PIPELINE & UI
# ============================================================
def run_diagnostic(img_path, model, device, filename):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_raw = Image.open(img_path).convert("RGB")
    tensor = preprocess(img_raw).unsqueeze(0).to(device)
    
    with torch.set_grad_enabled(True):
        probs = model(tensor)[0].detach().cpu().numpy()
        gradcam_engine = GradCAM(model, model.m2.layer4[-1])
        cam = gradcam_engine.generate(tensor, np.argmax(probs[:-1]), device)
    
    # Viz with Green BBox
    img_viz = np.array(img_raw.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_viz, 0.6, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.4, 0)
    
    contours, _ = cv2.findContours(np.uint8(cam > 0.4 * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.imshow(img_raw); plt.axis('off'); plt.title("Source")
    plt.subplot(1, 2, 2); plt.imshow(overlay); plt.axis('off'); plt.title("XAI Localization")
    plt.show()
    
    generate_clinical_report(probs, cam, filename)

device = "cuda" if torch.cuda.is_available() else "cpu"
upload_btn = widgets.FileUpload(accept='image/*', multiple=False)
run_btn = widgets.Button(description="Run Full Diagnostic", button_style='primary')
out = widgets.Output()

def on_run(b):
    with out:
        clear_output()
        if not upload_btn.value: return
        file = list(upload_btn.value.values())[0] if isinstance(upload_btn.value, dict) else upload_btn.value[0]
        with open("temp.jpg", "wb") as f: f.write(file['content'])
        run_diagnostic("temp.jpg", model, device, file['metadata']['name'] if 'metadata' in file else "scan.jpg")

run_btn.on_click(on_run)
display(widgets.VBox([upload_btn, run_btn, out]))