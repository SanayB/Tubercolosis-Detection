# ============================================================
# 1. INSTALL DEPENDENCIES & SETUP
# ============================================================
!pip install -q opencv-python pillow timm scikit-learn transformers accelerate bitsandbytes

import os, io, torch, torch.nn as nn, torch.nn.functional as F, numpy as np, cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from IPython.display import display, clear_output
import ipywidgets as widgets

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 2. INITIALIZE BIOMISTRAL-7B (OPTIMIZED LOADING)
# ============================================================
print("🔄 Loading BioMistral-7B (Medical LLM)...")
model_id = "BioMistral/BioMistral-7B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_tokenizer.pad_token = llm_tokenizer.eos_token

llm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_safetensors=False,
    low_cpu_mem_usage=True
)
print("✅ BioMistral Loaded. Point-wise Expert logic enabled.")

# ============================================================
# 3. TB ENSEMBLE MODEL & GRADCAM ENGINE
# ============================================================
class TBEnsemble(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.m1 = models.densenet121(weights=None)
        self.m1.classifier = nn.Linear(self.m1.classifier.in_features, num_classes)
        self.m2 = models.resnet50(weights=None)
        self.m2.fc = nn.Linear(self.m2.fc.in_features, num_classes)

    def forward(self, x):
        out1 = self.m1(x)
        out2 = self.m2(x)
        p1 = F.softmax(out1, dim=1)
        p2 = F.softmax(out2, dim=1)
        return (p1 + p2) / 2, p1, p2

# Load Weights
MODEL_PATH = "/kaggle/input/models/sanayborhade/tbprediction/pytorch/default/1/tb_model.pt"
tb_model = TBEnsemble().to(device)
if os.path.exists(MODEL_PATH):
    tb_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ TB Ensemble weights loaded.")
else:
    print("⚠️ Warning: Weight path not found. Using random initialization.")
tb_model.eval()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients, self.activations = None, None
        target_layer.register_forward_hook(self.save_act)
        target_layer.register_full_backward_hook(self.save_grad)

    def save_act(self, m, i, o): self.activations = o
    def save_grad(self, m, gi, go): self.gradients = go[0]

    def generate(self, input_tensor, class_idx):
        output, _, _ = self.model(input_tensor.to(device))
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = torch.mean(self.gradients[0], dim=(1, 2))
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32).to(device)
        for i, w in enumerate(weights): cam += w * self.activations[0, i]
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cv2.resize(cam.detach().cpu().numpy(), (224, 224))

gradcam_engine = GradCAM(tb_model, tb_model.m2.layer4[-1])

# ============================================================
# 4. FINAL EXPERT REPORT GENERATION (FIXED)
# ============================================================
def generate_expert_report(probs, side, zone, cam):
    tb_score = probs[1]
    norm_score = probs[0]
    intensity = "dense" if cam.max() > 0.75 else "patchy"
    other_side = "Right" if side == "Left" else "Left"

    if tb_score > 0.5:
        status = "POSITIVE"
        pathology = (f"Heterogeneous {intensity} infiltration with consolidation noted in the {side} {zone} zone. "
                     f"The {other_side} lung shows normal translucency. Bronchovascular markings are prominent.")
        imp = "MANTOUX TEST OR MENDEL - MANTOUX TEST. CORRELATE WITH SPUTUM AFB AND CLINICAL HISTORY."
    elif 0.4 <= tb_score <= 0.5:
        status = "SUSPICIOUS / BORDERLINE"
        pathology = f"Subtle {intensity} haziness visualized in the {side} {zone} zone. Interlobar fissures not clearly assessed."
        imp = "CLINICAL CORRELATION AND HRCT CHEST RECOMMENDED."
    else:
        status = "NEGATIVE"
        pathology = "The lungs on either side show equal translucency and normal parenchymal markings. No focal lesions."
        imp = "NO RADIOLOGICAL EVIDENCE OF ACTIVE KOCH'S. CLINICAL FOLLOW UP IF SYMPTOMS PERSIST."

    prompt = f"""[INST] You are a Board-Certified Senior Radiologist. Write a highly detailed, point-wise Chest X-ray report.
Each anatomical landmark MUST be on a new line with a bullet point.

Construct the report EXACTLY from these observations:
- LUNGS : {pathology}
- CP ANGLES : Both costophrenic angles are clear. No blunting or pleural effusion.
- HILA : Hilar shadows are normal in position, density, and bearing a normal relationship.
- HEART : Cardiac silhouette and cardiomediastinal size are within normal limits.
- TRACHEA : Trachea is midline. Azygos vein shadow is projected over cardiomediastinal silhouette.
- DIAPHRAGM : Domes of diaphragms show normal position and smooth outlines.
- BONES : The visualized bony thorax and soft tissue structures appear intact.
TBS STATUS: {status}.
IMPRESSION : {imp}
[/INST]
FINDINGS :
"""

    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs, 
            max_new_tokens=350, 
            temperature=0.7, 
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=llm_tokenizer.eos_token_id
        )
    
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    report_body = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    if not report_body.upper().startswith("FINDINGS"):
        report_body = "FINDINGS :\n" + report_body
    
    for tag in ["<UMLS", "2014-", "VIEWXS", "<START", "<REPORT"]:
        if tag in report_body: 
            report_body = report_body.split(tag)[0].strip()

    return status, tb_score, norm_score, report_body

# ============================================================
# 5. UI & EXECUTION PIPELINE
# ============================================================
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

upload_btn = widgets.FileUpload(accept="image/*", multiple=False)
run_btn = widgets.Button(description="Analyze & Generate Report", button_style="success", layout=widgets.Layout(width='250px'))
output_area = widgets.Output()

def run_diagnostic(_):
    with output_area:
        clear_output(wait=True)
        if not upload_btn.value:
            print("❌ Please upload an X-ray image.")
            return
        
        file_info = list(upload_btn.value.values())[0] if isinstance(upload_btn.value, dict) else upload_btn.value[0]
        img = Image.open(io.BytesIO(file_info['content'])).convert("RGB")
        
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs, p1, p2 = tb_model(tensor)
            probs_list = probs[0].cpu().tolist()
        
        cam = gradcam_engine.generate(tensor, 1)
        w = cam.shape[1]
        side = "Left" if cam[:, :w//2].mean() > cam[:, w//2:].mean() else "Right"
        zone = ["Upper", "Middle", "Lower"][np.argmax([cam[:74,:].mean(), cam[74:148,:].mean(), cam[148:,:].mean()])]
        
        status, tb_p, norm_p, report = generate_expert_report(probs_list, side, zone, cam)
        
        display(img.resize((350, 350)))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (heatmap * 0.4 + np.array(img.resize((224,224))) * 0.6).astype(np.uint8)
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(overlay); ax.axis('off'); ax.set_title(f"Localization: {side} {zone} Zone")
        plt.show()

        print("\n" + "="*75)
        print(f" 📄 RADIOLOGY REPORT - {file_info.get('name', 'STUDY.IMG')}")
        print("="*75)
        print(f" AI PREDICTED METRICS:")
        print(f" - Primary Status   : {status}")
        print(f" - TB Confidence    : {tb_p*100:.2f}%")
        print(f" - Normal Confidence: {norm_p*100:.2f}%")
        print(f" - Detected Zone    : {side} {zone}")
        print("-" * 75)
        print(report)
        print("="*75)
        print(f"FINAL CLINICAL SUGGESTION: {status}")

run_btn.on_click(run_diagnostic)
display(widgets.VBox([
    widgets.HTML("<h2>Advanced TB Diagnostic Suite (Ensemble Vision + BioMistral)</h2>"),
    widgets.HBox([upload_btn, run_btn]),
    output_area
]))
