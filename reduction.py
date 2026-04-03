# Reduce model size for edge deployment
import torch.quantization

def optimize_model_for_deployment(model):
    model.eval()
    # Quantize Linear layers to INT8
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), "tb_ensemble_lite.pt")
    print("Optimization Complete: Model ready for clinical deployment.")