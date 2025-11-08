import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from nudenet_distill import create_student_model

# class DifferentiableNudeNet(nn.Module):
#     def __init__(self, onnx_path):
#         super().__init__()
#         self.target_size = (256, 256)
        
#         # Use onnx2torch to convert the model directly
#         # This will return a PyTorch nn.Module
#         self.model = convert(onnx_path)
        
#         self.model.eval()

#     def forward(self, x):
#         # The input 'x' is a torch.Tensor from the VAE, range [-1, 1].
#         # Convert to [0, 1] range first.
#         x = (x + 1.0) / 2.0
        
#         # 1. Differentiable Resize
#         x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False, antialias=True)
        
#         # 2. Differentiable Normalization (check if needed)
#         # x = F.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         x = F.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Example normalization
        
#         # 3. Get model logits
#         logits = self.model(x)
        
#         # 4. Convert to a probability score [0, 1]
#         scores = torch.sigmoid(logits).squeeze(-1)
        
#         return scores


class DifferentiableNudeNet(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        # Use the same architecture as the student
        self.model = create_student_model()
        # Load the weights you just trained
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        self.target_size = (224, 224)

    def forward(self, x):
        # The input 'x' is a torch.Tensor from the VAE, range [-1, 1].
        # We need to preprocess it the same way we did during student training.
        x = (x + 1.0) / 2.0
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False, antialias=True)
        x = transforms.functional.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return self.model(x).squeeze(-1)

# Now in your main script, you can use it like this:
# eval_func = Eval(args) where DifferentiableNudeNet uses 'student_nudenet.pth'
