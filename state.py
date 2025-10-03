from Transformers.TFM_Prediction import StockTransformer, PositionalEncoding , TransformerEncoderBlock
import torch
import os

BASE_DIR = "/home/piyush/StockTransformer/Transformers"
full_model_path = os.path.join(BASE_DIR, "stock_transformer_Amazon.pth")
state_dict_path = os.path.join(BASE_DIR, "stock_transformer_Amazon_state.pth")

# Allow globals
torch.serialization.add_safe_globals([StockTransformer])

# Load full model
model = torch.load(full_model_path, map_location="cpu", weights_only=False)
model.eval()

# Save as state_dict
torch.save(model.state_dict(), state_dict_path)
print("Saved state_dict:", state_dict_path)
