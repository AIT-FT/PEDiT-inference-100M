import argparse
import os
import torch
import torch.nn as nn
from model import EdgeFlowPhaseV2


class TensorRTWrapper(nn.Module):
    def __init__(self, model: EdgeFlowPhaseV2):
        super().__init__()
        self.model = model

    def forward(
        self,
        z: torch.Tensor,
        cond_pooled: torch.Tensor,
        t_phase: torch.Tensor,
        mask: torch.Tensor,
        text_k: torch.Tensor,
        text_v: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            z=z,
            cond_pooled=cond_pooled,
            cond_seq=None,
            t_phase=t_phase,
            mask=mask,
            text_kvs=(text_k, text_v),
        )


def export_tensorrt():
    parser = argparse.ArgumentParser(description="Export Model to ONNX / TensorRT format.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file (.pt)")
    parser.add_argument("--output", type=str, default="edgeflow_v2.onnx", help="Output path for the ONNX file")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA weights from checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = dict(
        latent_channels=4,
        patch_size=2,
        dim=768,
        depth=8,
        heads=12,
        kv_heads=2,
        context_dim=768,
        num_phases=4,
        ffn_hidden=1024,
        ada_rank=64,
        cross_every=2,
        expert_hidden_list=[768, 768, 768, 768],
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ckpt_config = checkpoint.get("config", {})
        if ckpt_config:
            config.update(ckpt_config)

        model = EdgeFlowPhaseV2(**config).to(device)

        state_dict = None
        if args.use_ema and "ema_state_dict" in checkpoint:
            state_dict = checkpoint["ema_state_dict"]
            print("Loaded EMA weights from checkpoint.")
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("Loaded base model weights from checkpoint.")

        if state_dict is not None:
            clean_state = {k.replace("module.", ""): v for k, v in state_dict.items() if not k.endswith("n_averaged")}
            model.load_state_dict(clean_state, strict=False)
        del checkpoint
    else:
        print("No checkpoint provided or file not found. Exporting with default configuration...")
        model = EdgeFlowPhaseV2(**config).to(device)

    model.eval()
    wrapper = TensorRTWrapper(model).to(device)
    wrapper.eval()

    # Create dummy inputs
    batch_size = 2
    latent_h, latent_w = 32, 32
    
    z = torch.randn(batch_size, 4, latent_h, latent_w, device=device)
    cond_pooled = torch.randn(batch_size, config["dim"], device=device)
    cond_seq = torch.randn(batch_size, 128, config["dim"], device=device)
    t_phase = torch.tensor([0.5] * batch_size, device=device)
    mask = torch.ones(batch_size, 128, dtype=torch.long, device=device)
    
    # We use a tuple for text_kvs (k and v)
    text_k, text_v = model.encode_text(cond_seq)

    print(f"Exporting model to {args.output}...")

    try:
        import onnx
    except ImportError:
        print("[WARNING] The 'onnx' library is not installed in your Python environment.")
        print("To export the model to ONNX / TensorRT, please run:")
        print("    pip install onnx onnxruntime onnxscript")
        return

    # Export to ONNX with dynamic axes to support different batch sizes and image sizes
    torch.onnx.export(
        wrapper,
        (z, cond_pooled, t_phase, mask, text_k, text_v),
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['z', 'cond_pooled', 't_phase', 'mask', 'text_k', 'text_v'],
        output_names=['output'],
        dynamic_axes={
            'z': {0: 'batch_size', 2: 'height', 3: 'width'},
            'cond_pooled': {0: 'batch_size'},
            't_phase': {0: 'batch_size'},
            'mask': {0: 'batch_size', 1: 'seq_len'},
            'text_k': {0: 'batch_size', 2: 'seq_len'},
            'text_v': {0: 'batch_size', 2: 'seq_len'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    print("Model successfully exported to ONNX format.")
    print("To convert to TensorRT, use trtexec with generated ONNX file and specified profile shapes.")


if __name__ == "__main__":
    with torch.inference_mode():
        export_tensorrt()
