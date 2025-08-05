# **Sub-Memory Efficient Merging: FluxKreaSchnell**

> The Flux.1-Krea-Merged-Schnell repository features merged parameters that combine two leading image generation models: black-forest-labs/FLUX.1-schnell, renowned for its ultra-fast, precision prompt following and efficient 1–4 step professional image production, and black-forest-labs/FLUX.1-Krea-dev, celebrated for guidance-distilled training, aesthetic curation, and photorealistic output. The resulting unified model empowers users to generate visually striking, high-quality images at remarkable speed, balancing sharp style consistency and realism with creative flexibility, and is fully compatible with the Diffusers library’s FluxPipeline for streamlined, accessible text-to-image workflows in both research and creative settings.

| FLUX.1-schnell [4 steps] | FLUX.1-Krea-dev [28 steps] | Flux.1-Krea-Merged-Schnell [28 steps] |
|---------------------------|---------------------------|---------------------------------------|
| ![FS1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/ybvHcHYT4qXLAMEzPqrmF.webp) | ![Kear1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/4wqEgqfwnlqPxBTXOzQ9I.webp) | ![Schenell1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/6qgkq40Wue2MUdwuZF3tK.png) |
| ![FS2](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/1vCPtqzAbd5sgyyfDnhT2.webp) | ![Krea3](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/EGsgVlR5rym8l82vkN-ge.webp) | ![Schnell3](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/8F0dHIrth3dz16CRcty8K.png) |
| ![FS3](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/bNwmPJ_WS5eX4eRty5XwF.webp) | ![Krea2](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/HIEHRB_kUAzG2VMNB_gy5.webp) | ![Schnell2](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Axny8PmCxw3i5_aCwZgce.png) |

> prompt : a tiny astronaut hatching from an egg on mars

---

| FLUX.1-schnell [4 steps] | FLUX.1-Krea-dev [28 steps] | Flux.1-Krea-Merged-Schnell [28 steps] |
|------------------------------|-------------------------------|------------------------------------------|
| ![S](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Of4-_nMZdDVPzFEOoR8JA.webp) | ![Kera1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/LikSye79dIrtxi1O6NY4Q.webp) | ![FSS1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/9voGCo46upXU53aNHTkDV.png) |
| ![SSS](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Q2m-y8GnDkkb2CY0Ro0fK.webp) | ![Kera2](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/b94_deGUHm3edDQ3GIAAH.webp) | ![FSS2](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/LGPGZfanm_99BqIao8Bln.png) |
| ![SSS1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/H_BrSwFD-BTdXDgotNb0k.webp) | ![Kera3](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/u3hY2JhnoySufmZsSi546.webp) | ![FSS3](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/6_K_Vu_GboIOOBf5u9i14.png) |

> prompt : a tiny astronaut hatching from an egg on the moon


## Overview

This project implements a sophisticated model merging technique that:
- Averages non-guidance weights between FLUX.1-schnell and FLUX.1-Krea-dev models
- Preserves guidance weights exclusively from the schnell model
- Uses memory-efficient loading with empty weight initialization
- Processes models shard-by-shard to minimize memory footprint
- Outputs the merged model in bfloat16 precision for optimal performance

## Features

- **Memory Efficient**: Uses `init_empty_weights()` and shard-by-shard processing
- **Intelligent Merging**: Differentiates between guidance and non-guidance parameters
- **Error Handling**: Comprehensive validation for missing or unexpected keys
- **Optimized Output**: Saves in bfloat16 format for reduced memory usage
- **Clean Processing**: Proper cleanup of temporary state dictionaries

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Sufficient storage space (~20GB for model downloads)

### Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/diffusers.git
pip install git+https://github.com/huggingface/peft.git
pip install git+https://github.com/huggingface/accelerate.git
pip install safetensors huggingface_hub hf_xet
```

## Usage

### Basic Usage

Run the merging script:

```python
python merge_flux_models.py
```

The script will:
1. Download both FLUX.1-schnell and FLUX.1-Krea-dev models
2. Process them shard-by-shard
3. Merge non-guidance weights through averaging
4. Preserve guidance weights from schnell model
5. Save the merged model to `merged/transformer/`

### Advanced Configuration

You can modify the script to customize:

- **Output directory**: Change the save path in `save_pretrained()`
- **Precision**: Modify `.to(torch.bfloat16)` to use different precision
- **Model variants**: Replace model IDs for different FLUX variants

## Technical Details

### Merging Strategy

The merging process follows this logic:

1. **Guidance Parameters**: Exclusively from FLUX.1-schnell
   - These parameters control the model's guidance capabilities
   - Critical for maintaining schnell's fast inference characteristics

2. **Non-Guidance Parameters**: Averaged between both models
   - Standard transformer weights (attention, MLP, normalization layers)
   - Combines the strengths of both model variants

### Memory Optimization

- **Empty Weight Initialization**: Models are initialized without loading weights
- **Shard Processing**: Processes one shard pair at a time
- **Immediate Cleanup**: Removes processed weights from memory
- **Efficient Storage**: Uses bfloat16 for final model

### Error Handling

The script includes robust error checking:
- Validates key presence across model shards
- Identifies unexpected residual keys
- Ensures complete processing of all parameters

## Output

The merged model will be saved in the `merged/transformer/` directory with:
- Model configuration files
- Merged weight shards in safetensors format
- Compatible with standard diffusers pipeline loading

## Loading the Merged Model

```python
from diffusers import FluxTransformer2DModel

# Load the merged model
merged_model = FluxTransformer2DModel.from_pretrained(
    "merged/transformer",
    torch_dtype=torch.bfloat16
)
```

## Performance Considerations

- **Memory Usage**: Peak memory usage is significantly reduced compared to loading full models
- **Processing Time**: Shard-by-shard processing takes longer but uses less memory
- **Storage**: Final merged model maintains similar size to original models

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch processing or use smaller precision
2. **Missing Keys**: Ensure both models have compatible architectures
3. **Download Failures**: Check internet connection and HuggingFace Hub access

### Error Messages

- `Key {k} missing in krea shard`: Architectural mismatch between models
- `Residue in shard`: Incomplete processing, check for unexpected parameters
- `Unexpected non-guidance key`: Model structure differs from expected format

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project follows the same licensing terms as the underlying FLUX models. Please refer to the original model repositories for specific license information.

## Acknowledgments

- **Black Forest Labs** for the FLUX.1 model series
- **Hugging Face** for the diffusers and transformers libraries
- **Krea AI** for the FLUX.1-Krea-dev variant

## Repository

**GitHub**: [https://github.com/PRITHIVSAKTHIUR/Sub-Memory-Efficient-Merging-FluxKreaSchnell.git](https://github.com/PRITHIVSAKTHIUR/Sub-Memory-Efficient-Merging-FluxKreaSchnell.git)
