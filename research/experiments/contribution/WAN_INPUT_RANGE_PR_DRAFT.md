# Correct the documented pixel range for WanVideoVAE.batch_encode

The docstring currently advertises input values in [0,1], while the native Wan
preprocessor and WanVideoVAEEncoder.preprocess_video map pixels to [-1,1].
Callers following the docstring could feed inputs with the wrong range and
produce misleading representation comparisons. Change only the documented range
to [-1,1]; no runtime behavior or model defaults change.

Evidence: openwam/model/video_backbone/encoder/wan22_vae.py lines75–78 and117–124
perform that normalization and forward the tensor directly to batch_encode.
The implementation in openwam/model/video_backbone/wan/models/vae.py forwards
videos to model.encode without another range transformation.

Validation: git apply --check passes against the current frozen source. The
separately completed official-weight interface check verifies finite encoding,
correct shapes, exact causal first-frame/cache-reset behavior and finite pixel
decoding using [-1,1] inputs. It is not a reconstruction-quality benchmark.
No new test or GPU run is needed for this documentation-only correction.

Patch: wan-vae-input-range-doc.patch. Prepared locally; not applied to the frozen
experiment source and not submitted upstream. This correction is separate from
any claim of RAE, representation or policy improvement.
