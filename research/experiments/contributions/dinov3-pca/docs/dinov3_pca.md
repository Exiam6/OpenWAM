# Optional frozen PCA bottleneck for DINOv3

This encoder adds a simple comparison for the native DINO + S-VAE path: fit a fixed linear bottleneck on training features, then train the same world/action model. The optional `dinov3_pca` registry entry does not change existing encoder defaults. It has no pixel decoder and is not a complete representation autoencoder (RAE).

Fit statistics only on training trajectories, using exactly the image resize/composition, DINO layer and causal temporal pooling used by the policy. The required `pca_path` PyTorch file contains:

- `mean`: float tensor `[D]`, training feature channel means.
- `std`: positive float tensor `[D]`, training feature channel standard deviations.
- `basis`: float tensor `[D,K]`, leading covariance eigenvectors after channel standardization, in descending eigenvalue order.
- `eigenvalues`: float tensor with at least `K` elements, matching that order.

The transform is `((z-mean)/std) @ basis / sqrt(eigenvalues[:K])`, followed by the native per-token non-affine LayerNorm. The example configuration uses the native DINO dimensions with a48-channel bottleneck. Statistics remainFP32 when the parent model switches toBF16; output adopts input dtype. Projection has no learned parameters. Do not fit or recalibrate these statistics on validation/test scenes.

Example encoder configuration:

```yaml
name: dinov3_pca
model_path: /path/to/DINOv3
pca_path: /path/to/train_only_pca.pt
```

The encoder's advertised latent channel count drives the native DiT input/output heads, so those heads must be trained for the selected representation. Substituting PCA into a policy trained with S-VAE at inference time does not establish a valid comparison.

Native checkpoint saving stores DINO weights and PCA buffers inside the main state dictionary. `pca_config.json` contains dimensions/format only; `dinov3/config.json` supplies the native backbone skeleton. Deployment uses strict checkpoint loading and does not require the original PCA fitting file. A simultaneous S-VAE sidecar is rejected.

Tests use a tiny CPU backbone: projection/whitening arithmetic, precision preservation, causal first-frame behavior, batch invariance, shape/normalization, registry construction, strict reload, and malformed-statistic/ambiguous-sidecar rejection. No downloaded weights are needed for those tests. The current experiment also passed native full checkpoint reload and identical train/deploy image composition for all three routes.

Validation status: independent offline representation diagnostics motivate this baseline, but native closed-loop policy superiority has **not yet been established**. The ongoing matched experiment uses105 trajectories, three training seeds, fixed6,000 optimizer updates, and a frozen fresh-scene robustness protocol. Any result from the reduced native295M control must be distinguished from a full-size OpenWAM result. All conditions and failed/negative experiments must be retained.
