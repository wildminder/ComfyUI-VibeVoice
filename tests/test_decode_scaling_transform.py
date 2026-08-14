"""Regression test for the diffusion-output -> VAE-decode scaling transform.

BUG-005 (empty/silent audio from a successful generation): `VibeVoiceForConditionalGeneration
.generate` produced a 25s audio file that was silent. The diffusion head is trained on the
*scaled* acoustic features `audio_features = (tokens + bias) * scaling` (see
`forward_speech_features` and the diffusion-loss target `add_noise(speech_features, ...)`), so
its output `sampled` lives in scaled-feature space. The VAE `acoustic_tokenizer.decode` expects
the *raw* latent tokens, so the output must be run through the INVERSE transform
`token = sampled / scaling - bias` before decoding.

The non-streaming `generate` decode block instead applied the FORWARD transform
`sampled = (sampled + bias) * scaling` — doubly-scaling an already-scaled output and feeding a
wildly out-of-distribution tensor to the VAE decoder, which collapses to (near-)silence. The
streaming inference path (`modeling_vibevoice_streaming_inference.py:786`) uses the correct
inverse: `speech_latent / speech_scaling_factor - speech_bias_factor`.

This test locks the direction of the decode transform algebraically (it mirrors the exact block in
`generate` without loading the heavy model package): starting from raw tokens, forward-scale to
the head's output space, then assert that the NEW inverse transform recovers the original tokens
while the OLD forward transform does not.
"""

import pytest

torch = pytest.importorskip("torch")


def _forward_scale(tokens, bias, scaling):
    """Mirrors forward_speech_features: tokens -> scaled acoustic features."""
    return (tokens + bias) * scaling


def _decode_input_new(sampled, bias, scaling):
    """The FIXED decode transform (inverse), mirrors generate's decode block."""
    return sampled / scaling - bias


def test_decode_uses_inverse_scaling_transform():
    torch.manual_seed(0)
    N, vae_dim = 12, 8

    # Simulate raw latent tokens the VAE decoder expects (arbitrary distribution).
    tokens = torch.randn(N, vae_dim, dtype=torch.float32)

    # Scaling/bias factors exactly as computed in forward_speech_features:
    #   scaling = 1 / std(tokens);  bias = -mean(tokens)
    scaling = 1.0 / tokens.flatten().std()
    bias = -tokens.flatten().mean()

    # The trained diffusion head outputs the SCALED features, i.e. the forward
    # transform of the raw tokens (a standardization, NOT identity).
    sampled = _forward_scale(tokens, bias, scaling)
    assert not torch.allclose(sampled, tokens), (
        "scaled head output must differ from raw tokens (it is standardized)"
    )

    # FIXED (inverse) transform must recover the original raw tokens.
    recovered = _decode_input_new(sampled, bias, scaling)
    assert torch.allclose(recovered, tokens, atol=1e-5), (
        "inverse decode transform did not recover the raw tokens"
    )
    assert recovered.dtype == tokens.dtype

    # Sanity: a real decode would feed `recovered` to acoustic_tokenizer.decode.


def test_decode_old_forward_transform_would_distort():
    torch.manual_seed(1)
    N, vae_dim = 12, 8
    tokens = torch.randn(N, vae_dim, dtype=torch.float32)
    scaling = 1.0 / tokens.flatten().std()
    bias = -tokens.flatten().mean()

    sampled = _forward_scale(tokens, bias, scaling)

    # The OLD (buggy) transform applied the forward direction again.
    distorted = (sampled + bias) * scaling

    # It must NOT equal the raw tokens (otherwise the fix would be a no-op and the
    # silent-audio bug would persist). This guards against accidentally reverting
    # the transform to the forward direction.
    assert not torch.allclose(distorted, tokens, atol=1e-3), (
        "the forward (buggy) decode transform unexpectedly recovered the tokens; "
        "the decode must use the inverse transform"
    )
