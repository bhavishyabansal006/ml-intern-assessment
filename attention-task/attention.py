"""
Scaled Dot-Product Attention Implementation
Task 2: Implement the core attention mechanism from "Attention Is All You Need"
Using only NumPy
"""

import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute Scaled Dot-Product Attention.
    
    The attention mechanism is defined as:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    where:
    - Q: Query matrix
    - K: Key matrix
    - V: Value matrix
    - d_k: Dimension of the key vectors (used for scaling)
    
    Args:
        Q: Query matrix of shape (batch_size, seq_len_q, d_k) or (seq_len_q, d_k)
        K: Key matrix of shape (batch_size, seq_len_k, d_k) or (seq_len_k, d_k)
        V: Value matrix of shape (batch_size, seq_len_v, d_v) or (seq_len_v, d_v)
           Note: seq_len_k must equal seq_len_v
        mask: Optional mask of shape (seq_len_q, seq_len_k) or (batch_size, seq_len_q, seq_len_k)
              Used to prevent attention to certain positions (e.g., padding or future tokens)
              Values should be 0 for positions to attend to, -inf for positions to mask
    
    Returns:
        output: Attention output of shape (batch_size, seq_len_q, d_v) or (seq_len_q, d_v)
        attention_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)
    """
    
    # Step 1: Extract d_k (dimension of key vectors) for scaling
    # d_k is the last dimension of K
    d_k = K.shape[-1]
    
    # Step 2: Compute attention scores
    # Formula: scores = Q @ K^T
    # This computes the dot product between each query and all keys
    # Using matmul to handle both 2D and 3D tensors (with batch dimension)
    scores = np.matmul(Q, np.swapaxes(K, -1, -2))
    # Note: swapaxes(-1, -2) swaps the last two dimensions
    # For 2D: (seq_q, d_k) @ (d_k, seq_k) = (seq_q, seq_k)
    # For 3D: (batch, seq_q, d_k) @ (batch, d_k, seq_k) = (batch, seq_q, seq_k)
    
    # Step 3: Scale the scores
    # Formula: scaled_scores = scores / sqrt(d_k)
    # Scaling prevents the dot products from growing too large,
    # which would push the softmax into regions with extremely small gradients
    scaled_scores = scores / np.sqrt(d_k)
    
    # Step 4: Apply mask (if provided)
    # Masking is used to prevent attention to certain positions
    # Common use cases:
    # - Padding mask: ignore padding tokens
    # - Causal mask: prevent attending to future tokens in autoregressive models
    if mask is not None:
        # Add a very large negative number to masked positions
        # After softmax, these will become approximately zero
        scaled_scores = scaled_scores + mask
        # Alternative: scaled_scores = np.where(mask == 0, scaled_scores, -1e9)
    
    # Step 5: Apply softmax to get attention weights
    # Formula: attention_weights = softmax(scaled_scores)
    # Softmax converts scores to probabilities that sum to 1
    # Each row of attention_weights represents the attention distribution
    # for one query over all keys
    attention_weights = softmax(scaled_scores, axis=-1)
    
    # Step 6: Compute weighted sum of values
    # Formula: output = attention_weights @ V
    # This computes a weighted average of the value vectors,
    # where the weights are the attention probabilities
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights


def softmax(x, axis=-1):
    """
    Compute softmax values for each set of scores in x.
    
    Uses the numerically stable version:
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    
    Args:
        x: Input array
        axis: Axis along which to compute softmax
    
    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    # This prevents overflow in exp() without changing the result
    # because softmax(x) = softmax(x + c) for any constant c
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    
    # Compute softmax
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    softmax_x = exp_x / sum_exp_x
    
    return softmax_x


def create_causal_mask(seq_len):
    """
    Create a causal (autoregressive) mask.
    
    This mask prevents positions from attending to future positions.
    Used in decoder-only models like GPT.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Mask of shape (seq_len, seq_len) with 0s on and below diagonal, -inf above
    """
    # Create upper triangular matrix of 1s
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    
    # Replace 1s with -inf (these positions will be masked)
    # and 0s remain (these positions are allowed)
    mask = np.where(mask == 1, -np.inf, 0)
    
    return mask


def create_padding_mask(seq_lens, max_len):
    """
    Create a padding mask for variable-length sequences.
    
    Args:
        seq_lens: Array of actual sequence lengths for each item in batch
        max_len: Maximum sequence length (length after padding)
    
    Returns:
        Mask of shape (batch_size, 1, max_len) with 0 for real tokens, -inf for padding
    """
    batch_size = len(seq_lens)
    mask = np.zeros((batch_size, 1, max_len))
    
    for i, seq_len in enumerate(seq_lens):
        # Mark positions beyond actual sequence length as -inf
        mask[i, 0, seq_len:] = -np.inf
    
    return mask


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_attention():
    """
    Demonstrate the attention mechanism with example inputs.
    """
    print("="*70)
    print("Scaled Dot-Product Attention Demonstration")
    print("="*70)
    
    # Example 1: Simple 2D attention (single sequence, no batch)
    print("\n" + "="*70)
    print("Example 1: Basic Attention (Single Sequence)")
    print("="*70)
    
    seq_len = 4
    d_k = 8  # dimension of keys/queries
    d_v = 8  # dimension of values
    
    # Create random Q, K, V matrices
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)
    
    print(f"\nInput shapes:")
    print(f"  Q: {Q.shape} (queries)")
    print(f"  K: {K.shape} (keys)")
    print(f"  V: {V.shape} (values)")
    
    # Compute attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"\nOutput shapes:")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights: {attention_weights.shape}")
    
    print(f"\nAttention weights (each row sums to 1.0):")
    print(attention_weights)
    print(f"\nRow sums (should all be 1.0): {attention_weights.sum(axis=1)}")
    
    # Example 2: Batched attention
    print("\n" + "="*70)
    print("Example 2: Batched Attention")
    print("="*70)
    
    batch_size = 2
    Q_batch = np.random.randn(batch_size, seq_len, d_k)
    K_batch = np.random.randn(batch_size, seq_len, d_k)
    V_batch = np.random.randn(batch_size, seq_len, d_v)
    
    print(f"\nInput shapes:")
    print(f"  Q: {Q_batch.shape}")
    print(f"  K: {K_batch.shape}")
    print(f"  V: {V_batch.shape}")
    
    output_batch, attention_batch = scaled_dot_product_attention(Q_batch, K_batch, V_batch)
    
    print(f"\nOutput shapes:")
    print(f"  Output: {output_batch.shape}")
    print(f"  Attention weights: {attention_batch.shape}")
    
    # Example 3: Attention with causal mask
    print("\n" + "="*70)
    print("Example 3: Causal (Autoregressive) Attention")
    print("="*70)
    
    causal_mask = create_causal_mask(seq_len)
    print(f"\nCausal mask shape: {causal_mask.shape}")
    print(f"Causal mask (0 = attend, -inf = mask):")
    print(np.where(np.isinf(causal_mask), -999, causal_mask))  # Replace -inf with -999 for display
    
    output_causal, attention_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    
    print(f"\nCausal attention weights (lower triangular):")
    print(attention_causal)
    print(f"\nNotice: Each position only attends to current and previous positions")
    
    # Example 4: Cross-attention (Q from one sequence, K,V from another)
    print("\n" + "="*70)
    print("Example 4: Cross-Attention (Decoder attending to Encoder)")
    print("="*70)
    
    encoder_len = 6
    decoder_len = 4
    
    Q_decoder = np.random.randn(decoder_len, d_k)  # Decoder queries
    K_encoder = np.random.randn(encoder_len, d_k)  # Encoder keys
    V_encoder = np.random.randn(encoder_len, d_v)  # Encoder values
    
    print(f"\nInput shapes:")
    print(f"  Q (decoder): {Q_decoder.shape}")
    print(f"  K (encoder): {K_encoder.shape}")
    print(f"  V (encoder): {V_encoder.shape}")
    
    output_cross, attention_cross = scaled_dot_product_attention(Q_decoder, K_encoder, V_encoder)
    
    print(f"\nOutput shapes:")
    print(f"  Output: {output_cross.shape}")
    print(f"  Attention weights: {attention_cross.shape}")
    
    print(f"\nCross-attention weights:")
    print(f"(Each decoder position attends to all encoder positions)")
    print(attention_cross)
    
    print("\n" + "="*70)
    print("Demonstration Complete!")
    print("="*70)


def test_attention_properties():
    """
    Test mathematical properties of the attention mechanism.
    """
    print("\n" + "="*70)
    print("Testing Attention Properties")
    print("="*70)
    
    np.random.seed(42)
    seq_len = 5
    d_k = 16
    d_v = 16
    
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    # Test 1: Attention weights sum to 1
    row_sums = weights.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Attention weights should sum to 1"
    print("✓ Test 1 passed: Attention weights sum to 1.0")
    
    # Test 2: Output shape is correct
    expected_shape = (seq_len, d_v)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    print(f"✓ Test 2 passed: Output shape is {output.shape}")
    
    # Test 3: Weights shape is correct
    expected_weights_shape = (seq_len, seq_len)
    assert weights.shape == expected_weights_shape
    print(f"✓ Test 3 passed: Weights shape is {weights.shape}")
    
    # Test 4: Causal mask works
    mask = create_causal_mask(seq_len)
    _, masked_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    # Check upper triangle is zero (or very close to zero)
    upper_triangle = np.triu(masked_weights, k=1)
    assert np.allclose(upper_triangle, 0, atol=1e-7), "Upper triangle should be masked"
    print("✓ Test 4 passed: Causal mask correctly zeros upper triangle")
    
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)


if __name__ == "__main__":
    # Run demonstration
    demonstrate_attention()
    
    # Run tests
    test_attention_properties()
    
    print("\n" + "="*70)
    print("Scaled Dot-Product Attention Implementation Complete!")
    print("="*70)