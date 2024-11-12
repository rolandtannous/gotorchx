package functional

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	torch "github.com/wangkuiyi/gotorch"
)

func TestMultiHeadAttention(t *testing.T) {
	// Test dimensions
	seqLen := int64(4)    // S: sequence length
	batchSize := int64(2) // N: batch size
	embedDim := int64(8)  // E: embedding dimension
	numHeads := int64(2)  // embedDim must be divisible by numHeads

	// Create 3D input tensors with shape (S, N, E)
	query := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)
	key := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)
	value := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)

	// Create empty mask and dropout for optional parameters
	var mask, dropout torch.Tensor

	// Test basic attention
	output := MultiHeadAttention(query, key, value, numHeads, mask, dropout)

	// Check output shape should be (S, N, E)
	shape := output.Shape()
	assert.Equal(t, seqLen, shape[0], "Unexpected sequence length")
	assert.Equal(t, batchSize, shape[1], "Unexpected batch size")
	assert.Equal(t, embedDim, shape[2], "Unexpected embedding dimension")

	// Verify output is not nil and contains valid values
	assert.NotNil(t, output.T)
}

func TestMultiHeadAttentionWithMasks(t *testing.T) {
	seqLen := int64(4)
	batchSize := int64(2)
	embedDim := int64(8)
	numHeads := int64(2)

	query := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)
	key := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)
	value := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)

	t.Run("with_causal_mask", func(t *testing.T) {
		// Create a causal mask - we need it to be boolean type
		// First create a full mask of ones
		mask := torch.Full([]int64{batchSize, numHeads, seqLen, seqLen}, 1, false)
		mask = mask.Triu(1)
		// Convert to boolean by comparing with 1
		mask = mask.Eq(torch.Full([]int64{1}, 1, false))

		output := MultiHeadAttention(query, key, value, numHeads, mask, torch.Tensor{})

		shape := output.Shape()
		assert.Equal(t, seqLen, shape[0])
		assert.Equal(t, batchSize, shape[1])
		assert.Equal(t, embedDim, shape[2])
	})
}

func TestMultiHeadAttentionWithDropout(t *testing.T) {
	seqLen := int64(4)
	batchSize := int64(2)
	embedDim := int64(8)
	numHeads := int64(2)

	query := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)
	key := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)
	value := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)

	// Create dropout probability tensor using Full
	dropoutProb := torch.Full([]int64{1}, 0.5, false)

	output := MultiHeadAttention(query, key, value, numHeads, torch.Tensor{}, dropoutProb)

	shape := output.Shape()
	assert.Equal(t, seqLen, shape[0])
	assert.Equal(t, batchSize, shape[1])
	assert.Equal(t, embedDim, shape[2])
}

func TestMultiHeadAttentionErrors(t *testing.T) {
	t.Run("invalid_embed_dim", func(t *testing.T) {
		seqLen := int64(4)
		batchSize := int64(2)
		embedDim := int64(7) // Not divisible by numHeads
		numHeads := int64(2)

		query := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)
		key := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)
		value := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)

		assert.Panics(t, func() {
			MultiHeadAttention(query, key, value, numHeads, torch.Tensor{}, torch.Tensor{})
		})
	})

	t.Run("mismatched_dimensions", func(t *testing.T) {
		query := torch.RandN([]int64{4, 2, 8}, false)
		key := torch.RandN([]int64{5, 2, 8}, false) // Different sequence length
		value := torch.RandN([]int64{4, 2, 8}, false)

		assert.Panics(t, func() {
			MultiHeadAttention(query, key, value, 2, torch.Tensor{}, torch.Tensor{})
		})
	})
}

func TestMultiHeadAttentionDevice(t *testing.T) {
	// Skip if CUDA is not available
	if !torch.IsCUDAAvailable() {
		t.Skip("CUDA not available")
	}

	seqLen := int64(4)
	batchSize := int64(2)
	embedDim := int64(8)
	numHeads := int64(2)

	device := torch.NewDevice("cuda")

	// Create tensors on CPU first
	query := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)
	key := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)
	value := torch.RandN([]int64{seqLen, batchSize, embedDim}, false)

	// Move to GPU
	queryGPU := query.CopyTo(device)
	keyGPU := key.CopyTo(device)
	valueGPU := value.CopyTo(device)

	// Run on GPU
	outputGPU := MultiHeadAttention(
		queryGPU,
		keyGPU,
		valueGPU,
		numHeads,
		torch.Tensor{},
		torch.Tensor{},
	)

	// Move back to CPU for comparison
	outputGPUCPU := outputGPU.CopyTo(torch.NewDevice("cpu"))

	// Compare with CPU result
	outputCPU := MultiHeadAttention(query, key, value, numHeads, torch.Tensor{}, torch.Tensor{})
	assert.True(t, torch.AllClose(outputGPUCPU, outputCPU))
}

func TestFlashAttention(t *testing.T) {
	if !torch.IsCUDAAvailable() {
		t.Skip("CUDA not available")
	}

	batchSize := int64(2)
	numHeads := int64(8)
	seqLen := int64(128)
	headDim := int64(64)

	// Create base tensors
	query := torch.RandN([]int64{batchSize, numHeads, seqLen, headDim}, false)
	key := torch.RandN([]int64{batchSize, numHeads, seqLen, headDim}, false)
	value := torch.RandN([]int64{batchSize, numHeads, seqLen, headDim}, false)

	// Try both fp16 and bf16
	dtypes := []int8{torch.Half, torch.BFloat16}
	for _, dtype := range dtypes {
		t.Run(fmt.Sprintf("dtype_%d", dtype), func(t *testing.T) {
			// Convert to desired dtype and move to CUDA
			device := torch.NewDevice("cuda")
			q := query.CastTo(dtype).CopyTo(device)
			k := key.CastTo(dtype).CopyTo(device)
			v := value.CastTo(dtype).CopyTo(device)

			// Test without causal mask
			output := FlashAttention(q, k, v, 0.0, false)
			assert.Equal(t, []int64{batchSize, numHeads, seqLen, headDim}, output.Shape())

			// Test with causal mask
			outputCausal := FlashAttention(q, k, v, 0.0, true)
			assert.Equal(t, []int64{batchSize, numHeads, seqLen, headDim}, outputCausal.Shape())
		})
	}
}
