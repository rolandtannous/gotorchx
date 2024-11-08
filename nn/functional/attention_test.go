package functional

import (
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
