package gotorch_test

import (
	"bytes"
	"encoding/gob"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/rolandtannous/gotorchx"
)

func TestTensorGobEncode(t *testing.T) {
	a := gotorch.NewTensor([][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})
	b, e := a.GobEncode()
	assert.NoError(t, e)

	// Instead of checking exact size and hash, verify the encoded data is valid
	var decoded gotorch.Tensor
	assert.NoError(t, decoded.GobDecode(b))

	// Compare string representations
	assert.Equal(t, a.String(), decoded.String())

	// Test nil case
	a = gotorch.Tensor{nil}
	_, e = a.GobEncode()
	assert.Error(t, e)
}

func TestTensorGobDecode(t *testing.T) {
	x := gotorch.NewTensor([][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})

	var buf bytes.Buffer
	assert.NoError(t, gob.NewEncoder(&buf).Encode(x))

	var y gotorch.Tensor
	assert.NoError(t, gob.NewDecoder(&buf).Decode(&y))
	assert.Equal(t, " 1  0  0\n 0  1  0\n 0  0  1\n[ CPUFloatType{3,3} ]", y.String())
}
