package gotorch_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	torch "github.com/rolandtannous/gotorchx"
)

func TestDimname(t *testing.T) {
	t.Run("basic dimension names", func(t *testing.T) {
		// Test basic name creation
		name := torch.NewDimname("batch", false)
		assert.Equal(t, "batch", name.String())

		// Test wildcard creation
		wild := torch.NewDimname("", true)
		assert.Equal(t, "*", wild.String())

		// Test another basic name
		channel := torch.NewDimname("channel", false)
		assert.Equal(t, "channel", channel.String())
	})

	t.Run("tensor with named dimensions - 2D", func(t *testing.T) {
		// Create a 2D tensor
		tensor := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
		batch := torch.NewDimname("batch", false)
		features := torch.NewDimname("features", false)

		// Set names and verify
		namedTensor := tensor.SetNames(batch, features)
		names := namedTensor.Names()

		assert.Equal(t, 2, len(names))
		assert.Equal(t, "batch", names[0].String())
		assert.Equal(t, "features", names[1].String())
	})

	t.Run("tensor with named dimensions - 3D", func(t *testing.T) {
		// Create a 3D tensor
		tensor := torch.NewTensor([][][]float32{
			{{1, 2}, {3, 4}},
			{{5, 6}, {7, 8}},
		})
		batch := torch.NewDimname("batch", false)
		height := torch.NewDimname("height", false)
		width := torch.NewDimname("width", false)

		// Set names and verify
		namedTensor := tensor.SetNames(batch, height, width)
		names := namedTensor.Names()

		assert.Equal(t, 3, len(names))
		assert.Equal(t, "batch", names[0].String())
		assert.Equal(t, "height", names[1].String())
		assert.Equal(t, "width", names[2].String())
	})

	t.Run("tensor with mixed names and wildcards", func(t *testing.T) {
		tensor := torch.NewTensor([][]float32{{1, 2, 3}, {4, 5, 6}})
		batch := torch.NewDimname("batch", false)
		wild := torch.NewDimname("", true)

		// Set names with wildcard and verify
		namedTensor := tensor.SetNames(batch, wild)
		names := namedTensor.Names()

		assert.Equal(t, 2, len(names))
		assert.Equal(t, "batch", names[0].String())
		assert.Equal(t, "*", names[1].String())
	})

	t.Run("empty tensor with names", func(t *testing.T) {
		// Create an empty tensor with shape [0, 2]
		tensor := torch.Empty([]int64{0, 2}, false)
		batch := torch.NewDimname("batch", false)
		features := torch.NewDimname("features", false)

		// Set names and verify
		namedTensor := tensor.SetNames(batch, features)
		names := namedTensor.Names()

		assert.Equal(t, 2, len(names))
		assert.Equal(t, "batch", names[0].String())
		assert.Equal(t, "features", names[1].String())
	})

	t.Run("tensor name persistence", func(t *testing.T) {
		// Create and name a tensor
		tensor := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
		batch := torch.NewDimname("batch", false)
		features := torch.NewDimname("features", false)

		// Set names
		namedTensor := tensor.SetNames(batch, features)

		// Perform some operations
		namedTensor = namedTensor.Add(namedTensor, 1)

		// Verify names are still present
		names := namedTensor.Names()
		assert.Equal(t, 2, len(names))
		assert.Equal(t, "batch", names[0].String())
		assert.Equal(t, "features", names[1].String())
	})

	t.Run("zero dimension tensor", func(t *testing.T) {
		// Create a tensor with shape [1]
		tensor := torch.NewTensor([]float32{1.0})
		// Squeeze it to make it a scalar tensor
		tensor = tensor.Squeeze()

		// Verify getting names works with zero dimensions
		names := tensor.Names()
		assert.Equal(t, 0, len(names))
	})

	t.Run("multiple name operations", func(t *testing.T) {
		tensor := torch.NewTensor([][]float32{{1, 2}, {3, 4}})

		// First naming
		batch1 := torch.NewDimname("batch", false)
		features1 := torch.NewDimname("features", false)
		tensor = tensor.SetNames(batch1, features1)

		// Second naming (overwrite)
		batch2 := torch.NewDimname("samples", false)
		features2 := torch.NewDimname("dims", false)
		tensor = tensor.SetNames(batch2, features2)

		names := tensor.Names()
		assert.Equal(t, 2, len(names))
		assert.Equal(t, "samples", names[0].String())
		assert.Equal(t, "dims", names[1].String())
	})
}
