package transforms

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/rolandtannous/gotorchx"
)

func TestNormalizeTransform(t *testing.T) {
	a := assert.New(t)
	trans := Normalize([]float32{10.0}, []float32{2.3})
	t1 := torch.NewTensor([]float32{10.2, 11.3, 9.2})
	t2 := trans.Run(t1)

	expected := torch.NewTensor([]float32{
		(10.2 - 10.0) / 2.3,
		(11.3 - 10.0) / 2.3,
		(9.2 - 10.0) / 2.3})
	a.True(torch.AllClose(t2, expected))
}
func TestNormalizeTransform3D(t *testing.T) {
	a := assert.New(t)
	trans := Normalize([]float32{1.0, 2.0, 3.0}, []float32{2.3, 2.4, 2.5})
	// an image in torch should be a 3D tensor with CHW format
	t1 := torch.NewTensor([][][]float32{{{10.2}}, {{11.3}}, {{9.2}}})
	t2 := trans.Run(t1)

	expected := torch.NewTensor([][][]float32{
		{{(10.2 - 1.0) / 2.3}},
		{{(11.3 - 2.0) / 2.4}},
		{{(9.2 - 3.0) / 2.5}}})
	a.True(torch.AllClose(t2, expected))
}

func TestNormalizeTransformPanic(t *testing.T) {
	a := assert.New(t)
	// mean and stddev should be 1 or 3 dims
	a.Panics(func() {
		trans := Normalize([]float32{1.0, 2.0, 3.0, 4.0, 5.0}, []float32{2.3, 2.4, 2.5})
		trans.Run(torch.NewTensor([]float32{1.0}))
	})
	a.Panics(func() {
		trans := Normalize([]float32{1.0, 2.0, 3.0}, []float32{2.3, 2.4})
		trans.Run(torch.NewTensor([]float32{1.0}))
	})

}
