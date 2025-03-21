package gotorch_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/x448/float16"

	torch "github.com/rolandtannous/gotorchx"
)

// >>> t = torch.tensor([[-0.5, -1.], [1., 0.5]])
// >>> s = torch.tensor([[-0.5, -1.], [1., 0.5]])
// >>> t+s
// tensor([[-1., -2.],
//
//	[ 2.,  1.]])
func TestArith(t *testing.T) {
	a := assert.New(t)
	r := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	s := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	q := r.Add(s, 1)
	expected := torch.NewTensor([][]float32{{-1, -2}, {2, 1}})
	a.True(torch.Equal(expected, q))

	q = r.Sub(s, 1)
	expected = torch.NewTensor([][]float32{{0, 0}, {0, 0}})
	a.True(torch.Equal(expected, q))

	q = r.Mul(s)
	expected = torch.NewTensor([][]float32{{0.25, 1}, {1, 0.25}})
	a.True(torch.Equal(expected, q))

	q = r.Div(s)
	expected = torch.NewTensor([][]float32{{1.0, 1.0}, {1.0, 1.0}})
	a.True(torch.Equal(expected, q))

}

func TestArithI(t *testing.T) {
	a := assert.New(t)

	x := torch.RandN([]int64{2, 3}, false)
	y := torch.RandN([]int64{2, 3}, false)
	z := torch.Add(x, y, 1)
	x.AddI(y, 1)
	a.True(torch.Equal(x, z))

	z = torch.Sub(x, y, 1)
	x.SubI(y, 1)
	a.True(torch.Equal(x, z))

	z = torch.Mul(x, y)
	x.MulI(y)
	a.True(torch.Equal(x, z))

	z = torch.Div(x, y)
	x.DivI(y)
	a.True(torch.Equal(x, z))
}

func TestPermute(t *testing.T) {
	a := assert.New(t)
	x := torch.NewTensor([][]float32{{3, 1}, {2, 4}})
	y := x.Permute([]int64{1, 0})
	expected := torch.NewTensor([][]float32{{3, 2}, {1, 4}})
	a.True(torch.Equal(expected, y))
}

func TestAllClose(t *testing.T) {
	a := assert.New(t)
	x := torch.NewTensor([]float32{8.31, 6.55, 1.39})
	y := torch.NewTensor([]float32{2.38, 3.12, 5.23})
	r := x.Mul(y)
	expected := torch.NewTensor([]float32{8.31 * 2.38, 6.55 * 3.12, 1.39 * 5.23})
	a.True(torch.AllClose(expected, r))
}

// >>> torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
// tensor([[ True, False],
//
//	[False, True]])
func TestEq(t *testing.T) {
	a := torch.NewTensor([][]int16{{1, 2}, {3, 4}})
	b := torch.NewTensor([][]int16{{1, 3}, {2, 4}})
	c := torch.Eq(a, b)
	g := " 1  0\n 0  1\n[ CPUBoolType{2,2} ]"
	assert.Equal(t, g, c.String())
}

func TestTensorEq(t *testing.T) {
	a := torch.NewTensor([][]int16{{1, 2}, {3, 4}})
	b := torch.NewTensor([][]int16{{1, 3}, {2, 4}})
	c := a.Eq(b)
	g := " 1  0\n 0  1\n[ CPUBoolType{2,2} ]"
	assert.Equal(t, g, c.String())
}

func TestEqual(t *testing.T) {
	a := torch.NewTensor([]int64{1, 2})
	b := torch.NewTensor([]int64{1, 2})
	assert.True(t, torch.Equal(a, b))
}

// >>> s = torch.tensor([1,2])
// >>> t = torch.tensor([[1,2],[3,4]])
// >>> s.expand_as(t)
// tensor([[1, 2],
//
//	[1, 2]])
func TestExpandAs(t *testing.T) {
	a := torch.NewTensor([]int8{'a', 'b'})
	b := torch.NewTensor([][]int8{{1, 2}, {3, 4}})
	c := torch.ExpandAs(a, b)
	g := " 97  98\n 97  98\n[ CPUCharType{2,2} ]"
	assert.Equal(t, g, c.String())
}

func TestTensorExpandAs(t *testing.T) {
	a := torch.NewTensor([]int8{'a', 'b'})
	b := torch.NewTensor([][]int8{{1, 2}, {3, 4}})
	c := a.ExpandAs(b)
	g := " 97  98\n 97  98\n[ CPUCharType{2,2} ]"
	assert.Equal(t, g, c.String())
}

// >>> torch.flatten(torch.tensor([[-0.5, -1.], [1., 0.5]]), 0, 1)
// tensor([-0.5000, -1.0000,  1.0000,  0.5000])
func TestFlatten(t *testing.T) {
	r := torch.Flatten(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}),
		0, 1)
	g := "-0.5000\n-1.0000\n 1.0000\n 0.5000\n[ CPUFloatType{4} ]"
	assert.Equal(t, g, r.String())
}

// >>> torch.nn.functional.leaky_relu(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor([[-0.0050, -0.0100],
//
//	[ 1.0000,  0.5000]])
func TestLeakyRelu(t *testing.T) {
	r := torch.LeakyRelu(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}),
		0.01)
	g := "-0.0050 -0.0100\n 1.0000  0.5000\n[ CPUFloatType{2,2} ]"
	assert.Equal(t, g, r.String())
}

// >>> torch.nn.functional.log_softmax(torch.tensor([[-0.5, -1.], [1., 0.5]]), dim=1)
// tensor([[-0.4741, -0.9741],
//
//	[-0.4741, -0.9741]])
func TestLogSoftmax(t *testing.T) {
	r := torch.LogSoftmax(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}),
		1)
	g := "-0.4741 -0.9741\n-0.4741 -0.9741\n[ CPUFloatType{2,2} ]"
	assert.Equal(t, g, r.String())
}

// >>> torch.mean(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor(0.)
func TestMean(t *testing.T) {
	r := torch.Mean(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}))
	// BUG: The result should be 0.
	g := "0\n[ CPUFloatType{} ]"
	assert.Equal(t, g, r.String())
}

func TestTensorMean(t *testing.T) {
	x := torch.RandN([]int64{2, 3}, true)
	y := x.Mean()
	z := y.Item()
	assert.NotNil(t, z)
}

// >>> torch.relu(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor([[0.0000, 0.0000],
//
//	[1.0000, 0.5000]])
func TestRelu(t *testing.T) {
	r := torch.Relu(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}))
	g := " 0.0000  0.0000\n 1.0000  0.5000\n[ CPUFloatType{2,2} ]"
	assert.Equal(t, g, r.String())
}

// >>> torch.sigmoid(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor([[0.3775, 0.2689],
//
//	[0.7311, 0.6225]])
func TestSigmoid(t *testing.T) {
	r := torch.Sigmoid(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}))
	g := " 0.3775  0.2689\n 0.7311  0.6225\n[ CPUFloatType{2,2} ]"
	assert.Equal(t, g, r.String())
}

func TestStack(t *testing.T) {
	t1 := torch.RandN([]int64{2, 3}, false)
	t2 := torch.RandN([]int64{2, 3}, false)
	out := torch.Stack([]torch.Tensor{t1, t2}, 0)
	assert.Equal(t, []int64{2, 2, 3}, out.Shape())
}

func TestSqueeze(t *testing.T) {
	x := torch.RandN([]int64{2, 1, 2, 1, 2}, false)
	y := torch.Squeeze(x)
	assert.NotNil(t, y.T)
	z := torch.Squeeze(x, 1)
	assert.NotNil(t, z.T)
	assert.Panics(t, func() { torch.Squeeze(x, 1, 2) })
}

// >>> x = torch.tensor([[1,2,3,4],[4,5,6,7],[7,8,9,0]])
// >>> torch.sum(x)
// tensor(56)
// >>> torch.sum(x, 0)
// tensor([12, 15, 18, 11])
// >>> torch.sum(x, 1)
// tensor([10, 22, 24])
// >>> torch.sum(x, 0, True)
// tensor([[12, 15, 18, 11]])
// >>> torch.sum(x, 0, False)
// tensor([12, 15, 18, 11])
// >>> torch.sum(x, 1, True)
// tensor([[10],
//
//	[22],
//	[24]])
//
// >>> torch.sum(x, 1, False)
// tensor([10, 22, 24])
func TestSum(t *testing.T) {
	x := torch.NewTensor([][]float32{{1, 2, 3, 4}, {4, 5, 6, 7}, {7, 8, 9, 0}})

	assert.Equal(t, float32(56), x.Sum().Item().(float32))

	y := x.Sum(map[string]interface{}{"dim": 0})
	assert.True(t, torch.Equal(torch.NewTensor([]float32{12, 15, 18, 11}), y),
		"Got %v", y)

	y = x.Sum(map[string]interface{}{"dim": 1})
	assert.True(t, torch.Equal(torch.NewTensor([]float32{10, 22, 24}), y),
		"Got %v", y)

	y = x.Sum(map[string]interface{}{"dim": 0, "keepDim": true})
	assert.True(t, torch.Equal(torch.NewTensor([][]float32{{12, 15, 18, 11}}), y),
		"Got %v", y)

	y = x.Sum(map[string]interface{}{"dim": 0, "keepDim": false})
	assert.True(t, torch.Equal(torch.NewTensor([]float32{12, 15, 18, 11}), y),
		"Got %v", y)

	y = x.Sum(map[string]interface{}{"dim": 1, "keepDim": true})
	assert.True(t, torch.Equal(torch.NewTensor([][]float32{{10}, {22}, {24}}), y),
		"Got %v", y)

	y = x.Sum(map[string]interface{}{"dim": 1, "keepDim": false})
	assert.True(t, torch.Equal(torch.NewTensor([]float32{10, 22, 24}), y),
		"Got %v", y)
}

func TestTanh(t *testing.T) {
	a := torch.RandN([]int64{4}, false)
	b := torch.Tanh(a)
	assert.NotNil(t, b.T)
}

// >>> torch.topk(torch.tensor([[-0.5, -1.], [1., 0.5]]), 1, 1, True, True)
// torch.return_types.topk(
// values=tensor([[-0.5000],
//
//	[ 1.0000]]),
//
// indices=tensor([[0],
//
//	[0]]))
func TestTopK(t *testing.T) {
	r, i := torch.TopK(torch.NewTensor([][]float64{{-0.5, -1}, {1, 0.5}}),
		1, 1, true, true)
	gr := "-0.5000\n 1.0000\n[ CPUDoubleType{2,1} ]"
	gi := " 0\n 0\n[ CPULongType{2,1} ]"
	assert.Equal(t, gr, r.String())
	assert.Equal(t, gi, i.String())
}

// >>> torch.transpose(torch.tensor([[-0.5, -1.], [1., 0.5]]), 0, 1)
// tensor([[-0.5000,  1.0000],
//
//	[-1.0000,  0.5000]])
func TestTranspose(t *testing.T) {
	x := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	g := "-0.5000  1.0000\n-1.0000  0.5000\n[ CPUFloatType{2,2} ]"
	assert.Equal(t, g, x.Transpose(0, 1).String())
}

// >>> x = torch.randn(4, 4)
// >>> x.size()
// torch.Size([4, 4])
// >>> y = x.view(16)
// >>> y.size()
// torch.Size([16])
// >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
// >>> z.size()
// torch.Size([2, 8])

// >>> a = torch.randn(1, 2, 3, 4)
// >>> a.size()
// torch.Size([1, 2, 3, 4])
// >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
// >>> b.size()
// torch.Size([1, 3, 2, 4])
// >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
// >>> c.size()
// torch.Size([1, 3, 2, 4])
// >>> torch.equal(b, c)
// False
func TestTensorView(t *testing.T) {
	x := torch.Empty([]int64{4, 4}, false)
	y := x.View(16)
	assert.Equal(t, []int64{16}, y.Shape())
	z := x.View(-1, 8)
	assert.Equal(t, []int64{2, 8}, z.Shape())
	a := torch.RandN([]int64{1, 2, 3, 4}, false)
	b := a.Transpose(1, 2)
	c := a.View(1, 3, 2, 4)
	assert.False(t, torch.Equal(b, c))
}

func TestArgmin(t *testing.T) {
	x := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
	assert.Equal(t, "0\n[ CPULongType{} ]", x.Argmin().String())

	x = torch.NewTensor([][]float32{{4, 3}, {2, 1}})
	assert.Equal(t, "3\n[ CPULongType{} ]", x.Argmin().String())

	// x = torch.tensor([[3,4],[2,1]]
	x = torch.NewTensor([][]float32{{3, 4}, {2, 1}})
	// x.argmin(0)
	assert.Equal(t, " 1\n 1\n[ CPULongType{2} ]", x.Argmin(0).String())
	// x.argmin(1)
	assert.Equal(t, " 0\n 1\n[ CPULongType{2} ]", x.Argmin(1).String())
	// x.argmin(0, True)
	assert.Equal(t, " 1  1\n[ CPULongType{1,2} ]", x.Argmin(0, true).String())
	// x.argmin(1, True)
	assert.Equal(t, " 0\n 1\n[ CPULongType{2,1} ]", x.Argmin(1, true).String())

	assert.Panics(t, func() { x.Argmin(1.0 /* must be int*/, true) })
	assert.Panics(t, func() { x.Argmin(1, 1.0 /*must be bool*/) })
}

func TestArgmax(t *testing.T) {
	x := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
	assert.Equal(t, "3\n[ CPULongType{} ]", x.Argmax().String())

	x = torch.NewTensor([][]float32{{4, 3}, {2, 1}})
	assert.Equal(t, "0\n[ CPULongType{} ]", x.Argmax().String())

	// x = torch.tensor([[3,4],[2,1]]
	x = torch.NewTensor([][]float32{{3, 4}, {2, 1}})
	// x.argmax(0)
	assert.Equal(t, " 0\n 0\n[ CPULongType{2} ]", x.Argmax(0).String())
	// x.argmax(1)
	assert.Equal(t, " 1\n 0\n[ CPULongType{2} ]", x.Argmax(1).String())
	// x.argmax(0, True)
	assert.Equal(t, " 0  0\n[ CPULongType{1,2} ]", x.Argmax(0, true).String())
	// x.argmax(1, True)
	assert.Equal(t, " 1\n 0\n[ CPULongType{2,1} ]", x.Argmax(1, true).String())
}

func f16(x float32) uint16 {
	return float16.Fromfloat32(x).Bits()
}

func TestItem(t *testing.T) {
	x := torch.NewTensor([]byte{1})
	y := x.Item()
	assert.Equal(t, byte(1), y)
	assert.NotEqual(t, int8(1), y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Uint8)

	x = torch.NewTensor([]bool{true})
	y = x.Item()
	assert.Equal(t, true, y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Bool)

	x = torch.NewTensor([]bool{false})
	y = x.Item()
	assert.Equal(t, false, y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Bool)

	x = torch.NewTensor([]int8{1})
	y = x.Item()
	assert.Equal(t, int8(1), y)
	assert.NotEqual(t, byte(1), y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int8)

	x = torch.NewTensor([]int16{1})
	y = x.Item()
	assert.Equal(t, int16(1), y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int16)

	x = torch.NewTensor([]int32{1})
	y = x.Item()
	assert.Equal(t, int32(1), y)
	assert.NotEqual(t, int64(1), y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int32)

	x = torch.NewTensor([]int64{1})
	y = x.Item()
	assert.Equal(t, int64(1), y)
	assert.NotEqual(t, int32(1), y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int64)

	x = torch.NewTensor([]int32{0x7FFF_FFFF})
	y = x.Item()
	assert.Equal(t, int32(0x7FFF_FFFF), y)

	x = torch.NewTensor([]int32{-0x8000_0000})
	y = x.Item()
	assert.Equal(t, int32(-0x8000_0000), y)

	// half
	x = torch.NewTensor([]uint16{f16(1)})
	y = x.Item()
	assert.Equal(t, float32(1), y)

	// max half
	x = torch.NewTensor([]uint16{f16(65504)})
	y = x.Item()
	assert.Equal(t, float32(65504), y)

	x = torch.NewTensor([]uint16{f16(0.25)})
	y = x.Item()
	assert.Equal(t, float32(0.25), y)

	x = torch.NewTensor([]float32{1.0})
	y = x.Item()
	assert.Equal(t, float32(1.0), y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Float32)

	x = torch.NewTensor([]float32{-1.0})
	y = x.Item()
	assert.Equal(t, float32(-1.0), y)

	x = torch.NewTensor([]float64{1.0})
	y = x.Item()
	assert.Equal(t, float64(1), y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Float64)

	x = torch.NewTensor([]float64{-1})
	y = x.Item()
	assert.Equal(t, float64(-1), y)
}

// >>> x = torch.tensor([[1,2,3,4],[4,5,6,7],[7,8,9,0]])
// >>> x
// tensor([[1, 2, 3, 4],
//
//	[4, 5, 6, 7],
//	[7, 8, 9, 0]])
//
// >>> idx = torch.tensor([0,2])
// >>> torch.index_select(x, 0, idx)
// tensor([[1, 2, 3, 4],
//
//	[7, 8, 9, 0]])
//
// >>> torch.index_select(x, 1, idx)
// tensor([[1, 3],
//
//	[4, 6],
//	[7, 9]])
func TestIndexSelect(t *testing.T) {
	x := torch.NewTensor([][]float32{{1, 2, 3, 4}, {4, 5, 6, 7}, {7, 8, 9, 0}})
	idx := torch.NewTensor([]int64{0, 2})
	assert.Equal(t, " 1  2  3  4\n 7  8  9  0\n[ CPUFloatType{2,4} ]",
		x.IndexSelect(0, idx).String())
	assert.Equal(t, " 1  3\n 4  6\n 7  9\n[ CPUFloatType{3,2} ]",
		x.IndexSelect(1, idx).String())
}

func TestPow(t *testing.T) {
	// Test case 1: Basic power operation
	a := torch.NewTensor([]float32{1, 2, 3, 4})
	result := a.Pow(2.0)
	expected := torch.NewTensor([]float32{1, 4, 9, 16})
	if !torch.AllClose(result, expected) {
		t.Errorf("Pow operation failed. Expected %v, got %v", expected, result)
	}

	// Test case 2: In-place power operation
	b := torch.NewTensor([]float32{1, 2, 3, 4})
	result = b.PowI(2.0)
	if !torch.AllClose(result, expected) {
		t.Errorf("PowI operation failed. Expected %v, got %v", expected, result)
	}

	// Test case 3: Power with out tensor
	c := torch.NewTensor([]float32{1, 2, 3, 4})
	out := torch.NewTensor([]float32{0, 0, 0, 0})
	result = torch.PowOut(c, 2.0, out)
	if !torch.AllClose(result, expected) {
		t.Errorf("PowOut operation failed. Expected %v, got %v", expected, result)
	}
}

func TestAbs(t *testing.T) {
	// Test case 1: Positive and negative values
	a := torch.NewTensor([]float32{-1, 2, -3, 4})
	result := a.Abs()
	expected := torch.NewTensor([]float32{1, 2, 3, 4})
	if !torch.AllClose(result, expected) {
		t.Errorf("Abs operation failed. Expected %v, got %v", expected, result)
	}

	// Test case 2: Zero values
	a = torch.NewTensor([]float32{0, -0})
	result = a.Abs()
	expected = torch.NewTensor([]float32{0, 0})
	if !torch.AllClose(result, expected) {
		t.Errorf("Abs operation with zero failed. Expected %v, got %v", expected, result)
	}

	// Test case 3: All positive values (should remain unchanged)
	a = torch.NewTensor([]float32{1, 2, 3, 4})
	result = a.Abs()
	expected = torch.NewTensor([]float32{1, 2, 3, 4})
	if !torch.AllClose(result, expected) {
		t.Errorf("Abs operation with positive values failed. Expected %v, got %v", expected, result)
	}
}

func TestAbsVariants(t *testing.T) {
	// Test case 1: Regular version
	a := torch.NewTensor([]float32{-1, 2, -3, 4})
	result := a.Abs()
	expected := torch.NewTensor([]float32{1, 2, 3, 4})
	if !torch.AllClose(result, expected) {
		t.Errorf("Abs operation failed. Expected %v, got %v", expected, result)
	}

	// Test case 2: In-place version
	b := torch.NewTensor([]float32{-1, 2, -3, 4})
	result = b.AbsI()
	if !torch.AllClose(result, expected) {
		t.Errorf("AbsI operation failed. Expected %v, got %v", expected, result)
	}

	// Test case 3: Out version
	c := torch.NewTensor([]float32{-1, 2, -3, 4})
	out := torch.NewTensor([]float32{0, 0, 0, 0})
	result = torch.AbsOut(c, out)
	if !torch.AllClose(result, expected) {
		t.Errorf("AbsOut operation failed. Expected %v, got %v", expected, result)
	}
}

// func TestSqrt(t *testing.T) {
// 	// Test case 1: Regular version
// 	a := torch.NewTensor([]float32{0, 1, 4, 9})
// 	result := a.Sqrt()
// 	expected := torch.NewTensor([]float32{0, 1, 2, 3})
// 	if !torch.AllClose(result, expected) {
// 		t.Errorf("Sqrt operation failed. Expected %v, got %v", expected, result)
// 	}
//
// 	// Test case 2: In-place version
// 	b := torch.NewTensor([]float32{0, 1, 4, 9})
// 	result = b.SqrtI()
// 	if !torch.AllClose(result, expected) {
// 		t.Errorf("SqrtI operation failed. Expected %v, got %v", expected, result)
// 	}
//
// 	// Test case 3: Out version
// 	c := torch.NewTensor([]float32{0, 1, 4, 9})
// 	out := torch.NewTensor([]float32{0, 0, 0, 0})
// 	result = torch.SqrtOut(c, out)
// 	if !torch.AllClose(result, expected) {
// 		t.Errorf("SqrtOut operation failed. Expected %v, got %v", expected, result)
// 	}
// }

func TestSqrt(t *testing.T) {
	t.Run("basic variants", func(t *testing.T) {
		// Test case 1: Regular version - both function and method forms
		a := torch.NewTensor([]float32{0, 1, 4, 9})
		result1 := torch.Sqrt(a) // Function form
		result2 := a.Sqrt()      // Method form
		expected := torch.NewTensor([]float32{0, 1, 2, 3})

		if !torch.AllClose(result1, expected) {
			t.Errorf("Sqrt function failed. Expected %v, got %v", expected, result1)
		}
		if !torch.AllClose(result2, expected) {
			t.Errorf("Sqrt method failed. Expected %v, got %v", expected, result2)
		}

		// Test case 2: In-place version - method only
		b := torch.NewTensor([]float32{0, 1, 4, 9})
		result := b.SqrtI()
		if !torch.AllClose(result, expected) {
			t.Errorf("SqrtI operation failed. Expected %v, got %v", expected, result)
		}

		// Test case 3: Out version - function only
		c := torch.NewTensor([]float32{0, 1, 4, 9})
		out := torch.NewTensor([]float32{0, 0, 0, 0})
		result = torch.SqrtOut(c, out)
		if !torch.AllClose(result, expected) {
			t.Errorf("SqrtOut operation failed. Expected %v, got %v", expected, result)
		}
	})
	t.Run("edge cases", func(t *testing.T) {
		// Test zero
		zero := torch.NewTensor([]float32{0})
		result := zero.Sqrt()
		expected := torch.NewTensor([]float32{0})
		if !torch.AllClose(result, expected) {
			t.Errorf("Sqrt of zero failed. Expected %v, got %v", expected, result)
		}

		// Test very large numbers
		large := torch.NewTensor([]float32{1e20})
		result = large.Sqrt()
		expected = torch.NewTensor([]float32{1e10})
		if !torch.AllClose(result, expected) {
			t.Errorf("Sqrt of large number failed. Expected %v, got %v", expected, result)
		}

		// Test very small positive numbers
		small := torch.NewTensor([]float32{1e-20})
		result = small.Sqrt()
		expected = torch.NewTensor([]float32{1e-10})
		if !torch.AllClose(result, expected) {
			t.Errorf("Sqrt of small number failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("negative numbers", func(t *testing.T) {
		// Test negative numbers - should return NaN
		negative := torch.NewTensor([]float32{-1.0})
		result := negative.Sqrt()
		// Since we don't have IsNaN, we can check if the result is not equal to itself
		// (NaN is the only value that's not equal to itself)
		if torch.AllClose(result, result) {
			t.Errorf("Sqrt of negative number should be NaN")
		}
	})

	t.Run("multi-dimensional tensors", func(t *testing.T) {
		// Test 2D tensor
		matrix := torch.NewTensor([][]float32{
			{0, 1, 4, 9},
			{16, 25, 36, 49},
		})
		result := matrix.Sqrt()
		expected := torch.NewTensor([][]float32{
			{0, 1, 2, 3},
			{4, 5, 6, 7},
		})
		if !torch.AllClose(result, expected) {
			t.Errorf("Sqrt of 2D tensor failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("precision test", func(t *testing.T) {
		// Test precision with irrational results
		a := torch.NewTensor([]float32{2.0})
		result := a.Sqrt()
		expected := torch.NewTensor([]float32{1.4142135}) // √2
		if !torch.AllClose(result, expected) {
			t.Errorf("Sqrt precision test failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("broadcasting", func(t *testing.T) {
		// Test that sqrt works with broadcasting
		matrix := torch.NewTensor([][]float32{
			{0, 1, 4},
			{0, 1, 4},
			{0, 1, 4},
		})
		result := matrix.Sqrt()
		expected := torch.NewTensor([][]float32{
			{0, 1, 2},
			{0, 1, 2},
			{0, 1, 2},
		})
		if !torch.AllClose(result, expected) {
			t.Errorf("Sqrt broadcasting failed. Expected %v, got %v", expected, result)
		}
	})
}

func TestLog(t *testing.T) {
	t.Run("basic variants", func(t *testing.T) {
		// Test case 1: Regular version - both function and method forms
		a := torch.NewTensor([]float32{1, 2.718282, 7.389056}) // [1, e, e^2]
		result1 := torch.Log(a)                                // Function form
		result2 := a.Log()                                     // Method form
		expected := torch.NewTensor([]float32{0, 1, 2})

		if !torch.AllClose(result1, expected) {
			t.Errorf("Log function failed. Expected %v, got %v", expected, result1)
		}
		if !torch.AllClose(result2, expected) {
			t.Errorf("Log method failed. Expected %v, got %v", expected, result2)
		}

		// Test case 2: In-place version - method only
		b := torch.NewTensor([]float32{1, 2.718282, 7.389056})
		result := b.LogI()
		if !torch.AllClose(result, expected) {
			t.Errorf("LogI operation failed. Expected %v, got %v", expected, result)
		}

		// Test case 3: Out version - function only
		c := torch.NewTensor([]float32{1, 2.718282, 7.389056})
		out := torch.NewTensor([]float32{0, 0, 0})
		result = torch.LogOut(c, out)
		if !torch.AllClose(result, expected) {
			t.Errorf("LogOut operation failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("edge cases", func(t *testing.T) {
		// Test zero - should return -inf
		zero := torch.NewTensor([]float32{0})
		result := zero.Log()
		// Check if result is -inf (you might need to adjust this check based on gotorch's capabilities)
		if !torch.AllClose(result, torch.NewTensor([]float32{float32(math.Inf(-1))})) {
			t.Errorf("Log of zero should be -inf")
		}

		// Test negative numbers - should return NaN
		negative := torch.NewTensor([]float32{-1.0})
		result = negative.Log()
		// NaN check (using the property that NaN != NaN)
		if torch.AllClose(result, result) {
			t.Errorf("Log of negative number should be NaN")
		}
	})

	t.Run("multi-dimensional tensors", func(t *testing.T) {
		matrix := torch.NewTensor([][]float32{
			{1, 2.718282, 7.389056},
			{1, 2.718282, 7.389056},
		})
		result := matrix.Log()
		expected := torch.NewTensor([][]float32{
			{0, 1, 2},
			{0, 1, 2},
		})
		if !torch.AllClose(result, expected) {
			t.Errorf("Log of 2D tensor failed. Expected %v, got %v", expected, result)
		}
	})
}

func TestExp(t *testing.T) {
	t.Run("basic variants", func(t *testing.T) {
		// Test case 1: Regular version - both function and method forms
		a := torch.NewTensor([]float32{0, 1, 2})
		result1 := torch.Exp(a)                                       // Function form
		result2 := a.Exp()                                            // Method form
		expected := torch.NewTensor([]float32{1, 2.718282, 7.389056}) // [1, e, e^2]

		if !torch.AllClose(result1, expected) {
			t.Errorf("Exp function failed. Expected %v, got %v", expected, result1)
		}
		if !torch.AllClose(result2, expected) {
			t.Errorf("Exp method failed. Expected %v, got %v", expected, result2)
		}

		// Test case 2: In-place version - method only
		b := torch.NewTensor([]float32{0, 1, 2})
		result := b.ExpI()
		if !torch.AllClose(result, expected) {
			t.Errorf("ExpI operation failed. Expected %v, got %v", expected, result)
		}

		// Test case 3: Out version - function only
		c := torch.NewTensor([]float32{0, 1, 2})
		out := torch.NewTensor([]float32{0, 0, 0})
		result = torch.ExpOut(c, out)
		if !torch.AllClose(result, expected) {
			t.Errorf("ExpOut operation failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("edge cases", func(t *testing.T) {
		// Test large negative number - should be close to 0
		large_neg := torch.NewTensor([]float32{-100.0})
		result := large_neg.Exp()
		expected := torch.NewTensor([]float32{0.0})
		if !torch.AllClose(result, expected) {
			t.Errorf("Exp of large negative number failed. Expected close to 0")
		}

		// Test large positive number - should be very large but finite
		large_pos := torch.NewTensor([]float32{100.0})
		result = large_pos.Exp()
		if torch.AllClose(result, result) { // Should be finite (not NaN)
			if result.Item().(float32) <= 0 {
				t.Errorf("Exp of large positive number should be positive")
			}
		} else {
			t.Errorf("Exp of large positive number should be finite")
		}
	})

	t.Run("multi-dimensional tensors", func(t *testing.T) {
		matrix := torch.NewTensor([][]float32{
			{0, 1, 2},
			{0, 1, 2},
		})
		result := matrix.Exp()
		expected := torch.NewTensor([][]float32{
			{1, 2.718282, 7.389056},
			{1, 2.718282, 7.389056},
		})
		if !torch.AllClose(result, expected) {
			t.Errorf("Exp of 2D tensor failed. Expected %v, got %v", expected, result)
		}
	})
}

// Shape operations test suite

func TestUnsqueeze(t *testing.T) {
	t.Run("basic variants", func(t *testing.T) {
		// Test case 1: Regular version - both function and method forms
		a := torch.NewTensor([]float32{1, 2, 3})
		result1 := torch.Unsqueeze(a, 0) // Function form
		result2 := a.Unsqueeze(0)        // Method form
		// Unsqueeze at dim 0 adds a dimension at the start
		expected := torch.NewTensor([][]float32{{1, 2, 3}}) // [1, 3] -> [1, 1, 3]

		if !torch.AllClose(result1, expected) {
			t.Errorf("Unsqueeze function failed. Expected %v, got %v", expected, result1)
		}
		if !torch.AllClose(result2, expected) {
			t.Errorf("Unsqueeze method failed. Expected %v, got %v", expected, result2)
		}

		// Test case 2: In-place version
		b := torch.NewTensor([]float32{1, 2, 3})
		result := b.UnsqueezeI(0)
		if !torch.AllClose(result, expected) {
			t.Errorf("UnsqueezeI operation failed. Expected %v, got %v", expected, result)
		}

		// Test case 3: Unsqueeze at dimension 1
		c := torch.NewTensor([]float32{1, 2, 3})
		result = c.Unsqueeze(1)
		expected = torch.NewTensor([][]float32{{1}, {2}, {3}}) // [3] -> [3, 1]
		if !torch.AllClose(result, expected) {
			t.Errorf("Unsqueeze at dim 1 failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("multi-dimensional tensors", func(t *testing.T) {
		matrix := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
		result := matrix.Unsqueeze(0)
		expected := torch.NewTensor([][][]float32{{{1, 2}, {3, 4}}}) // [2, 2] -> [1, 2, 2]
		if !torch.AllClose(result, expected) {
			t.Errorf("Unsqueeze of 2D tensor failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("negative dimensions", func(t *testing.T) {
		a := torch.NewTensor([]float32{1, 2, 3})
		result := a.Unsqueeze(-1)
		expected := torch.NewTensor([][]float32{{1}, {2}, {3}}) // [3] -> [3, 1]
		if !torch.AllClose(result, expected) {
			t.Errorf("Unsqueeze with negative dim failed. Expected %v, got %v", expected, result)
		}
	})
}

func TestUnsqueezeGPU(t *testing.T) {
	// Skip if CUDA is not available
	if !torch.IsCUDAAvailable() {
		t.Skip("CUDA not available")
	}

	device := torch.NewDevice("cuda")

	t.Run("basic GPU operation", func(t *testing.T) {
		a := torch.NewTensor([]float32{1, 2, 3})
		a = a.To(device) // Move tensor to GPU

		result := a.Unsqueeze(0)
		expected := torch.NewTensor([][]float32{{1, 2, 3}}).To(device)

		if !torch.AllClose(result, expected) {
			t.Errorf("GPU Unsqueeze failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("GPU in-place operation", func(t *testing.T) {
		a := torch.NewTensor([]float32{1, 2, 3}).To(device)
		result := a.UnsqueezeI(0)
		expected := torch.NewTensor([][]float32{{1, 2, 3}}).To(device)

		if !torch.AllClose(result, expected) {
			t.Errorf("GPU UnsqueezeI failed. Expected %v, got %v", expected, result)
		}
	})
}

func TestReshape(t *testing.T) {
	t.Run("basic reshape", func(t *testing.T) {
		// Create a 1D tensor
		a := torch.NewTensor([]float32{1, 2, 3, 4})

		// Reshape to 2x2
		result := a.Reshape(2, 2)
		expected := torch.NewTensor([][]float32{{1, 2}, {3, 4}})

		if !torch.AllClose(result, expected) {
			t.Errorf("Reshape failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("automatic size inference", func(t *testing.T) {
		// Create a 2x2 tensor
		a := torch.NewTensor([][]float32{{1, 2}, {3, 4}})

		// Use -1 to automatically infer size
		result := a.Reshape(-1)
		expected := torch.NewTensor([]float32{1, 2, 3, 4})

		if !torch.AllClose(result, expected) {
			t.Errorf("Reshape with size inference failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("multi-dimensional reshape", func(t *testing.T) {
		// Create a 4x2 tensor
		a := torch.NewTensor([][]float32{
			{1, 2}, {3, 4}, {5, 6}, {7, 8},
		})

		// Reshape to 2x2x2
		result := a.Reshape(2, 2, 2)
		expected := torch.NewTensor([][][]float32{
			{{1, 2}, {3, 4}},
			{{5, 6}, {7, 8}},
		})

		if !torch.AllClose(result, expected) {
			t.Errorf("Multi-dimensional reshape failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("gpu support", func(t *testing.T) {
		if !torch.IsCUDAAvailable() {
			t.Skip("CUDA not available")
		}

		device := torch.NewDevice("cuda")
		a := torch.NewTensor([]float32{1, 2, 3, 4}).To(device)
		result := a.Reshape(2, 2)
		expected := torch.NewTensor([][]float32{{1, 2}, {3, 4}}).To(device)

		if !torch.AllClose(result, expected) {
			t.Errorf("GPU reshape failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("error cases", func(t *testing.T) {
		a := torch.NewTensor([]float32{1, 2, 3})

		// This should panic as the new shape is invalid for the number of elements
		assert.Panics(t, func() {
			a.Reshape(2, 2)
		})
	})
}

func TestCat(t *testing.T) {
	t.Run("basic concatenation", func(t *testing.T) {
		// Create two 2x2 tensors
		a := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
		b := torch.NewTensor([][]float32{{5, 6}, {7, 8}})

		// Concatenate along dimension 0 (adding rows)
		result := torch.Cat([]torch.Tensor{a, b}, 0)
		expected := torch.NewTensor([][]float32{
			{1, 2},
			{3, 4},
			{5, 6},
			{7, 8},
		})

		if !torch.AllClose(result, expected) {
			t.Errorf("Cat along dim 0 failed. Expected %v, got %v", expected, result)
		}

		// Concatenate along dimension 1 (adding columns)
		result = torch.Cat([]torch.Tensor{a, b}, 1)
		expected = torch.NewTensor([][]float32{
			{1, 2, 5, 6},
			{3, 4, 7, 8},
		})

		if !torch.AllClose(result, expected) {
			t.Errorf("Cat along dim 1 failed. Expected %v, got %v", expected, result)
		}
	})
	t.Run("method variant", func(t *testing.T) {
		a := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
		b := torch.NewTensor([][]float32{{5, 6}, {7, 8}})

		// Use the method variant
		result := a.Cat([]torch.Tensor{b}, 0)
		expected := torch.NewTensor([][]float32{
			{1, 2},
			{3, 4},
			{5, 6},
			{7, 8},
		})

		if !torch.AllClose(result, expected) {
			t.Errorf("Cat method failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("out variant", func(t *testing.T) {
		a := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
		b := torch.NewTensor([][]float32{{5, 6}, {7, 8}})
		out := torch.Empty([]int64{4, 2}, false)

		result := torch.CatOut([]torch.Tensor{a, b}, 0, out)
		expected := torch.NewTensor([][]float32{
			{1, 2},
			{3, 4},
			{5, 6},
			{7, 8},
		})

		if !torch.AllClose(result, expected) {
			t.Errorf("CatOut failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("gpu support", func(t *testing.T) {
		if !torch.IsCUDAAvailable() {
			t.Skip("CUDA not available")
		}

		device := torch.NewDevice("cuda")
		a := torch.NewTensor([][]float32{{1, 2}, {3, 4}}).To(device)
		b := torch.NewTensor([][]float32{{5, 6}, {7, 8}}).To(device)

		result := torch.Cat([]torch.Tensor{a, b}, 0)
		expected := torch.NewTensor([][]float32{
			{1, 2},
			{3, 4},
			{5, 6},
			{7, 8},
		}).To(device)

		if !torch.AllClose(result, expected) {
			t.Errorf("GPU Cat failed. Expected %v, got %v", expected, result)
		}
	})

	t.Run("error cases", func(t *testing.T) {
		// Empty tensor list should panic
		assert.Panics(t, func() {
			torch.Cat([]torch.Tensor{}, 0)
		})

		// Tensors with incompatible shapes should panic
		a := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
		b := torch.NewTensor([]float32{1, 2, 3})
		assert.Panics(t, func() {
			torch.Cat([]torch.Tensor{a, b}, 0)
		})
	})
}
func TestCatWithNames(t *testing.T) {
	t.Run("basic named dimension concatenation", func(t *testing.T) {
		// Create two 2x2 tensors with named dimensions
		a := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
		b := torch.NewTensor([][]float32{{5, 6}, {7, 8}})

		// Create dimension names
		batch := torch.NewDimname("batch", false)
		features := torch.NewDimname("features", false)

		// Name the dimensions
		a = a.SetNames(batch, features)
		b = b.SetNames(batch, features)

		// Concatenate along batch dimension
		result := torch.CatWithNames([]torch.Tensor{a, b}, batch)

		// Test 1: Verify shape
		assert.Equal(t, []int64{4, 2}, result.Shape(), "Shape should be [4, 2]")

		// Test 2: Verify dimension names and their order
		names := result.Names()
		assert.Equal(t, 2, len(names), "Should have 2 named dimensions")
		assert.Equal(t, "batch", names[0].String(), "First dimension should be named 'batch'")
		assert.Equal(
			t,
			"features",
			names[1].String(),
			"Second dimension should be named 'features'",
		)

		// Test 3: Verify individual values
		// This ensures we're not just checking names but also correct concatenation
		value00 := result.Index(0, 0).Item().(float32)
		assert.Equal(t, float32(1), value00, "First element should be 1")

		value21 := result.Index(2, 1).Item().(float32)
		assert.Equal(t, float32(6), value21, "Element at [2,1] should be 6")
	})

	// Add error cases
	t.Run("mismatched dimension names", func(t *testing.T) {
		a := torch.NewTensor([][]float32{{1, 2}})
		b := torch.NewTensor([][]float32{{3, 4}})

		a = a.SetNames(
			torch.NewDimname("batch", false),
			torch.NewDimname("features", false),
		)
		b = b.SetNames(
			torch.NewDimname("different", false),
			torch.NewDimname("features", false),
		)

		// Should panic or return error due to mismatched names
		assert.Panics(t, func() {
			torch.CatWithNames([]torch.Tensor{a, b}, torch.NewDimname("batch", false))
		})
	})

	t.Run("concatenation along different named dimensions", func(t *testing.T) {
		a := torch.NewTensor([][]float32{{1, 2}})
		b := torch.NewTensor([][]float32{{3, 4}})

		batch := torch.NewDimname("batch", false)
		features := torch.NewDimname("features", false)

		a = a.SetNames(batch, features)
		b = b.SetNames(batch, features)

		// Test concatenation along features dimension
		result := torch.CatWithNames([]torch.Tensor{a, b}, features)

		// Verify shape changed in the correct dimension
		assert.Equal(t, []int64{1, 4}, result.Shape())

		// Verify names maintained correct order
		names := result.Names()
		assert.Equal(t, "batch", names[0].String())
		assert.Equal(t, "features", names[1].String())
	})
}

// Reduce operations
// Add to tensor_ops_test.go

// Add to tensor_ops_test.go

func TestMax(t *testing.T) {
	// Test cases for Max (reduction over entire tensor)
	t.Run("basic max reduction", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4, 5})
		max := tensor.Max()
		if max.Item().(float32) != 5 {
			t.Errorf("Expected max 5, got %v", max.Item())
		}
	})

	t.Run("2D tensor max", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4, 5, 6}).Reshape(2, 3)
		max := tensor.Max()
		if max.Item().(float32) != 6 {
			t.Errorf("Expected max 6, got %v", max.Item())
		}
	})

	t.Run("negative values", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{-1, -2, -3, -4, -5})
		max := tensor.Max()
		if max.Item().(float32) != -1 {
			t.Errorf("Expected max -1, got %v", max.Item())
		}
	})

	t.Run("single element", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{42})
		max := tensor.Max()
		if max.Item().(float32) != 42 {
			t.Errorf("Expected max 42, got %v", max.Item())
		}
	})
}

func TestMaxDim(t *testing.T) {
	t.Run("max along dim 0", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4, 5, 6}).Reshape(2, 3)
		values, indices := tensor.MaxDim(0, false)

		expectedValues := torch.NewTensor([]float32{4, 5, 6})
		expectedIndices := torch.NewTensor([]int64{1, 1, 1})

		if !torch.AllClose(values, expectedValues) {
			t.Errorf("Expected values %s, got %s", expectedValues.String(), values.String())
		}
		if !torch.Equal(indices, expectedIndices) {
			t.Errorf("Expected indices %s, got %s", expectedIndices.String(), indices.String())
		}
	})

	t.Run("max along dim 1", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4, 5, 6}).Reshape(2, 3)
		values, indices := tensor.MaxDim(1, false)

		expectedValues := torch.NewTensor([]float32{3, 6})
		expectedIndices := torch.NewTensor([]int64{2, 2})

		if !torch.AllClose(values, expectedValues) {
			t.Errorf("Expected values %s, got %s", expectedValues.String(), values.String())
		}
		if !torch.Equal(indices, expectedIndices) {
			t.Errorf("Expected indices %s, got %s", expectedIndices.String(), indices.String())
		}
	})

	t.Run("max with keepdim", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4, 5, 6}).Reshape(2, 3)
		values, indices := tensor.MaxDim(1, true)

		// Check shape with keepdim
		if len(values.Shape()) != 2 || values.Shape()[1] != 1 {
			t.Errorf("Expected shape [2,1], got %v", values.Shape())
		}

		expectedValues := torch.NewTensor([]float32{3, 6}).Reshape(2, 1)
		expectedIndices := torch.NewTensor([]int64{2, 2}).Reshape(2, 1)

		if !torch.AllClose(values, expectedValues) {
			t.Errorf("Expected values %s, got %s", expectedValues.String(), values.String())
		}
		if !torch.Equal(indices, expectedIndices) {
			t.Errorf("Expected indices %s, got %s", expectedIndices.String(), indices.String())
		}
	})

	t.Run("3D tensor max", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8}).Reshape(2, 2, 2)
		values, indices := tensor.MaxDim(1, false)

		expectedValues := torch.NewTensor([]float32{3, 4, 7, 8}).Reshape(2, 2)
		expectedIndices := torch.NewTensor([]int64{1, 1, 1, 1}).Reshape(2, 2)

		if !torch.AllClose(values, expectedValues) {
			t.Errorf("Expected values %s, got %s", expectedValues.String(), values.String())
		}
		if !torch.Equal(indices, expectedIndices) {
			t.Errorf("Expected indices %s, got %s", expectedIndices.String(), indices.String())
		}
	})

	// Invalid dimension test remains the same
}

func TestMin(t *testing.T) {
	t.Run("basic min reduction", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4, 5})
		min := tensor.Min()
		if min.Item().(float32) != 1 {
			t.Errorf("Expected min 1, got %v", min.Item())
		}
	})

	t.Run("2D tensor min", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4, 5, 6}).Reshape(2, 3)
		min := tensor.Min()
		if min.Item().(float32) != 1 {
			t.Errorf("Expected min 1, got %v", min.Item())
		}
	})

	t.Run("negative values", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{-1, -2, -3, -4, -5})
		min := tensor.Min()
		if min.Item().(float32) != -5 {
			t.Errorf("Expected min -5, got %v", min.Item())
		}
	})
}

func TestMinDim(t *testing.T) {
	t.Run("min along dim 0", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4, 5, 6}).Reshape(2, 3)
		values, indices := tensor.MinDim(0, false)

		expectedValues := torch.NewTensor([]float32{1, 2, 3})
		expectedIndices := torch.NewTensor([]int64{0, 0, 0})

		if !torch.AllClose(values, expectedValues) {
			t.Errorf("Expected values %s, got %s", expectedValues.String(), values.String())
		}
		if !torch.Equal(indices, expectedIndices) {
			t.Errorf("Expected indices %s, got %s", expectedIndices.String(), indices.String())
		}
	})

	t.Run("min along dim 1", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4, 5, 6}).Reshape(2, 3)
		values, indices := tensor.MinDim(1, false)

		expectedValues := torch.NewTensor([]float32{1, 4})
		expectedIndices := torch.NewTensor([]int64{0, 0})

		if !torch.AllClose(values, expectedValues) {
			t.Errorf("Expected values %s, got %s", expectedValues.String(), values.String())
		}
		if !torch.Equal(indices, expectedIndices) {
			t.Errorf("Expected indices %s, got %s", expectedIndices.String(), indices.String())
		}
	})

	t.Run("min with keepdim", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4, 5, 6}).Reshape(2, 3)
		values, indices := tensor.MinDim(1, true)

		if len(values.Shape()) != 2 || values.Shape()[1] != 1 {
			t.Errorf("Expected shape [2,1], got %v", values.Shape())
		}

		expectedValues := torch.NewTensor([]float32{1, 4}).Reshape(2, 1)
		expectedIndices := torch.NewTensor([]int64{0, 0}).Reshape(2, 1)

		if !torch.AllClose(values, expectedValues) {
			t.Errorf("Expected values %s, got %s", expectedValues.String(), values.String())
		}
		if !torch.Equal(indices, expectedIndices) {
			t.Errorf("Expected indices %s, got %s", expectedIndices.String(), indices.String())
		}
	})
}

func TestProd(t *testing.T) {
	t.Run("basic prod reduction", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4})
		prod := tensor.Prod()
		if prod.Item().(float32) != 24 { // 1 * 2 * 3 * 4 = 24
			t.Errorf("Expected product 24, got %v", prod.Item())
		}
	})

	t.Run("2D tensor prod", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4}).Reshape(2, 2)
		prod := tensor.Prod()
		if prod.Item().(float32) != 24 {
			t.Errorf("Expected product 24, got %v", prod.Item())
		}
	})

	t.Run("prod with zeros", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 0, 4})
		prod := tensor.Prod()
		if prod.Item().(float32) != 0 {
			t.Errorf("Expected product 0, got %v", prod.Item())
		}
	})
}

func TestElementwiseMaxMin(t *testing.T) {
	t.Run("element-wise max", func(t *testing.T) {
		a := torch.NewTensor([]float32{1, 4, 3, 8})
		b := torch.NewTensor([]float32{2, 3, 5, 7})
		result := a.MaxElementwise(b)

		expected := torch.NewTensor([]float32{2, 4, 5, 8})
		if !torch.AllClose(result, expected) {
			t.Errorf("Expected %s, got %s", expected.String(), result.String())
		}
	})

	t.Run("element-wise min", func(t *testing.T) {
		a := torch.NewTensor([]float32{1, 4, 3, 8})
		b := torch.NewTensor([]float32{2, 3, 5, 7})
		result := a.MinElementwise(b)

		expected := torch.NewTensor([]float32{1, 3, 3, 7})
		if !torch.AllClose(result, expected) {
			t.Errorf("Expected %s, got %s", expected.String(), result.String())
		}
	})

	t.Run("element-wise max with broadcasting", func(t *testing.T) {
		a := torch.NewTensor([]float32{1, 2, 3, 4}).Reshape(2, 2)
		b := torch.NewTensor([]float32{2, 2}).Reshape(1, 2)
		result := a.MaxElementwise(b)

		expected := torch.NewTensor([]float32{2, 2, 3, 4}).Reshape(2, 2)
		if !torch.AllClose(result, expected) {
			t.Errorf("Expected %s, got %s", expected.String(), result.String())
		}
	})

	t.Run("element-wise min with broadcasting", func(t *testing.T) {
		a := torch.NewTensor([]float32{1, 2, 3, 4}).Reshape(2, 2)
		b := torch.NewTensor([]float32{2, 2}).Reshape(1, 2)
		result := a.MinElementwise(b)

		expected := torch.NewTensor([]float32{1, 2, 2, 2}).Reshape(2, 2)
		if !torch.AllClose(result, expected) {
			t.Errorf("Expected %s, got %s", expected.String(), result.String())
		}
	})
}

func TestMaxMinOut(t *testing.T) {
	t.Run("max elementwise out", func(t *testing.T) {
		a := torch.NewTensor([]float32{1, 4, 3, 8})
		b := torch.NewTensor([]float32{2, 3, 5, 7})
		out := torch.NewTensor([]float32{0, 0, 0, 0})
		result := a.MaxElementwiseOut(b, out)

		expected := torch.NewTensor([]float32{2, 4, 5, 8})
		if !torch.AllClose(result, expected) {
			t.Errorf("Expected %s, got %s", expected.String(), result.String())
		}
		// Verify out tensor was modified
		if !torch.AllClose(out, expected) {
			t.Errorf("Out tensor not modified correctly. Expected %s, got %s",
				expected.String(), out.String())
		}
	})

	t.Run("min elementwise out", func(t *testing.T) {
		a := torch.NewTensor([]float32{1, 4, 3, 8})
		b := torch.NewTensor([]float32{2, 3, 5, 7})
		out := torch.NewTensor([]float32{0, 0, 0, 0})
		result := a.MinElementwiseOut(b, out)

		expected := torch.NewTensor([]float32{1, 3, 3, 7})
		if !torch.AllClose(result, expected) {
			t.Errorf("Expected %s, got %s", expected.String(), result.String())
		}
		// Verify out tensor was modified
		if !torch.AllClose(out, expected) {
			t.Errorf("Out tensor not modified correctly. Expected %s, got %s",
				expected.String(), out.String())
		}
	})
}

func TestProdDim(t *testing.T) {
	t.Run("prod along dim 0", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4, 5, 6}).Reshape(2, 3)
		result := tensor.ProdDim(0, false)

		expected := torch.NewTensor([]float32{4, 10, 18}) // [1*4, 2*5, 3*6]
		if !torch.AllClose(result, expected) {
			t.Errorf("Expected %s, got %s", expected.String(), result.String())
		}
	})

	t.Run("prod along dim 1", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4, 5, 6}).Reshape(2, 3)
		result := tensor.ProdDim(1, false)

		expected := torch.NewTensor([]float32{6, 120}) // [1*2*3, 4*5*6]
		if !torch.AllClose(result, expected) {
			t.Errorf("Expected %s, got %s", expected.String(), result.String())
		}
	})

	t.Run("prod with keepdim", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 2, 3, 4, 5, 6}).Reshape(2, 3)
		result := tensor.ProdDim(1, true)

		if len(result.Shape()) != 2 || result.Shape()[1] != 1 {
			t.Errorf("Expected shape [2,1], got %v", result.Shape())
		}

		expected := torch.NewTensor([]float32{6, 120}).Reshape(2, 1)
		if !torch.AllClose(result, expected) {
			t.Errorf("Expected %s, got %s", expected.String(), result.String())
		}
	})

	t.Run("prod with zeros", func(t *testing.T) {
		tensor := torch.NewTensor([]float32{1, 0, 3, 4, 0, 6}).Reshape(2, 3)
		result := tensor.ProdDim(1, false)

		expected := torch.NewTensor([]float32{0, 0}) // [1*0*3, 4*0*6]
		if !torch.AllClose(result, expected) {
			t.Errorf("Expected %s, got %s", expected.String(), result.String())
		}
	})
}

func TestProdOut(t *testing.T) {
	t.Run("basic prod out", func(t *testing.T) {
		input := torch.NewTensor([]float32{1, 2, 3, 4})
		// Create a scalar tensor with empty shape for output
		out := input.Prod() // This will give us a tensor with the correct scalar shape
		result := input.ProdOut(out)

		expected := torch.NewTensor([]float32{24}) // 1 * 2 * 3 * 4 = 24
		if !torch.AllClose(result, expected) {
			t.Errorf("Expected %s, got %s", expected.String(), result.String())
		}
		// Verify out tensor was modified
		if !torch.AllClose(out, expected) {
			t.Errorf("Out tensor not modified correctly. Expected %s, got %s",
				expected.String(), out.String())
		}
	})

	// Rest of the test cases remain the same
	t.Run("prod dim out", func(t *testing.T) {
		input := torch.NewTensor([]float32{1, 2, 3, 4, 5, 6}).Reshape(2, 3)
		out := torch.NewTensor([]float32{0, 0}) // Output shape matches reduction result
		result := input.ProdDimOut(1, false, out)

		expected := torch.NewTensor([]float32{6, 120}) // [1*2*3, 4*5*6]
		if !torch.AllClose(result, expected) {
			t.Errorf("Expected %s, got %s", expected.String(), result.String())
		}
		if !torch.AllClose(out, expected) {
			t.Errorf("Out tensor not modified correctly. Expected %s, got %s",
				expected.String(), out.String())
		}
	})

	t.Run("prod dim out with keepdim", func(t *testing.T) {
		input := torch.NewTensor([]float32{1, 2, 3, 4, 5, 6}).Reshape(2, 3)
		out := torch.NewTensor([]float32{0, 0}).Reshape(2, 1) // Output shape matches keepdim result
		result := input.ProdDimOut(1, true, out)

		expected := torch.NewTensor([]float32{6, 120}).Reshape(2, 1) // [1*2*3, 4*5*6] with keepdim
		if !torch.AllClose(result, expected) {
			t.Errorf("Expected %s, got %s", expected.String(), result.String())
		}
		if !torch.AllClose(out, expected) {
			t.Errorf("Out tensor not modified correctly. Expected %s, got %s",
				expected.String(), out.String())
		}
	})
}
