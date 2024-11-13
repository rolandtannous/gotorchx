package gotorch

// #include <stdlib.h>
// #include "cgotorch/cgotorch.h"
import "C"
import (
	"runtime"
	"unsafe"
)

type DimnameType int

const (
	Basic DimnameType = iota
	Wildcard
)

// Dimname represents a dimension name
type Dimname struct {
	dimname C.Dimname // Change from *unsafe.Pointer to C.Dimname
}

// NewDimname creates a new dimension name
func NewDimname(name string, isWildcard bool) Dimname {
	var d C.Dimname
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	MustNil(unsafe.Pointer(C.NewDimname(cname, C.bool(isWildcard), &d)))

	dimname := Dimname{d}
	runtime.SetFinalizer(&dimname, func(d *Dimname) {
		C.FreeDimname(d.dimname)
	})
	return dimname
}

// String returns the string representation of the dimension name
func (d Dimname) String() string {
	var result *C.char
	MustNil(unsafe.Pointer(C.DimnameToString(d.dimname, &result)))
	defer C.free(unsafe.Pointer(result))
	return C.GoString(result)
}

// Add methods to Tensor for named dimensions support
func (t Tensor) SetNames(names ...Dimname) Tensor {
	if len(names) == 0 {
		return t
	}

	cnames := make([]C.Dimname, len(names))
	for i, name := range names {
		cnames[i] = name.dimname
	}

	MustNil(unsafe.Pointer(C.SetTensorDimnames(
		C.Tensor(*t.T),
		(*C.Dimname)(unsafe.Pointer(&cnames[0])),
		C.int64_t(len(names)),
	)))
	return t
}

func (t Tensor) Names() []Dimname {
	var names *C.Dimname
	var numNames C.int64_t

	MustNil(unsafe.Pointer(C.GetTensorDimnames(
		C.Tensor(*t.T),
		&names,
		&numNames,
	)))

	result := make([]Dimname, int(numNames))
	namesSlice := unsafe.Slice(names, int(numNames))

	for i := 0; i < int(numNames); i++ {
		result[i] = Dimname{namesSlice[i]}
	}

	C.free(unsafe.Pointer(names))
	return result
}
