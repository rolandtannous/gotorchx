// gotorchx: Go bindings for the PyTorch deep learning framework
// Previously known as github.com/wangkuiyi/gotorch
module github.com/rolandtannous/gotorchx

// Using Go 1.23 for modern language features
go 1.23

require (
	// Testing framework for assertions and mocks
	github.com/stretchr/testify v1.6.1

	// IEEE 754 half-precision floating-point (binary16) implementation
	github.com/x448/float16 v0.8.4

	// Go package for computer vision using OpenCV
	gocv.io/x/gocv v0.24.0
)

require (
	// Indirect dependencies required by testify
	github.com/davecgh/go-spew v1.1.0 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	gopkg.in/yaml.v3 v3.0.0-20200313102051-9f266ea9e77c // indirect
)

// Local development configuration
// Points to local directory for development instead of GitHub repository
replace github.com/rolandtannous/gotorchx => /home/mlengineer/gotorch
