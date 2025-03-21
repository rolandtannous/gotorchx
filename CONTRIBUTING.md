# How to Contribute

## Build and Test

Run the following command to retrieve the source code into the directory
`$GOPATH/src/github.com/rolandtannous/gotorchx`.

```bash
go get github.com/rolandtannous/gotorchx
```

Feel free to set the environment variable `GOPATH` to an arbitrary directory
`export GOPATH=some/directory`.

Build the Cgo binding of `libtorch`.

```bash
$GOPATH/src/github.com/rolandtannous/gotorchx/cgotorch/build.sh
```

Run the Go examples and unit tests.

```bash
cd $GOPATH/src/github.com/rolandtannous/gotorchx
go test -v github.com/rolandtannous/gotorchx/...
```

The above `go test` command might fail and complain that it cannot find the
`.so` or `.dylib` files.  If so, please run the following command.

```bash
export CGOTORCH=$GOPATH/src/github.com/rolandtannous/gotorchx/cgotorch/libtorch/lib
export LD_LIBRARY_PATH=$CGOTORCH:$LD_LIBRARY_PATH
```
