package main

import (
	"fmt"
	"strings"
)

func main() {
	path := "/home//foo/"

	split := strings.Split(path, "/")

	fmt.Println(split, len(split))

	fmt.Println(split[0] == "")
}
