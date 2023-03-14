package dmsxl

import (
	"math"
)

func sortedSquares(nums []int) []int {

	res := make([]int, len(nums))

	left := 0
	right := len(nums) - 1

	i := right
	for left <= right {
		if absCompare(nums[left], nums[right]) {
			res[i] = nums[left] * nums[left]
			left++
		} else {
			res[i] = nums[right] * nums[right]
			right--
		}
		i--
	}

	return res
}

func absCompare(a, b int) bool {
	if a < 0 {
		a = -a
	}
	if b < 0 {
		b = -b
	}

	return a > b
}

func minSubArrayLen(target int, nums []int) int {
	left := 0
	right := 0
	sum := nums[0]

	length := math.MaxInt
	for right < len(nums) {
		if sum >= target {
			length = min(length, right-left+1)
			sum -= nums[left]
			left++
		} else {
			if right == len(nums)-1 {
				break
			} else {
				right++
				sum += nums[right]
			}
		}
	}
	if length == math.MaxInt {
		return 0
	}
	return length
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func minSubArrayLen2(target int, nums []int) int {

	sum := 0
	length := math.MaxInt

	left := 0

	for i := range nums {
		sum += nums[i]

		for sum >= target {
			length = min(length, i-left+1)
			sum -= nums[left]
			left++
		}
	}

	if length == math.MaxInt {
		return 0
	}
	return length

}

func generateMatrix(n int) [][]int {

	direction := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	dire := 0

	res := make([][]int, n)
	for i := range res {
		res[i] = make([]int, n)
	}

	x := 0
	y := -1
	for i := 1; i <= n*n; i++ {
		nx := x + direction[dire][0]
		ny := y + direction[dire][1]
		if nx < n && nx >= 0 && ny < n && ny >= 0 && res[nx][ny] == 0 {
			res[nx][ny] = i
			x = nx
			y = ny
		} else {
			dire = (dire + 1) % 4
			i--
		}
	}

	return res
}

func spiralOrder(matrix [][]int) []int {
	//direction := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	//dire := 0

	m := len(matrix)
	n := len(matrix[0])

	up, down, left, right := -1, m, -1, n

	startX := 0
	startY := -17

	res := make([]int, m*n)
	i := 0
	for i < m*n {
		x := startX
		y := startY
		startX++
		startY++
		for i < m*n && y < right-1 {
			y++
			res[i] = matrix[x][y]
			i++
		}
		up++

		for i < m*n && x < down-1 {
			x++
			res[i] = matrix[x][y]
			i++

		}
		right--

		for i < m*n && y > left+1 {
			y--
			res[i] = matrix[x][y]
			i++

		}
		down--

		for i < m*n && x > up+1 {
			x--
			res[i] = matrix[x][y]
			i++

		}
		left++
	}

	return res

}
