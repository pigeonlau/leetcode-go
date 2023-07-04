package backTracking

import "fmt"

var (
	tempRes []int
	tempSum int
)

func CombinationalSum(arr []int, target int) {
	backTracking([]int{2, 4, 6, 8}, 0, 8)
}

func backTracking(arr []int, index int, target int) {
	n := len(arr)

	if target == tempSum {
		fmt.Println(tempRes)
		return
	}
	if index >= n {
		return
	}

	if target < tempSum {
		return
	}

	backTracking(arr, index+1, target)
	tempSum += arr[index]
	tempRes = append(tempRes, arr[index])
	backTracking(arr, index, target)
	tempSum -= arr[index]
	tempRes = tempRes[:len(tempRes)-1]

}

var (
	row = []int{1, 2, 2, 1, -1, -2, -2, -1}
	col = []int{2, 1, -1, -2, -2, -1, 1, 2}
)

func checkValidGrid(grid [][]int) bool {
	return backTrackingKnight(grid, 0, 0, 0)
}

func backTrackingKnight(grid [][]int, x, y, current int) bool {
	if grid[x][y] != current {
		return false
	}
	if grid[x][y] == len(grid)*len(grid[0])-1 {
		return true
	}

	for i := range row {
		xx := x + row[i]
		yy := y + col[i]
		if xx >= 0 && xx < len(grid) && yy >= 0 && yy < len(grid[0]) && grid[xx][yy] == current+1 {
			return backTrackingKnight(grid, xx, yy, current+1)
		}
	}

	return false

}

func Knight(m, n int) {
	grid := make([][]int, m)
	for i := range grid {
		grid[i] = make([]int, n)
	}
	backTrackingKnightFullPath(grid, 0, 0, 1)
}

func backTrackingKnightFullPath(grid [][]int, x, y int, current int) {
	grid[x][y] = current
	if current == len(grid[0])*len(grid) {
		fmt.Println(grid)
		return
	}
	for i := range row {
		xx := x + row[i]
		yy := y + col[i]
		if xx >= 0 && xx < len(grid) && yy >= 0 && yy < len(grid[0]) && grid[xx][yy] == 0 {
			backTrackingKnightFullPath(grid, xx, yy, current+1)
		}
	}
	grid[x][y] = 0

}

var (
	visited [][]bool
	path    [][]int

	r = []int{0, 1, 0, -1}
	c = []int{1, 0, -1, 0}
)

func FindPathsFromCorner(grid [][]int) {
	visited = make([][]bool, 9)
	for i := range visited {
		visited[i] = make([]bool, 9)
	}

	grid = [][]int{
		{3, 5, 4, 4, 7, 3, 4, 6, 3},
		{6, 7, 5, 6, 6, 2, 6, 6, 2},
		{3, 3, 4, 3, 2, 5, 4, 7, 2},
		{6, 5, 5, 1, 2, 3, 6, 5, 6},
		{3, 3, 4, 3, 0, 1, 4, 3, 4},
		{3, 5, 4, 3, 2, 2, 3, 3, 5},
		{3, 5, 4, 3, 2, 6, 4, 4, 3},
		{3, 5, 1, 3, 7, 5, 3, 6, 4},
		{6, 2, 4, 3, 4, 5, 4, 5, 1}}

	backTrackingFindPaths(grid, 4, 4, 0, 0)
}

func backTrackingFindPaths(grid [][]int, targetX, targetY int, x, y int) {
	visited[x][y] = true
	path = append(path, []int{x, y})
	if x == targetX && y == targetY {
		fmt.Println(path)
		return
	}

	for i := range r {
		xx := x + grid[x][y]*r[i]
		yy := y + grid[x][y]*c[i]

		if xx >= 0 && xx < len(grid) && yy >= 0 && yy < len(grid[0]) && visited[xx][yy] == false {
			backTrackingFindPaths(grid, targetX, targetY, xx, yy)
		}
	}

	path = path[:len(path)-1]
}

func Rat() {

	grid := [][]int{{2, 1, 0, 0},
		{3, 0, 0, 1},
		{0, 1, 0, 1},
		{0, 0, 0, 1}}

	visited = make([][]bool, 4)
	for i := range visited {
		visited[i] = make([]bool, 4)
	}

	backTrackingRat(grid, 0, 0, len(grid)-1, len(grid[0])-1)
}

func backTrackingRat(grid [][]int, x, y int, targetX, targetY int) {
	if x == targetX && y == targetY {
		fmt.Println(visited)
		return
	}
	n := grid[x][y]
	visited[x][y] = true
	for i := 1; i <= n; i++ {
		if x+i < len(grid) {
			backTrackingRat(grid, x+i, y, targetX, targetY)
		}
		if y+i < len(grid[0]) {
			backTrackingRat(grid, x, y+i, targetX, targetY)
		}
	}

	visited[x][y] = false
}

func findPeakElement(nums []int) int {
	n := len(nums)
	if n == 1 || nums[0] > nums[1] {
		return 0
	}
	if nums[n-1] > nums[n-2] {
		return n - 1
	}

	left := 1
	right := n - 1

	for left < right {
		mid := left + (right-left)/2
		if nums[mid] > nums[mid-1] && nums[mid] > nums[mid+1] {
			return mid
		} else if nums[mid] > nums[mid+1] {
			right = mid
		} else {
			left = mid + 1
		}
	}

	return -1
}
