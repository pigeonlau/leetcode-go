package main

import (
	"fmt"
	"math"
)

func options(op []byte, a []int, prevAdd int, sum int, index int) {
	if index >= len(op) {
		if sum == 100 {
			fmt.Print(a[0])
			for i := 1; i < len(op); i++ {
				if op[i] != ' ' {
					fmt.Print(string(op[i]), a[i])
				} else {
					fmt.Print(a[i])
				}
			}
			fmt.Println("=100")
		}

		return
	}

	op[index] = '+'
	sum += a[index]
	options(op, a, a[index], sum, index+1)
	sum -= a[index]

	op[index] = '-'
	sum -= a[index]
	options(op, a, -a[index], sum, index+1)
	sum += a[index]

	op[index] = ' '
	sum -= prevAdd
	temp := 0
	if prevAdd > 0 {
		temp = prevAdd*10 + a[index]
	} else {
		temp = prevAdd*10 - a[index]
	}
	sum += temp
	options(op, a, temp, sum, index+1)
	sum -= temp
	sum += prevAdd
}

func integerBreak(n int) int {

	if n == 1 || n == 2 {
		return 1
	}
	dp := make([]int, n+1)
	dp[1] = 1
	dp[2] = 1

	for i := 3; i <= n; i++ {
		temp := 0
		for j := 1; j <= i/2; j++ {
			temp = max(temp, max(dp[i], i)*max(dp[i-j], i-j))
		}
		dp[i] = temp
	}

	return dp[n]
}

func max(a, b int) int {
	if a < b {
		return b
	}
	return a
}

func integerBreakToMaxK(n, k int) int {

	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, k+1)
	}
	for i := 1; i < k+1; i++ {
		dp[1][i] = 1
	}

	for i := 2; i <= n; i++ {
		for j := 1; j <= k; j++ {
			if j > i {
				dp[i][j] = dp[i][j-1]
				continue
			}
			dp[i][j] = dp[i][j-1] + dp[i-j][j]
			if i == j {
				dp[i][j]++
			}
		}
	}

	return dp[n][k]

}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func trianglePath(cost [][]int) int {
	n := len(cost)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}

	dp[1][1] = cost[0][0]

	for i := 2; i <= n; i++ {
		for j := 1; j <= i; j++ {
			if j == 1 {
				dp[i][j] = cost[i-1][j-1] + dp[i-1][j]
				continue
			} else if j == i {
				dp[i][j] = cost[i-1][j-1] + dp[i-1][j-1]
				continue
			}
			dp[i][j] = cost[i-1][j-1] + min(dp[i-1][j-1], dp[i-1][j])
		}
	}

	res := math.MaxInt32
	for i := 1; i <= n; i++ {
		res = min(res, dp[n][i])
	}

	return res
}

func main() {

	//println(solution.ResourceAllocate([][]int{{0, 3, 7, 9, 12, 13}, {0, 5, 10, 11, 11, 11}, {0, 4, 6, 11, 12, 12}}, 5))

	//println(solution.MeetingAllocate([][]int{{1, 4}, {3, 5}, {0, 6}, {5, 7}, {3, 8}, {5, 9}, {6, 10}, {8, 11}, {8, 12}, {2, 13}, {12, 15}}))

	//solution.CombinationalSum(nil, 0)

	//backTracking.Knight(5, 5)

	//backTracking.FindPathsFromCorner(nil)

	//fmt.Println(len(strings.Split("leetcode exercises sound delightfu", " ")))

	var a int = 10
	for i := 1; i <= 100; i++ {
		a = a * 10
	}
	a = int(1e100)
	fmt.Println(a)
}
