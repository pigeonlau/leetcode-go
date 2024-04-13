package solution

import (
	"container/heap"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a < b {
		return b
	}
	return a
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// leetcode-2383
// 赢得比赛需要的最少训练时长
func minNumberOfHours(initialEnergy int, initialExperience int, energy []int, experience []int) int {

	ene := initialEnergy
	exp := initialExperience

	res := 0
	for i := range energy {
		if ene <= energy[i] {
			res += energy[i] + 1 - ene
			ene = 1
		} else {
			ene = ene - energy[i]
		}

		if exp <= experience[i] {
			res += experience[i] - exp + 1
			exp = experience[i] + 1
		}
		exp += experience[i]

	}

	return res
}

// leetcode - mst 17.05
// 字母与数字
func findLongestSubarray(array []string) []string {

	m := map[int]int{0: len(array)}

	numCount, letterCount := 0, 0

	maxCount := 0
	index := len(array)
	for i := len(array) - 1; i >= 0; i-- {
		str := array[i]
		if str[0] >= '0' && str[0] <= '9' {
			numCount++
		} else {
			letterCount++
		}

		temp := numCount - letterCount
		if ind, ok := m[temp]; ok {
			if maxCount <= ind-i {
				maxCount = ind - i
				index = i
			}
		} else {
			m[temp] = i
		}
	}
	return array[index : index+maxCount]

}

// leetcode - 2395
// 和相等的子数组
func findSubarrays(nums []int) bool {

	m := map[int]bool{}

	for i := 0; i < len(nums)-1; i++ {
		temp := nums[i] + nums[i+1]
		if m[temp] {
			return true
		} else {
			m[temp] = true
		}
	}

	return false
}

// leetcode 2367
// 算术三元组的数目
func arithmeticTriplets(nums []int, diff int) int {

	m := map[int]bool{}

	for _, num := range nums {
		m[num] = true
	}

	res := 0
	for i := 0; i < len(nums)-2; i++ {
		num := nums[i]
		if m[num+diff] && m[num+diff+diff] {
			res++
		}
	}

	return res
}

// leetcode  1015
// 可被K整除的最小整数
func smallestRepunitDivByK(k int) int {

	length := 1
	set := map[int]bool{}

	mod := 1 % k

	for true {
		if mod == 0 {
			return length
		} else if set[mod] {
			return -1
		} else {
			set[mod] = true
			mod = (mod*10 + 1) % k
			length++
		}
	}
	return -1
}

// leetcode 1016
// 子串能表示从1到N数字的二进制串
func queryString(s string, n int) bool {

	length := len(s)
	maxWindowSize := min(30, length)

	bytes := []byte(s)

	set := map[int]bool{}

	for i := 1; i <= maxWindowSize; i++ {
		for j := 0; j <= length-i; j++ {
			num := bytesToInt(bytes[j : j+i])
			set[num] = true
		}
	}

	for i := 1; i < n; i++ {
		if !set[i] {
			return false
		}
	}
	return true

}

func bytesToInt(bytes []byte) int {
	n := len(bytes)
	res := 0

	for i := n - 1; i >= 0; i-- {
		if bytes[i] == '1' {
			res += 1 << (n - i - 1)
		}
	}

	fmt.Println(n, res)
	return res
}

// leetcode 101
// 总持续时间可以被60整除的歌曲
func numPairsDivisibleBy60(time []int) int {
	m := map[int]int{}

	ans := 0
	for i := 0; i < len(time); i++ {
		m[time[i]%60]++
	}

	for t, sum := range m {
		pairTime := (60 - t) % 60

		if t == pairTime {
			ans += sum * (sum - 1) / 2
		} else if pairTime < t {
			continue
		} else {
			ans += sum * m[pairTime]
		}
	}

	return ans
}

// leetcode 1423
// 可获得的最大点数
func maxScore(cardPoints []int, k int) int {

	n := len(cardPoints)
	prefixSum := make([]int, n+1)
	suffixSum := make([]int, n+1)

	prefixSum[0] = 0
	suffixSum[0] = 0
	for i := 0; i < n; i++ {

		prefixSum[i+1] = prefixSum[i] + cardPoints[i]
		suffixSum[i+1] = suffixSum[i] + cardPoints[n-1-i]

	}

	ans := 0
	for i := 0; i <= k; i++ {
		ans = max(ans, suffixSum[i]+prefixSum[k-i])
	}

	return ans

}

// leetcode 56
// 合并区间
func merge(intervals [][]int) [][]int {

	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})

	left := intervals[0][0]
	right := intervals[0][1]

	res := make([][]int, 0)
	for i := 1; i < len(intervals); i++ {
		if right < intervals[i][0] {
			res = append(res, []int{left, right})
			left = intervals[i][0]
			right = intervals[i][1]
		} else if right >= intervals[i][1] {
			continue
		} else {
			right = intervals[i][1]
		}
	}

	res = append(res, []int{left, right})

	return res

}

func insert(intervals [][]int, newInterval []int) [][]int {

	intervals = append(intervals, newInterval)

	return merge(intervals)
}

// leetcode 71
// 简化路径
func simplifyPath(path string) string {

	split := strings.Split(path, "/")

	dir := make([]string, 0)

	for _, s := range split {
		if s == ".." {
			if len(dir) > 0 {
				dir = dir[0 : len(dir)-1]
			}
		} else if s == "." || s == "" {
			continue
		} else {
			dir = append(dir, s)
		}
	}

	return "/" + strings.Join(dir, "/")

}

func setZeroes(matrix [][]int) {

	row := make(map[int]bool)
	col := make(map[int]bool)

	for i := range matrix {
		for j := range matrix[i] {
			if matrix[i][j] == 0 {
				row[i] = true
				col[j] = true
			}
		}
	}

	for r := range row {
		for j := range matrix[r] {
			matrix[r][j] = 0
		}
	}

	for c := range col {
		for i := range matrix {
			matrix[i][c] = 0
		}
	}

}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func sufficientSubset(root *TreeNode, limit int) *TreeNode {

	if root == nil {
		return nil
	}

	sssssss(root, limit, 0)

	if root.Left == nil && root.Right == nil && root.Val < limit {
		return nil
	}

	return root
}

func sssssss(root *TreeNode, limit int, sum int) (subSum int) {
	if root == nil {
		return 0
	}

	left := sssssss(root.Left, limit, sum+root.Val)
	right := sssssss(root.Right, limit, sum+root.Val)

	subSum += root.Val
	if root.Left != nil && root.Right != nil {
		subSum += max(left, right)
	} else if root.Left != nil {
		subSum += left
	} else {
		subSum += right
	}

	if left+sum+root.Val < limit {
		root.Left = nil
	}

	if right+sum+root.Val < limit {
		root.Right = nil
	}

	return
}

//func permute(nums []int) [][]int {
//
//	n := len(nums)
//	res := make([][]int, 0, n+1)
//
//	var dfs func(nums []int, start int)
//
//	dfs = func(nums []int, start int) {
//		if start == len(nums) {
//			ans := make([]int, n)
//			copy(ans, nums)
//			res = append(res, ans)
//			return
//		}
//
//		for i := start; i < n; i++ {
//			nums[i], nums[start] = nums[start], nums[i]
//			dfs(nums, start+1)
//			nums[i], nums[start] = nums[start], nums[i]
//		}
//	}
//
//	dfs(nums, 0)
//
//	return res
//
//}

func oddString(words []string) string {
	lowString := low(words[0])

	for i := 1; i < len(words); i++ {
		l := low(words[i])
		if i == 1 && l != lowString {
			if lowString == low(words[2]) {
				return words[i]
			} else {
				return words[0]
			}
		}

		if l != lowString {
			return words[i]
		}
	}

	return ""
}

func low(word string) string {
	bytes := []byte(word)
	res := make([]byte, len(bytes))

	minus := bytes[0] - 'a'
	for i := range bytes {
		res[i] = bytes[i] - minus
	}

	return string(res)
}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func inorderTraversal(root *TreeNode) []int {

	stack := make([]*TreeNode, 0)
	res := make([]int, 0)

	for root != nil || len(stack) > 0 {

		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		root = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(res, root.Val)
		root = root.Right
	}

	return res
}

func inorderTraversal2(root *TreeNode) []int {

	res := make([]int, 0)

	for root != nil {
		if root.Left != nil {
			pre := root.Left
			for pre.Right != nil && pre.Right != root {
				pre = pre.Right
			}
			if pre.Right == nil {
				pre.Right = root
				root = root.Left
			} else {
				res = append(res, root.Val)
				pre.Right = nil
				root = root.Right
			}
		} else {
			res = append(res, root.Val)
			root = root.Right
		}
	}

	return res
}

var (
// direction = [][]int{{0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}}
)

func shortestPathBinaryMatrix(grid [][]int) int {

	if grid[0][0] != 0 {
		return -1
	}

	path := make([][]int, len(grid))

	for i := range path {
		path[i] = make([]int, len(grid[0]))
		for j := 0; j < len(grid[0]); j++ {
			path[i][j] = 0x3f3f3f3f
		}
	}
	path[0][0] = 1

	queue := [][]int{{0, 0}}
	for len(queue) != 0 {
		point := queue[0]
		queue = queue[1:]
		x := point[0]
		y := point[1]

		if x == len(grid)-1 && y == len(grid[0])-1 {
			return path[x][y]
		}
		for _, dire := range direction {
			nx := x + dire[0]
			ny := y + dire[1]
			if nx < 0 || nx >= len(grid) || ny < 0 || ny >= len(grid[0]) {
				continue
			}
			if grid[nx][ny] == 1 || path[nx][ny] <= path[x][y]+1 {
				continue
			}

			path[nx][ny] = path[x][y] + 1
			queue = append(queue, []int{nx, ny})
		}
	}

	return -1
}

func threeSumMulti(arr []int, target int) int {

	sort.Ints(arr)
	n := len(arr)
	res := make([][]int, 0, 32)
	count := map[int]int{}
	for _, num := range arr {
		count[num]++
	}

	for i := 0; i < n-2; i++ {
		if i > 0 && arr[i] == arr[i-1] {
			continue
		}
		if arr[i] > target {
			break
		}

		k := n - 1
		for j := i + 1; j < n-1; j++ {
			if j > i+1 && arr[j] == arr[j-1] {
				continue
			}

			for k > j && arr[j]+arr[k] > target-arr[i] {
				k--
			}
			if j >= k {
				break
			}
			if arr[j]+arr[i]+arr[k] == target {
				res = append(res, []int{arr[i], arr[j], arr[k]})
			}
		}
	}

	sum := 0
	for _, tuple := range res {
		if tuple[0] == tuple[1] && tuple[1] == tuple[2] {
			sum += c(count[tuple[0]], 3)
		} else if tuple[0] == tuple[1] {
			sum += c(count[tuple[0]], 2) * count[tuple[2]]
		} else if tuple[1] == tuple[2] {
			sum += c(count[tuple[1]], 2) * count[tuple[0]]
		} else {
			sum += (count[tuple[0]] * count[tuple[1]] * count[tuple[2]]) % (1e9 + 7)
		}

		fmt.Println(sum)
	}

	return sum % (1e9 + 7)

}

func c(n, r int) int {
	return factorial(n) / (factorial(r) * factorial(n-r))
}
func factorial(n int) int {
	if n <= 1 {
		return 1
	}
	return n * factorial(n-1)
}

func distinctAverages(nums []int) int {

	set := make(map[float64]bool, 0)
	sort.Ints(nums)

	i, j := 0, len(nums)-1
	for i < j {
		avg := float64(nums[i]+nums[j]) / 2.0
		set[avg] = true
		i++
		j--
	}

	return len(set)
}

func equalPairs(grid [][]int) int {

	n := len(grid)
	count := map[string]int{}
	for _, row := range grid {
		rowString := ""
		for _, num := range row {
			rowString += strconv.Itoa(num) + "_"
		}
		count[rowString]++
	}

	res := 0
	for j := 0; j < n; j++ {
		colString := ""
		for i := 0; i < n; i++ {
			colString += strconv.Itoa(grid[i][j]) + "_"
		}
		res += count[colString]
	}

	return res
}

var (
	res  [][]int
	temp []int
)

//func subsets(nums []int) [][]int {
//
//	res = make([][]int, 0)
//	temp = make([]int, 0)
//
//	backtrack(nums, 0)
//
//	return res
//}

//func backtrack(nums []int, index int) {
//	n := len(nums)
//	if index >= n {
//		subset := make([]int, len(temp), len(temp))
//		copy(subset, temp)
//		res = append(res, subset)
//		return
//	}
//	temp = append(temp, nums[index])
//	backtrack(nums, index+1)
//	temp = temp[:len(temp)-1]
//	backtrack(nums, index+1)
//}

// var (
//
//	res [][]int
//	temp []int
//
// )
func permute(nums []int) [][]int {

	res = make([][]int, 0)
	temp = make([]int, 0)
	backstack(nums, 0)
	return res
}

func backstack(nums []int, index int) {

	n := len(nums)
	if index == n {
		ans := make([]int, len(temp))
		copy(ans, temp)
		res = append(res, ans)
		return
	}

	for i := index; i < n; i++ {
		nums[i], nums[index] = nums[index], nums[i]
		temp = append(temp, nums[index])
		backstack(nums, index+1)
		nums[i], nums[index] = nums[index], nums[i]
		temp = temp[:len(temp)-1]
	}

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
			temp = max(temp, max(dp[j], j)*max(dp[i-j], i-j))
		}
		dp[i] = temp
	}

	return dp[n]
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

func trianglePath(cost [][]int) int {
	n := len(cost)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}

	dp[1][1] = cost[0][0]

	for i := 2; i <= n; i++ {
		for j := 1; j <= n; j++ {
			dp[i][j] = cost[i-1][j-1] + min(dp[i-1][j-1], dp[i-1][j])
		}
	}

	res := math.MaxInt32
	for i := 1; i <= n; i++ {
		res = min(res, dp[n][i])
	}

	return res
}

func numSmallerByFrequency(queries []string, words []string) []int {
	count := make([]int, 12)
	for _, w := range words {
		count[f(w)]++
	}
	for i := 9; i > 0; i-- {
		count[i] += count[i+1]
	}

	res := make([]int, 0, len(queries))
	for _, query := range queries {
		que := f(query)
		res = append(res, count[que+1])
	}

	return res
}

func f(str string) int {
	count := [26]int{}

	for _, b := range []byte(str) {
		count[b-'a']++
	}

	for i := 0; i < 26; i++ {
		if count[i] != 0 {
			return count[i]
		}
	}
	return 0
}

type ListNode struct {
	Val  int
	Next *ListNode
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func removeZeroSumSublists(head *ListNode) *ListNode {

	dum := &ListNode{}
	dum.Next = head

	arr := make([]int, 0)
	for head != nil {
		arr = append(arr, head.Val)
		head = head.Next
	}

	prefixSum := make([]int, len(arr)+1)
	for i := range arr {
		prefixSum[i+1] = prefixSum[i] + arr[i]
	}
	del := make([]bool, len(arr))

	m := map[int]int{}
	for index, prefix := range prefixSum {
		if _, ok := m[prefix]; ok && !del[m[prefix]] {
			for i := m[prefix]; i < index; i++ {
				del[i] = true
			}
		}
		m[prefix] = index
	}

	node := dum
	for _, delete := range del {
		if delete {
			node.Next = node.Next.Next
		} else {
			node = node.Next
		}
	}

	return dum.Next

}

func Bag01(items []int, values []int, w int) int {

	n := len(items)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, w+1)
	}

	for i := 1; i <= n; i++ {
		for j := 0; j <= w; j++ {
			if j < items[i-1] {
				dp[i][j] = dp[i-1][j]
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i-1][j-items[i-1]]+values[i-1])
			}
		}
	}

	return dp[n][w]

}

func Bag0k(items []int, values []int, w int) int {
	n := len(items)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, w+1)
	}
	for i := 1; i <= n; i++ {
		for j := 0; j <= w; j++ {
			if j < items[i-1] {
				dp[i][j] = dp[i-1][j]
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-items[i-1]]+values[i-1])
			}
		}
	}

	return dp[n][w]

}

func ResourceAllocate(gain [][]int, people int) int {
	n := len(gain)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, people+1)
	}

	for i := 1; i <= n; i++ {
		for j := 0; j <= people; j++ {
			m := 0
			for k := 0; k <= j; k++ {
				m = max(m, dp[i-1][k]+gain[i-1][j-k])
			}
			dp[i][j] = m
		}
	}

	fmt.Println(dp)
	return dp[n][people]

}

func MeetingAllocate(order [][]int) int {
	n := len(order)
	endTime := order[n-1][1]

	dp := make([]int, endTime+1)

	for _, time := range order {
		dp[time[1]] = max(dp[time[0]]+time[1]-time[0], dp[time[1]-1])
	}

	return dp[endTime]
}

func climbStairs(n int) int {
	if n == 1 {
		return 1
	}

	dp := []int{1, 2}

	for i := 3; i <= n; i++ {
		dp[0], dp[1] = dp[1], dp[0]+dp[1]
	}

	return dp[1]

}

var (
	answer [][]string
)

func solveNQueens(n int) [][]string {
	answer = make([][]string, 0)
	backTrackQueens(0, make([]int, n))

	return answer
}
func backTrackQueens(index int, board []int) {
	if index == len(board) {
		solution := make([]string, 0)
		for i := range board {
			row := ""
			for j := 0; j < len(board); j++ {
				if j == board[i] {
					row += "Q"
				} else {
					row += "."
				}
			}
			solution = append(solution, row)
		}
		answer = append(answer, solution)
	} else {
		for i := 0; i < len(board); i++ {
			if check(board, index, i) {
				board[index] = i
				backTrackQueens(index+1, board)
			}
		}
	}
}

func check(board []int, m, n int) bool {
	for i := 0; i < m; i++ {
		if board[i] == n {
			return false
		}
	}
	x, y := m-1, n-1
	for x >= 0 && y >= 0 {
		if board[x] == y {
			return false
		}
		x--
		y--
	}
	x, y = m-1, n+1
	for x >= 0 && y < len(board) {
		if board[x] == y {
			return false
		}
		x--
		y++
	}

	return true
}

func superEggDrop(k int, n int) int {

	if n == 0 || n == 1 {
		return n
	}
	dp := make([][]int, k+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
		dp[i][0] = 0
		dp[i][1] = 1
	}

	for i := 0; i <= n; i++ {
		dp[1][i] = i
	}

	for i := 2; i <= k; i++ {
		for j := 2; j <= n; j++ {
			dp[i][j] = math.MaxInt32
			for x := 0; x <= j; x++ {
				dp[i][j] = min(dp[i][j], 1+max(dp[i-1][x-1], dp[i][j-x]))
			}
		}
	}

	return dp[k][n]

}

func canPartition(nums []int) bool {
	sum := 0
	for _, num := range nums {
		sum += num
	}

	if sum%2 != 0 {
		return false
	}
	n := len(nums)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, sum/2+1)
	}

	for i := 0; i < n; i++ {
		for j := 0; j <= sum/2; j++ {
			if nums[i] > j {
				dp[i+1][j] = dp[i][j]
			} else {
				dp[i+1][j] = max(dp[i][j], dp[i][j-nums[i]]+nums[i])
			}
		}
	}

	return dp[n][sum/2] == sum/2
}

func longestPalindromeSubseq(s string) int {

	bytes := []byte(s)
	n := len(bytes)

	dp := make([][]int, n)
	for i := range dp {
		dp[i] = make([]int, n)
		dp[i][i] = 1
	}

	for x := 1; x <= n-1; x++ {
		for i := 0; i < n-x; i++ {
			j := i + x
			if bytes[i] == bytes[j] {
				dp[i][j] = dp[i+1][j-1] + 2
			} else {
				dp[i][j] = max(dp[i+1][j], dp[i][j-1])
			}
		}
	}

	return dp[0][n-1]
}

func jump(nums []int) int {

	n := len(nums)
	dp := make([]int, n)
	for i := range dp {
		dp[i] = math.MaxInt32
	}
	dp[0] = 0
	for i := range dp {
		for j := i + 1; j <= nums[i]+i; j++ {
			if j >= n {
				break
			}
			dp[j] = min(1+dp[i], dp[j])
		}
	}

	return dp[n-1]
}

func change(amount int, coins []int) int {

	dp := make([]int, amount)
	dp[0] = 1
	for i := 0; i <= amount; i++ {
		for _, coin := range coins {
			if i+coin <= amount {
				dp[i+coin] = 1 + dp[i]

			}
		}
	}

	return dp[amount]
}

var size int

//func pondSizes(land [][]int) []int {
//	m, n := len(land), len(land[0])
//
//	res := make([]int, 0)
//	for i := 0; i < m; i++ {
//		for j := 0; j < n; j++ {
//			if land[i][j] == 0 {
//				size = 0
//				dfs(land, m, n, i, j)
//				res = append(res, size)
//			}
//		}
//	}
//
//	sort.Ints(res)
//	return res
//
//}
//
//func dfs(land [][]int, m, n, x, y int) {
//	if x >= 0 && x < m && y >= 0 && y < n && land[x][y] == 0 {
//		size++
//		land[x][y] = -1
//		dfs(land, m, n, x+1, y)
//		dfs(land, m, n, x, y+1)
//		dfs(land, m, n, x+1, y+1)
//		dfs(land, m, n, x+1, y-1)
//
//	}
//}

var sizeOfClosedIsland = 0

func closedIsland(grid [][]int) int {
	sizeOfClosedIsland = 0
	m := len(grid)
	n := len(grid[0])

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 0 && dfs(grid, m, n, i, j) {
				sizeOfClosedIsland++
			}
		}
	}

	return sizeOfClosedIsland
}

func dfs(grid [][]int, m, n, x, y int) bool {
	if x >= 0 && x < m && y >= 0 && y < n && grid[x][y] == 0 {
		grid[x][y] = -1
		b1 := dfs(grid, m, n, x, y+1)
		b2 := dfs(grid, m, n, x, y-1)
		b3 := dfs(grid, m, n, x-1, y)
		b4 := dfs(grid, m, n, x+1, y)

		return !(x == 0 || x == m-1 || y == 0 || y == n-1) &&
			b1 &&
			b2 &&
			b3 &&
			b4

	}

	return true
}

func maximumSum(arr []int) int {
	n := len(arr)
	if n == 1 {
		return arr[0]
	}
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, 2)
	}
	dp[0][0] = arr[0]
	dp[0][1] = 0

	res := arr[0]
	for i := 1; i <= n; i++ {
		dp[i][0] = max(dp[i-1][0]+arr[i-1], arr[i-1])
		dp[i][1] = max(dp[i-1][1]+arr[i-1], dp[i-1][0])
		res = max(res, max(dp[i][0], dp[i][1]))
	}

	return res
}

func reconstructMatrix(upper int, lower int, colsum []int) [][]int {

	n := len(colsum)
	up := make([]int, n)
	low := make([]int, n)

	for i := 0; i < n; i++ {

		if colsum[i] == 2 {
			up[i], low[i] = 1, 1
			upper--
			lower--
		} else if colsum[i] == 1 {
			if upper > lower {
				up[i] = 1
				upper--
			} else {
				low[i] = 1
				lower--
			}
		}

		if upper < 0 || lower < 0 {
			return [][]int{}
		}
	}

	if upper != 0 || lower != 0 {
		return [][]int{}
	}

	return [][]int{up, low}
}

func vowelStrings(words []string, queries [][]int) []int {
	prefix := make([]int, len(words)+1)
	for i := range words {
		if isOk(words[i]) {
			prefix[i+1] = 1 + prefix[i]
		} else {
			prefix[i+1] = prefix[i]
		}
	}

	res := make([]int, len(queries))
	for i, query := range queries {
		res[i] = prefix[query[1]+1] - prefix[query[0]]
	}

	return res
}

func isOk(word string) bool {
	bytes := []byte(word)
	b := bytes[0]

	if b != 'a' && b != 'o' && b != 'i' && b != 'e' && b != 'u' {
		return false
	}
	if len(word) == 1 {
		return true
	}

	b = bytes[len(bytes)-1]
	return b == 'a' || b == 'e' || b == 'i' || b == 'o' || b == 'u'
}

//
//func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
//
//	dum := &ListNode{}
//	cur := dum
//	add := 0
//	for l1 != nil && l2 != nil {
//		num := l1.Val + l2.Val + add
//		cur.Next = &ListNode{Val: num % 10}
//		if num >= 10 {
//			add = 1
//		} else {
//			add = 0
//		}
//		cur = cur.Next
//		l1 = l1.Next
//		l2 = l2.Next
//	}
//
//	for l1 != nil {
//		num := add + l1.Val
//		cur.Next = &ListNode{Val: num % 10}
//		if num >= 10 {
//			add = 1
//		} else {
//			add = 0
//		}
//		cur = cur.Next
//		l1 = l1.Next
//	}
//
//	for l2 != nil {
//		num := add + l2.Val
//		cur.Next = &ListNode{Val: num % 10}
//		if num >= 10 {
//			add = 1
//		} else {
//			add = 0
//		}
//		cur = cur.Next
//		l2 = l2.Next
//
//	}
//
//	if add != 0 {
//		cur.Next = &ListNode{Val: add}
//	}
//	return dum.Next
//}

func longestAlternatingSubarray(nums []int, threshold int) int {

	n := len(nums)
	curLength := 0
	maxLength := 0

	for start := 0; start < n; start++ {
		if nums[start]%2 == 0 && nums[start] <= threshold {
			curLength = 1
			for i := start + 1; i < n; i++ {
				if nums[i] <= threshold && nums[i]%2 == (i-start)%2 {
					curLength++
				} else {
					start = i - 1
					break
				}
			}
			maxLength = max(maxLength, curLength)
		}
	}

	return maxLength
}

func findPrimePairs(n int) [][]int {

	if n <= 3 {
		return nil
	}

	prime := make([]bool, n+1)
	for i := range prime {
		prime[i] = true
	}
	prime[0], prime[1], prime[2] = false, false, true
	for i := 2; i <= n/2; i++ {
		for j := 2; ; j++ {
			num := i * j
			if num > n {
				break
			} else {
				prime[num] = false
			}
		}
	}

	res := make([][]int, 0)
	for i := 2; i <= n/2; i++ {
		if prime[i] && prime[n-i] {
			res = append(res, []int{i, n - i})
		}
	}

	return res
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {

	stack1 := make([]int, 0)
	stack2 := make([]int, 0)

	for l1 != nil {
		stack1 = append(stack1, l1.Val)
		l1 = l1.Next
	}

	for l2 != nil {
		stack2 = append(stack2, l2.Val)
		l2 = l2.Next
	}

	add := 0
	res := make([]int, 0)
	for len(stack1) != 0 && len(stack2) != 0 {
		num := stack2[len(stack2)-1] + stack1[len(stack1)-1] + add
		if num >= 10 {
			add = 1
		} else {
			add = 0
		}
		res = append(res, num%10)
		stack1 = stack1[:len(stack1)-1]
		stack2 = stack2[:len(stack2)-1]
	}

	for len(stack1) != 0 {
		num := stack1[len(stack1)-1] + add
		if num >= 10 {
			add = 1
		} else {
			add = 0
		}
		res = append(res, num%10)
		stack1 = stack1[:len(stack1)-1]
	}

	for len(stack2) != 0 {
		num := stack2[len(stack2)-1] + add
		if num >= 10 {
			add = 1
		} else {
			add = 0
		}
		res = append(res, num%10)
		stack2 = stack2[:len(stack2)-1]
	}

	if add != 0 {
		res = append(res, 1)
	}

	dum := &ListNode{}
	cur := dum
	for i := len(res) - 1; i >= 0; i-- {
		cur.Next = &ListNode{Val: res[i]}
		cur = cur.Next
	}
	return dum.Next

}

func matrixSum(nums [][]int) int {
	m := len(nums)
	n := len(nums[0])
	if m == 0 {
		return 0
	}

	for i := range nums {
		sort.Ints(nums[i])
	}

	score := 0
	for j := 0; j < n; j++ {
		max := nums[0][j]
		for i := 1; i < m; i++ {
			if nums[i][j] > max {
				max = nums[i][j]
			}
		}
		score += max
	}

	return score
}

func dailyTemperatures(temperatures []int) []int {
	n := len(temperatures)
	stack := make([]int, 0, n)
	res := make([]int, n)
	for i := 0; i < n; i++ {

		for len(stack) > 0 && temperatures[stack[len(stack)-1]] < temperatures[i] {
			top := stack[len(stack)-1]
			res[top] = i - top
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)

	}

	return res
}

func nextGreaterElements(nums []int) []int {

	n := len(nums)
	ints := make([]int, n*2)
	for i := range nums {
		ints[i] = nums[i]
		ints[i+n] = nums[i]
	}

	stack := make([]int, 0, 2*n)

	res := make([]int, 2*n)
	for i := range res {
		res[i] = -1
	}
	for i := 0; i < 2*n; i++ {

		for len(stack) > 0 && ints[stack[len(stack)-1]] < ints[i] {
			top := stack[len(stack)-1]
			res[top] = ints[i]
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}

	return res[:n]
}

func trap(height []int) int {

	n := len(height)
	stack := make([]int, 0, n)
	res := 0

	for i := 0; i < n; i++ {
		if len(stack) == 0 || height[i] <= height[stack[len(stack)-1]] {
			stack = append(stack, i)
		} else {
			for len(stack) > 0 && height[i] > height[stack[len(stack)-1]] {
				mid := stack[len(stack)-1]
				stack = stack[:len(stack)-1]

				if len(stack) > 0 {
					left := stack[len(stack)-1]
					res += (min(height[left], height[i]) - height[mid]) * (i - left - 1)
				}
			}

			stack = append(stack, i)
		}
	}

	return res
}

func largestRectangleArea(heights []int) int {

	n := len(heights)
	stack := make([]int, 0, n)

	heights = append([]int{0}, heights...)
	heights = append(heights, 0)
	stack = append(stack, 0)

	res := 0

	for i := 1; i < len(heights); i++ {
		if heights[stack[len(stack)-1]] <= heights[i] {
			stack = append(stack, i)
		} else {
			for len(stack) > 0 && heights[i] < heights[stack[len(stack)-1]] {
				h := heights[stack[len(stack)-1]]
				stack = stack[:len(stack)-1]
				w := i - stack[len(stack)-1] - 1
				res = max(res, h*w)
			}
			stack = append(stack, i)
		}
	}

	return res
}

func groupAnagrams(strs []string) [][]string {

	m := make(map[[26]int][]string)

	for _, str := range strs {
		counts := [26]int{}
		for _, b := range []byte(str) {
			counts[b-'a']++
		}
		m[counts] = append(m[counts], str)
	}

	res := make([][]string, 0)
	for _, v := range m {
		res = append(res, v)
	}

	return res
}

func longestConsecutive(nums []int) int {

	n := len(nums)
	if n <= 1 {
		return n
	}

	set := make(map[int]bool)
	for _, num := range nums {
		set[num] = true
	}

	res := 1
	for _, num := range nums {
		if set[num-1] {
			continue
		}
		cur := 1
		for set[num+1] {
			num += 1
			cur++
		}
		if res < cur {
			res = cur
		}
	}

	return res
}

func findAnagrams(s string, p string) []int {
	n := len(s)
	if n < len(p) {
		return nil
	}

	temp := [26]int{}
	for _, b := range []byte(p) {
		temp[b-'a']++
	}

	window := [26]int{}

	bytes := []byte(s)
	for i := 0; i < len(p); i++ {
		window[bytes[i]-'a']++
	}

	res := make([]int, 0)
	for i := 0; i < len(s)-len(p); i++ {
		if window == temp {
			res = append(res, i)
		}
		window[bytes[i]-'a']--
		window[bytes[i+len(p)]-'a']++
	}

	if window == temp {
		res = append(res, len(s)-len(p))
	}

	return res

}

func subarraySum(nums []int, k int) int {
	n := len(nums)

	prefix := make([]int, n+1)
	m := make(map[int][]int)
	m[0] = []int{0}

	res := 0
	for i := 1; i <= n; i++ {
		prefix[i] = prefix[i-1] + nums[i-1]
		res += len(m[prefix[i]-k])

		if _, ok := m[prefix[i]]; !ok {
			m[prefix[i]] = make([]int, 0)
		}
		m[prefix[i]] = append(m[prefix[i]], i)
	}

	return res
}

func rotate(nums []int, k int) {

	n := len(nums)
	k = k % n
	if k == 0 {
		return
	}

	reverse(nums, 0, n-1)
	reverse(nums, 0, k-1)
	reverse(nums, k, n-1)
}

func reverse(nums []int, start, end int) {
	for start < end {
		nums[start], nums[end] = nums[end], nums[start]
		start++
		end--
	}
}

func maximumEvenSplit(finalSum int64) []int64 {
	if finalSum%2 != 0 {
		return nil
	}

	sum := int64(0)
	res := make([]int64, 0)
	for i := int64(2); sum < finalSum; i += 2 {
		if finalSum >= sum+i {
			sum += i
			res = append(res, i)
		} else {
			res[len(res)-1] += finalSum - sum
			sum = finalSum
		}
	}

	return res
}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func rightSideView(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	queue := []*TreeNode{root}
	res := make([]int, 0)

	for len(queue) > 0 {
		n := len(queue)
		for i := 0; i < n; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}

			if i == n-1 {
				res = append(res, node.Val)
			}
		}
	}

	return res
}
func twoSum(numbers []int, target int) []int {

	n := len(numbers)
	left := 0
	right := n - 1
	for left < right {
		sum := numbers[left] + numbers[right]
		if sum == target {
			return []int{left + 1, right + 1}
		} else if sum > target {
			right--
		} else {
			left++
		}
	}

	return nil
}

func threeSum(nums []int) [][]int {

	sort.Ints(nums)
	n := len(nums)
	res := make([][]int, 0)

	for i := 0; i < n-2; i++ {
		if nums[i] > 0 {
			break
		}
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		k := n - 1
		for j := i + 1; j < n-1; j++ {
			if j > i+1 && nums[j] == nums[j-1] {
				continue
			}
			for j < k && nums[i]+nums[j]+nums[k] > 0 {
				k--
			}
			if k == j {
				break
			}
			if nums[i]+nums[j]+nums[k] == 0 {
				res = append(res, []int{nums[i], nums[j], nums[k]})
			}
		}

	}

	return res
}

func threeSumClosest(nums []int, target int) int {

	sort.Ints(nums)
	cur := -10000000

	n := len(nums)
	fmt.Println(nums)

	for i := 0; i < n-2; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		k := n - 1
		if nums[i]+nums[k]+nums[k-1] <= target {
			cur = closest(cur, nums[i]+nums[k]+nums[k-1], target)
			continue
		}
		if nums[i]+nums[i+1]+nums[i+2] >= target {
			cur = closest(cur, nums[i]+nums[i+1]+nums[i+2], target)
			break
		}
		for j := i + 1; j < n-1; j++ {
			if k < j {
				break
			}

			if j > i+1 && nums[j] == nums[j-1] {
				continue
			}
			for j < k && nums[i]+nums[j]+nums[k] > target {
				k--
			}

			if k == j {
				cur = closest(cur, nums[i]+nums[j]+nums[k+1], target)
				fmt.Println(cur, "1", nums[i], nums[j], nums[k+1])
			} else {
				cur1 := closest(cur, nums[i]+nums[j]+nums[k], target)
				if k < n-1 {
					cur2 := closest(cur, nums[i]+nums[j]+nums[k+1], target)
					cur = closest(cur1, cur2, target)
				} else {
					cur = cur1
				}

				fmt.Println(cur, "2", nums[i], nums[j], nums[k])

			}

		}
	}

	return cur

}

func closest(x, y, target int) int {
	xx := target - x
	yy := target - y
	if xx < 0 {
		xx = -xx
	}
	if yy < 0 {
		yy = -yy
	}

	if xx < yy {
		return x
	}

	return y
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func partition(head *ListNode, x int) *ListNode {

	if head == nil {
		return head
	}

	more := &ListNode{}
	moreCur := more
	less := &ListNode{}
	lessCur := less

	for head != nil {
		if head.Val < x {
			moreCur.Next = head
			moreCur = moreCur.Next
		} else {
			lessCur.Next = head
			lessCur = lessCur.Next
		}
		head = head.Next
	}
	lessCur.Next = nil

	moreCur.Next = less.Next

	return more.Next

}

//func maxAlternatingSum(nums []int) int64 {
//	n := len(nums)
//	dp := make([][]int, n)
//	for i := range dp {
//		dp[i] = make([]int, 4)
//	}
//	// 选为偶数
//	dp[0][0] = nums[0]
//	//选为奇数
//	dp[0][1] = 0
//	//没选,但下一个是偶数
//	dp[0][2] = 0
//	// 没选,但下一个是奇数
//	dp[0][3] = 0
//
//	for i := 1; i < n; i++ {
//		dp[i][0] = nums[i] + max(dp[i-1][1], dp[i-1][2])
//		dp[i][1] = -nums[i] + max(dp[i-1][0], dp[i-1][3])
//		dp[i][2] = max(dp[i-1][1], dp[i-1][2])
//		dp[i][3] = max(dp[i-1][3], dp[i-1][0])
//	}
//
//	res := dp[n-1][0]
//	for i := 1; i <= 3; i++ {
//		res = max(res, dp[n-1][i])
//	}
//	return int64(res)
//}

func maxAlternatingSum(nums []int) int64 {
	n := len(nums)
	dp := make([][]int, n)
	for i := range dp {
		dp[i] = make([]int, 2)
	}
	// 选为偶数
	dp[0][0] = nums[0]
	//选为奇数
	dp[0][1] = 0

	for i := 1; i < n; i++ {
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]+nums[i])
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]-nums[i])

	}

	return int64(max(dp[n-1][0], dp[n-1][1]))
}

func alternateDigitSum(n int) int {

	stack := make([]int, 0)

	for n > 0 {
		stack = append(stack, n%10)
		n = n / 10
	}

	sign := 1
	res := 0
	for i := len(stack) - 1; i >= 0; i-- {
		res += sign * stack[i]
		sign = -sign
	}

	return res
}

func minFallingPathSum(matrix [][]int) int {

	m := len(matrix)
	n := len(matrix[0])

	dp := make([][]int, m)
	for i := range dp {
		dp[i] = make([]int, n)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i == 0 {
				dp[i][j] = matrix[i][j]
				continue
			}

			if j == 0 {
				dp[i][j] = min(dp[i-1][j], dp[i-1][j+1]) + matrix[i][j]
			} else if j == n-1 {
				dp[i][j] = min(dp[i-1][j], dp[i-1][j-1]) + matrix[i][j]
			} else {
				dp[i][j] = min(min(dp[i-1][j], dp[i-1][j-1]), dp[i-1][j+1]) + matrix[i][j]
			}
		}
	}

	res := math.MaxInt32
	for j := 0; j < n; j++ {
		res = min(res, dp[m-1][j])
	}

	return res
}

var rand7 func() int

func rand10() int {
	two := rand7()
	for two == 7 {
		two = rand7()
	}
	five := rand7()
	for five > 5 {
		five = rand7()
	}

	if two%2 == 1 {
		return five
	}
	return five + 5
}

var weight []int

func getKth(lo int, hi int, k int) int {
	if lo == hi {
		return lo
	}
	weight = make([]int, hi+1)
	weight[1] = 0
	weight[2] = 1
	arr := make([]int, hi-lo+1)
	for i := lo; i <= hi; i++ {
		getWeight(i)
		arr[i-lo] = i
	}

	sort.Slice(arr, func(i, j int) bool {
		if weight[arr[i]] < weight[arr[j]] {
			return true
		} else if weight[arr[i]] > weight[arr[j]] {
			return false
		} else {
			return arr[i] < arr[j]
		}
	})

	return arr[k-1]

}

func getWeight(num int) int {
	if num == 1 {
		return 0
	}
	time := 0
	n := num
	for n != 1 {
		time++
		if n%2 == 0 {
			n = n / 2
		} else {
			n = 3*n + 1
		}
	}
	weight[num] = time
	return weight[num]
}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
var move int

func distributeCoins(root *TreeNode) int {

	move = 0

	countAndSum(root)
	return move
}

func countAndSum(root *TreeNode) (int, int) {
	if root == nil {
		return 0, 0
	}

	l1, l2 := countAndSum(root.Left)
	r1, r2 := countAndSum(root.Right)
	move += abs(l2-l1) + abs(r2-r1)
	return l1 + r1 + 1, l2 + r2 + root.Val
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

func fourSum(nums []int, target int) [][]int {

	sort.Ints(nums)
	n := len(nums)

	res := make([][]int, 0)
	for i := 0; i < n-3; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		l := n - 1

		for ; l >= i+3; l-- {
			if l < n-1 && nums[l] == nums[l+1] {
				continue
			}
			tar := target - nums[i] - nums[l]
			j := i + 1
			k := l - 1
			for j < k {
				s := nums[j] + nums[k]
				if s == tar {
					res = append(res, []int{nums[i], nums[j], nums[k], nums[l]})
					for j < k && nums[j] == nums[j+1] {
						j++
					}
					j++
					for j < k && nums[k] == nums[k-1] {
						k--
					}
					k--
				} else if s > tar {
					k--
				} else {
					j++
				}
			}
		}

	}

	return res
}

func sortArray(nums []int) []int {

	_sort(nums, 0, len(nums)-1)
	return nums
}

func _sort(nums []int, left, right int) {
	if left >= right {
		return
	}

	i := part(nums, left, right)
	_sort(nums, left, i-1)
	_sort(nums, i+1, right)
}

func part(nums []int, left, right int) int {
	index := left + (right-left)/2
	nums[left], nums[index] = nums[index], nums[left]

	mid := nums[left]
	index = left

	for i := index + 1; i <= right; i++ {
		if nums[i] < mid {
			index++
			nums[i], nums[index] = nums[index], nums[i]
		}
	}

	nums[left], nums[index] = nums[index], nums[left]
	return index
}

var direction [][]int = [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}

func robotSim(commands []int, obstacles [][]int) int {

	obstaclesSet := make(map[int]bool)

	for _, ob := range obstacles {
		obstaclesSet[ob[0]*1000000+ob[1]] = true
	}

	directionIndex := 0
	x := 0
	y := 0
	res := 0
	for _, cmd := range commands {
		if cmd == -1 {
			directionIndex = (directionIndex + 1) % 4
		} else if cmd == -2 {
			directionIndex = (directionIndex - 1 + 4) % 4
		} else {
			for i := 0; i < cmd; i++ {
				xx := x + direction[directionIndex][0]
				yy := y + direction[directionIndex][1]
				if obstaclesSet[xx*1000000+yy] {
					res = max(res, x*x+y*y)
					continue
				} else {
					x = xx
					y = yy
					res = max(res, x*x+y*y)
				}
			}
		}
	}

	return res

}

func lemonadeChange(bills []int) bool {

	five, ten := 0, 0
	for _, bill := range bills {
		if bill == 5 {
			five++
		} else if bill == 10 {
			if five < 1 {
				return false
			} else {
				five--
				ten++
			}
		} else {
			if ten >= 1 && five >= 1 {
				ten--
				five--
			} else if five >= 3 {
				five -= 3
			} else {
				return false
			}
		}
	}

	return true
}

func successfulPairs(spells []int, potions []int, success int64) []int {
	n := len(spells)
	res := make([]int, n)

	sort.Ints(potions)
	for i, spell := range spells {
		res[i] = binary(potions, float64(success)/float64(spell))
	}

	return res
}

func binary(potions []int, target float64) (number int) {
	fmt.Println(target)
	m := len(potions)

	left := 0
	right := m
	for left < right {
		mid := left + (right-left)/2
		if float64(potions[mid]) < target {
			left = mid + 1
		} else if float64(potions[mid]) >= target {
			right = mid
		}
	}

	return m - left
}

func findTheCity(n int, edges [][]int, distanceThreshold int) int {

	graph := make([][]int, n)
	for i := range graph {
		graph[i] = make([]int, n)
		for j := 0; j < n; j++ {
			graph[i][j] = math.MaxInt32 / 2
		}
	}

	for _, edge := range edges {
		graph[edge[0]][edge[1]] = edge[2]
		graph[edge[1]][edge[0]] = edge[2]
	}

	for i := 0; i < n; i++ {
		for src := 0; src < n; src++ {

			for end := src + 1; end < n; end++ {

				graph[src][end] = min(graph[src][end], graph[src][i]+graph[i][end])
				graph[end][src] = graph[src][end]

			}

		}
	}

	//fmt.Println(graph)
	ansIndex := 0
	minCity := math.MaxInt32
	for i := 0; i < n; i++ {
		cityNum := 0
		for j := 0; j < n; j++ {
			if graph[i][j] <= distanceThreshold {
				cityNum++
			}
		}
		if cityNum <= minCity {
			minCity = cityNum
			ansIndex = i
		}
	}

	return ansIndex

}

func maximumSum(nums []int) int {
	m := map[int]int{}
	res := -1
	for _, num := range nums {
		sum := calSum(num)
		now, ok := m[sum]
		if ok {
			res = max(res, now+num)
			m[sum] = max(now, num)
		} else {
			m[sum] = num
		}
	}

	return res
}

func calSum(num int) int {
	res := 0
	for num > 0 {
		res += num % 10
		num /= 10
	}

	return res
}

func minDeletion(nums []int) int {

	n := len(nums)
	cur := 0

	res := 0

	for cur < n-1 {
		if (cur-res)%2 == 0 {
			if nums[cur] == nums[cur+1] {
				res++
				cur++
			} else {
				cur += 2
			}
		} else {
			cur++
		}
	}

	if (n-res)%2 != 0 {
		return res + 1
	}
	return res
}

func minPathCost(grid [][]int, moveCost [][]int) int {

	m := len(grid)
	n := len(grid[0])
	dp := make([][]int, m)
	for i := range dp {
		dp[i] = make([]int, n)
		if i == 0 {
			for j := 0; j < n; j++ {
				dp[0][j] = grid[0][j]
			}
		} else {
			for j := 0; j < n; j++ {
				dp[i][j] = math.MaxInt32
			}
		}
	}

	for i := 1; i < m; i++ {
		for j := 0; j < n; j++ {
			for k := 0; k < n; k++ {
				dp[i][j] = min(dp[i-1][k]+moveCost[grid[i-1][k]][j]+grid[i][j], dp[i][j])
			}
		}
	}

	res := math.MaxInt32
	for j := 0; j < n; j++ {
		res = min(res, dp[m-1][j])
	}

	return res
}

func sumSubarrayMins(arr []int) int {
	var mod int = 1e9 + 7
	n := len(arr)
	stack := []int{}
	left := make([]int, n)
	right := make([]int, n)

	for i, num := range arr {
		for len(stack) > 0 && num <= arr[stack[len(stack)-1]] {
			stack = stack[:len(stack)-1]
		}

		if len(stack) == 0 {
			left[i] = i + 1
		} else {
			left[i] = i - stack[len(stack)-1]
		}

		stack = append(stack, i)
	}

	stack = []int{}
	for i := n - 1; i >= 0; i-- {
		for len(stack) > 0 && arr[i] < arr[stack[len(stack)-1]] {
			stack = stack[:len(stack)-1]
		}
		if len(stack) == 0 {
			right[i] = n - i
		} else {
			right[i] = stack[len(stack)-1] - i
		}

		stack = append(stack, i)
	}

	res := 0
	for i := 0; i < n; i++ {
		res += left[i] * arr[i] * right[i]
		res %= mod
	}

	fmt.Println(left)
	fmt.Println(right)
	return res
}

func firstCompleteIndex(arr []int, mat [][]int) int {

	m := len(mat)
	n := len(mat[0])

	rowColored := make([]int, m)
	colColored := make([]int, n)
	for i := range rowColored {
		rowColored[i] = n
	}
	for i := range colColored {
		colColored[i] = m
	}

	matIndex := map[int][2]int{}
	for i := range mat {
		for j := range mat[i] {
			matIndex[mat[i][j]] = [2]int{i, j}
		}
	}

	for i, num := range arr {
		x, y := matIndex[num][0], matIndex[num][1]
		rowColored[x]--
		colColored[y]--
		if rowColored[x] == 0 || colColored[y] == 0 {
			return i
		}
	}

	return -1
}

func carPooling(trips [][]int, capacity int) bool {

	fromPassengers := make([]int, 1001)
	toPassengers := make([]int, 1001)

	for _, trip := range trips {
		fromPassengers[trip[1]] += trip[0]
		toPassengers[trip[2]] += trip[0]
	}

	for i := 0; i <= 1000; i++ {
		capacity += toPassengers[i]
		capacity -= fromPassengers[i]

		if capacity < 0 {
			return false
		}
	}

	return true

}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func bstToGst(root *TreeNode) *TreeNode {

	dfs1038(root, 0)

	return root
}

func dfs1038(root *TreeNode, num int) int {
	if root == nil {
		return num
	}

	rightSum := dfs1038(root.Right, num)

	leftSum := dfs1038(root.Left, rightSum+root.Val)
	root.Val = rightSum + root.Val

	return leftSum
}

var sumOfFuel int64 = 0

func minimumFuelCost(roads [][]int, seats int) int64 {

	sumOfFuel = 0
	g := make([][]int, len(roads)+1)
	for _, road := range roads {
		g[road[0]] = append(g[road[0]], road[1])
		g[road[1]] = append(g[road[1]], road[0])
	}
	for _, from := range g[0] {
		dfs2477(from, 0, seats, g)
	}

	return sumOfFuel

}

func dfs2477(f, t, seats int, g [][]int) int {

	sum := 1
	for _, from := range g[f] {
		if from != t {
			sum += dfs2477(from, f, seats, g)
		}
	}

	if sum%seats == 0 {
		sumOfFuel += int64(sum / seats)
	} else {
		sumOfFuel += int64(sum/seats) + 1
	}

	return sum
}

var numOfRedConn = 0

func minReorder(n int, connections [][]int) int {

	numOfRedConn = 0
	g := make([][]int, n)
	dire := make(map[string]bool)

	for _, conn := range connections {
		g[conn[0]] = append(g[conn[0]], conn[1])
		g[conn[1]] = append(g[conn[1]], conn[0])
		dire[strconv.Itoa(conn[0])+"_"+strconv.Itoa(conn[1])] = true
	}

	for _, conn := range g[0] {
		dfs1466(g, dire, conn, 0)
	}

	return numOfRedConn
}

func dfs1466(g [][]int, dire map[string]bool, from, to int) {
	if !dire[strconv.Itoa(from)+"_"+strconv.Itoa(to)] {
		numOfRedConn++
	}
	for _, conn := range g[from] {
		if conn != to {
			dfs1466(g, dire, conn, from)
		}
	}
}

func minCostClimbingStairs(cost []int) int {

	n := len(cost)
	dp := make([]int, n+1)
	dp[0], dp[1] = 0, 0
	for i := 2; i <= n; i++ {
		dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2])
	}

	return dp[n]
}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

func reverseOddLevels(root *TreeNode) *TreeNode {

	queue := []*TreeNode{root}
	flag := true
	for len(queue) > 0 {
		n := len(queue)
		if queue[0].Left == nil {
			break
		}
		if flag {
			oddLevelValueArr := make([]int, 0, n*2)
			for i := 0; i < n; i++ {
				oddLevelValueArr = append(oddLevelValueArr, queue[i].Left.Val, queue[i].Right.Val)
			}
			for i := 0; i < n; i++ {
				queue[i].Left.Val = oddLevelValueArr[2*n-i*2-1]
				queue[i].Right.Val = oddLevelValueArr[2*n-i*2-2]
				queue = append(queue, queue[i].Left, queue[i].Right)
			}

			queue = queue[n:]
			flag = false
		} else {
			for i := 0; i < n; i++ {
				queue = append(queue, queue[i].Left, queue[i].Right)
			}
			queue = queue[n:]
			flag = true
		}
	}

	return root

}

func makeSmallestPalindrome(s string) string {

	letters := []byte(s)
	n := len(letters)
	ans := make([]byte, len(letters))

	for i, letter := range letters {
		pLetter := letters[n-i-1]
		if letter < pLetter {
			ans[i] = letter
		} else {
			ans[i] = pLetter
		}
	}

	return string(ans)
}

func minimumEffortPath(heights [][]int) int {

	m, n := len(heights), len(heights[0])

	dp := make([][]int, m)
	for i := range dp {
		dp[i] = make([]int, n)
	}

	dp[0][0] = 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i == 0 && j == 0 {
				continue
			} else if i == 0 {
				dp[i][j] = max(dp[i][j-1], abs(heights[i][j]-heights[i][j-1]))
			} else if j == 0 {
				dp[i][j] = max(dp[i-1][j], abs(heights[i][j]-heights[i-1][j]))
			} else {
				leftMax := max(dp[i][j-1], abs(heights[i][j]-heights[i][j-1]))
				topMax := max(dp[i-1][j], abs(heights[i][j]-heights[i-1][j]))
				dp[i][j] = min(leftMax, topMax)
			}
		}
	}

	return dp[m-1][n-1]
}

func maximumSumOfHeights(maxHeights []int) int64 {

	n := len(maxHeights)
	left := make([]int, n)
	right := make([]int, n)
	left[0] = maxHeights[0]
	right[n-1] = maxHeights[n-1]

	stack := []int{}
	for i, height := range maxHeights {
		if len(stack) == 0 {
			stack = append(stack, i)
			continue
		}

		for len(stack) != 0 && maxHeights[stack[len(stack)-1]] >= height {
			stack = stack[:len(stack)-1]
		}
		if len(stack) == 0 {
			left[i] = (i + 1) * height
		} else {
			top := stack[len(stack)-1]
			left[i] = left[top] + (i-top)*height
		}
		stack = append(stack, i)
	}

	stack = make([]int, 0)
	for i := n - 1; i >= 0; i-- {
		height := maxHeights[i]
		if len(stack) == 0 {
			stack = append(stack, i)
			continue
		}
		for len(stack) > 0 && maxHeights[stack[len(stack)-1]] >= height {
			stack = stack[:len(stack)-1]
		}

		if len(stack) == 0 {
			right[i] = (n - i) * height
		} else {
			top := stack[len(stack)-1]
			right[i] = right[top] + (top-i)*height
		}

		stack = append(stack, i)

	}
	res := 0
	for i := 0; i < n; i++ {
		res = max(res, left[i]+right[i]-maxHeights[i])
	}

	return int64(res)
}

type numHeap []int

func (n numHeap) Len() int {
	return len(n)
}

func (n *numHeap) Less(i, j int) bool {
	return (*n)[i] >= (*n)[j]
}

func (n numHeap) Swap(i, j int) {
	n[i], n[j] = n[j], n[i]
}

func (n *numHeap) Push(x any) {
	num := x.(int)
	*n = append(*n, num)
}

func (n *numHeap) Pop() any {
	num := (*n)[n.Len()-1]
	*n = (*n)[:n.Len()-1]

	return num
}

func minStoneSum(piles []int, k int) int {

	ints := numHeap(piles)
	heap.Init(&ints)

	for i := 0; i < k; i++ {
		pop := heap.Pop(&ints).(int)
		heap.Push(&ints, int(pop-pop/2))
	}
	fmt.Println(ints)

	res := 0
	for ints.Len() != 0 {
		res += heap.Pop(&ints).(int)
		fmt.Println(res)
	}

	return res
}

func minimumPerimeter(neededApples int64) int64 {

	var currentApples int64 = 0
	var currentCircles int64 = 0

	for currentApples < neededApples {
		currentCircles++

		// 计算单边长度
		var wideSum int64 = currentCircles
		for i := int64(0); i < currentCircles; i++ {
			wideSum += 2 * (currentCircles + i)
		}

		// 计算新的一圈有多少个果子
		var apples int64 = -4*currentCircles*2 + wideSum*4
		currentApples += apples
	}

	return currentCircles * 8

}

func numOfBurgers(tomatoSlices int, cheeseSlices int) []int {

	if tomatoSlices%2 != 0 || tomatoSlices > 4*cheeseSlices || tomatoSlices < 2*cheeseSlices {
		return nil
	}

	jumbo := (tomatoSlices - 2*cheeseSlices) / 2
	return []int{jumbo, (tomatoSlices - jumbo*4) / 2}
}

func minOperationsMaxProfit(customers []int, boardingCost int, runningCost int) int {

	if runningCost > boardingCost<<2 {
		return -1
	}

	n := len(customers)
	currentOperations := 0
	currentProfit := 0
	maxProfit := 0
	maxProfileOps := -1
	currentPeople := 0

	for true {
		if currentPeople == 0 && currentOperations >= n {
			return maxProfileOps
		}
		if currentOperations < n {
			currentPeople += customers[currentOperations]
			currentOperations++
		}
		if currentPeople <= 4 {
			currentProfit += currentPeople*boardingCost - runningCost
			currentPeople = 0
		} else {
			currentPeople -= 4
			currentProfit += 4*boardingCost - runningCost
		}
		fmt.Println(currentProfit)
		if currentProfit > maxProfit {
			maxProfit = currentProfit
			maxProfileOps = currentOperations + 1
		}
	}

	return maxProfileOps
}

func minCost(nums []int, x int) int64 {

	var minSum int64 = math.MaxInt64
	n := len(nums)
	minNums := make([]int, n)
	copy(minNums, nums)
	minSum = sumOfInts(minNums)
	for i := 1; i < n; i++ {
		for j := 0; j < n; j++ {
			minNums[j] = min(minNums[j], minNums[(j+n)%n])
		}
		currentSum := sumOfInts(minNums) + int64(i)*int64(x)
		minSum = min(currentSum, minSum)
	}

	return minSum
}

func sumOfInts(nums []int) (sum int64) {
	for _, n := range nums {
		sum += int64(n)
	}
	return
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func removeNodes(head *ListNode) *ListNode {

	stack := make([]*ListNode, 0)

	dum := &ListNode{Next: head}

	current := dum
	for current.Next != nil {
		if len(stack) == 0 {
			stack = append(stack, current)
			current = current.Next
			continue
		}
		currentVal := current.Next.Val
		for len(stack) > 0 && currentVal > stack[len(stack)-1].Next.Val {
			topPrev := stack[len(stack)-1]
			topPrev.Next = topPrev.Next.Next
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, current)
		current = current.Next
	}

	return dum.Next
}

func minimumRemoval(beans []int) int64 {

	n := len(beans)
	sort.Ints(beans)
	prefix := make([]int, n)
	for i, bean := range beans {
		if i == 0 {
			prefix[i] = bean
			continue
		}
		prefix[i] = prefix[i-1] + bean
	}

	var res int64 = math.MaxInt64
	var current int64 = 0
	for i := range prefix {
		current = 0
		if i != 0 {
			current += int64(prefix[i-1])
		}
		current += int64(prefix[n-1]-prefix[i]) - int64((n-i-1)*beans[i])
		fmt.Println(current)
		res = min(res, current)
	}

	return res
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func insertGreatestCommonDivisors(head *ListNode) *ListNode {

	dum := &ListNode{Next: head}

	cur := head
	for cur.Next != nil {
		value := gcd(cur.Val, cur.Next.Val)
		mid := &ListNode{Val: value, Next: cur.Next}
		cur.Next = mid
		cur = mid.Next
	}
	return dum.Next
}

func gcd(a, b int) int {
	if b > a {
		a, b = b, a
	}
	if a%b == 0 {
		return b
	} else {
		return gcd(b, a%b)
	}
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func removeNodes(head *ListNode) *ListNode {

	dum := &ListNode{Next: head}
	stack := make([]*ListNode, 0)
	cur := head
	for cur != nil {
		if len(stack) == 0 {
			dum.Next = cur
			stack = append(stack, cur)
			cur = cur.Next
			continue
		}

		for len(stack) != 0 && cur.Val > stack[len(stack)-1].Val {
			stack = stack[:len(stack)-1]
		}
		if len(stack) == 0 {
			dum.Next = cur
		} else {
			stack[len(stack)-1].Next = cur
		}
		stack = append(stack, cur)

		cur = cur.Next

	}

	return dum.Next
}

func maximumRows(matrix [][]int, numSelect int) int {

	m := len(matrix)
	n := len(matrix[0])

	arr := make([]int, m)
	for i := range arr {
		num := 0
		for j := range matrix[i] {
			num |= matrix[i][j] << j
		}
		arr[i] = num
	}

	selectArr := make([]int, 0)
	var produceSelect func(n, index, numSelect, now int)
	produceSelect = func(n, index, numSelect, now int) {
		if index == n {
			if numSelect == 0 {
				selectArr = append(selectArr, now)
			}
			return
		}

		if numSelect < 0 {
			return
		}

		produceSelect(n, index+1, numSelect, now)

		// select 1
		now |= 1 << index
		produceSelect(n, index+1, numSelect-1, now)
	}

	produceSelect(n, 0, numSelect, 0)

	res := 0
	for _, selected := range selectArr {
		cur := 0
		for _, num := range arr {
			if check2397(num, selected) {
				cur++
			}
		}
		res = max(res, cur)
	}

	fmt.Println(arr)
	fmt.Println(selectArr)

	return res

}

func check2397(num, selected int) bool {
	for i := 0; i < 31; i++ {
		if num>>i&1 == 1 {
			if selected>>i&1 == 1 {
				continue
			} else {
				return false
			}
		}
	}

	return true
}

func sumIndicesWithKSetBits(nums []int, k int) int {

	res := 0

	for i, num := range nums {
		numberOf1 := 0
		for i != 0 {
			if i&1 == 1 {
				numberOf1++
			}
			if numberOf1 > k {
				break
			}
			i = i >> 1
		}

		if numberOf1 == k {
			res += num
		}
	}

	return res
}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func rangeSumBST(root *TreeNode, low int, high int) int {
	if root == nil {
		return 0
	}
	if root.Val >= low && root.Val <= high {
		return root.Val + rangeSumBST(root.Left, low, high) + rangeSumBST(root.Right, low, high)
	} else {
		return rangeSumBST(root.Left, low, high) + rangeSumBST(root.Right, low, high)
	}
}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val   int
 *     Left  *TreeNode
 *     Right *TreeNode
 * }
 */

func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root.Val == p.Val || root.Val == q.Val {
		return root
	}
	if root.Val < p.Val && root.Val < q.Val {
		return lowestCommonAncestor(root.Right, p, q)
	}
	if root.Val > p.Val && root.Val > q.Val {
		return lowestCommonAncestor(root.Left, p, q)
	}

	return root

}

var arr []int

func closestNodes(root *TreeNode, queries []int) [][]int {
	arr = []int{}
	middleSeq(root)

	n := len(arr)
	ans := make([][]int, 0, len(queries))
	for _, query := range queries {
		if query < arr[0] {
			ans = append(ans, []int{-1, arr[0]})
		} else if query > arr[n-1] {
			ans = append(ans, []int{arr[n-1], -1})
		} else {
			left, right := 0, n
			for left < right {
				mid := left + (right-left)/2
				if query == arr[mid] {
					left = mid
					break
				} else if query > arr[mid] {
					left = mid + 1
				} else {
					right = mid
				}
			}
			if arr[left] == query {
				ans = append(ans, []int{query, query})
			} else {
				ans = append(ans, []int{arr[left-1], arr[left]})
			}
		}
	}

	return ans
}

func middleSeq(root *TreeNode) {
	if root == nil {
		return
	}
	middleSeq(root.Left)
	arr = append(arr, root.Val)
	middleSeq(root.Right)
}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func kthLargestLevelSum(root *TreeNode, k int) int64 {
	queue := []*TreeNode{root}

	sumOfLevel := make([]int64, 0, k)
	for len(queue) != 0 {
		n := len(queue)
		var sum int64 = 0
		for i := 0; i < n; i++ {
			sum += int64(queue[i].Val)
			if queue[i].Left != nil {
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				queue = append(queue, queue[i].Right)
			}
		}
		queue = queue[n:]
		sumOfLevel = append(sumOfLevel, sum)
	}

	if len(sumOfLevel) < k {
		return -1
	}
	sort.Slice(sumOfLevel, func(i, j int) bool {
		if sumOfLevel[i] < sumOfLevel[j] {
			return false
		}
		return true
	})

	return sumOfLevel[k-1]

}

func validPartition(nums []int) bool {

	n := len(nums)
	dp := make([]bool, n+1)
	dp[0] = true
	dp[1] = false
	if nums[1] == nums[0] {
		dp[2] = true
	} else {
		dp[2] = false
	}
	for i := 2; i < n; i++ {
		if nums[i] == nums[i-1] && nums[i] == nums[i-2] {
			dp[i+1] = dp[i-2] || dp[i-1]
		} else if nums[i] == nums[i-1] {
			dp[i+1] = dp[i-1]
		} else if nums[i] == nums[i-1]+1 && nums[i] == nums[i-2]+2 {
			dp[i+1] = dp[i-1]
		}
	}

	return dp[n]
}

type FrequencyTracker struct {
	// 记录每个数字的频率值
	numberFrequency map[int]int

	// 记录每个频率值的个数count
	frequencyCount map[int]int
}

func Constructor() FrequencyTracker {
	return FrequencyTracker{make(map[int]int, 0), make(map[int]int, 0)}
}

func (this *FrequencyTracker) Add(number int) {
	if fre, ok := this.numberFrequency[number]; !ok {
		this.numberFrequency[number] = 1
		this.frequencyCount[1]++
	} else {
		this.numberFrequency[number]++
		this.frequencyCount[fre]--
		this.frequencyCount[fre+1]++

	}
}

func (this *FrequencyTracker) DeleteOne(number int) {
	if fre, ok := this.numberFrequency[number]; ok && fre > 0 {
		this.numberFrequency[number]--
		this.frequencyCount[fre]--
		this.frequencyCount[fre-1]++
	}
}

func (this *FrequencyTracker) HasFrequency(frequency int) bool {
	fre := this.frequencyCount[frequency]
	return fre > 0
}

func minimumSum(nums []int) int {

	n := len(nums)
	leftMin := make([]int, n)
	rightMin := make([]int, n)

	leftMin[0] = nums[0]
	rightMin[n-1] = nums[n-1]
	for i := 1; i < n; i++ {
		leftMin[i] = min(leftMin[i-1], nums[i])
		rightMin[n-i-1] = min(rightMin[n-i], nums[n-i-1])
	}

	var res = math.MaxInt32
	for i := 1; i < n-1; i++ {
		if nums[i] > leftMin[i-1] && nums[i] > rightMin[i+1] {
			res = min(res, nums[i]+leftMin[i-1]+rightMin[i+1])
		}
	}

	if res == math.MaxInt32 {
		return -1
	} else {
		return res
	}
}

func isValidSerialization(preorder string) bool {

	letters := strings.Split(preorder, ",")
	stack := make([]string, 0)
	for _, letter := range letters {
		stack = append(stack, letter)
		for len(stack) >= 3 && stack[len(stack)-1] == "#" && stack[len(stack)-2] == "#" && stack[len(stack)-3] != "#" {
			stack = stack[:len(stack)-3]
			stack = append(stack, "#")
		}
	}

	fmt.Println(stack)
	return len(stack) == 1 && stack[0] == "#"
}

const mod = 1e9 + 7

func firstDayBeenInAllRooms(nextVisit []int) int {

	n := len(nextVisit)
	dp := make([]int, n)
	dp[0] = 0
	dp[1] = 2
	for i := 2; i < n; i++ {
		fmt.Println(dp[i-1] - dp[nextVisit[i-1]])
		dp[i] = (dp[i-1] - dp[nextVisit[i-1]] + 2) % mod
		dp[i] = (dp[i] + dp[i-1]) % mod
	}

	fmt.Println(dp)

	return dp[n-1] % mod
}

func finalString(s string) string {

	n := len(s)
	res := make([]byte, 0, n)
	bytes := []byte(s)
	for _, b := range bytes {
		if b == 'i' {
			reverseI(res)
		} else {
			res = append(res, b)
		}
	}

	return string(res)

}

func reverseI(bytes []byte) {
	n := len(bytes)
	for i := 0; i < n/2; i++ {
		bytes[i], bytes[n-i-1] = bytes[n-i-1], bytes[i]
	}
}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
var maxDiff int

func maxAncestorDiff(root *TreeNode) int {
	maxDiff = 0

	getMaxAndMinOfTree(root, root.Val)

	return maxDiff
}

func getMaxAndMinOfTree(root *TreeNode, parentVal int) (maxValue, minValue int) {
	if root == nil {
		return parentVal, parentVal
	}
	leftMax, leftMin := getMaxAndMinOfTree(root.Left, root.Val)
	rightMax, rightMin := getMaxAndMinOfTree(root.Right, root.Val)

	maxDiff = max(maxDiff, abs(root.Val-max(leftMax, rightMax)))
	maxDiff = max(maxDiff, abs(root.Val-min(leftMin, rightMin)))

	maxValue = max(root.Val, max(leftMax, rightMax))
	minValue = min(root.Val, min(leftMin, rightMin))
	return
}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func sumNumbers(root *TreeNode) int {

	return dfs049(root, []int{})
}

func dfs049(root *TreeNode, numList []int) (sum int) {
	if root == nil {
		return 0
	}
	if root.Left == nil && root.Right == nil {
		numList = append(numList, root.Val)
		for _, num := range numList {
			sum *= 10
			sum += num
		}
		numList = numList[:len(numList)-1]

		return
	}

	numList = append(numList, root.Val)
	leftSum := dfs049(root.Left, numList)
	rightSum := dfs049(root.Right, numList)

	numList = numList[:len(numList)-1]
	return leftSum + rightSum
}
