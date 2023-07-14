package solution

import (
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
	direction = [][]int{{0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}}
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
