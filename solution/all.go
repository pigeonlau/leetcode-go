package solution

import "fmt"

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
