package dmsxl

// 代码随想录算法训练营第一天|704.二分查找、27.移除元素
func search(nums []int, target int) int {

	left := 0
	right := len(nums)

	for left < right {
		mid := left + (right-left)>>1
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid
		}
	}

	return -1
}

func removeElement(nums []int, val int) int {
	slow := 0
	fast := 0

	for fast < len(nums) {
		if nums[fast] != val {
			nums[slow] = nums[fast]
			slow++
			fast++
		} else {
			fast++
		}
	}

	return slow
}
