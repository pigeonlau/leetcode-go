package solution

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
