package dmsxl

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */

type ListNode struct {
	Val  int
	Next *ListNode
}

func removeElements(head *ListNode, val int) *ListNode {

	dum := &ListNode{}
	dum.Next = head

	cur := dum

	for cur != nil {
		for cur.Next != nil && cur.Next.Val == val {
			cur.Next = cur.Next.Next
		}
		cur = cur.Next
	}

	return dum.Next
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {

	var pre *ListNode
	cur := head
	for cur != nil {
		after := cur.Next
		cur.Next = pre

		pre = cur
		cur = after
	}

	return pre
}
