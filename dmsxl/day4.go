package dmsxl

func detectCycle(head *ListNode) *ListNode {

	if head == nil {
		return nil
	}
	slow, fast := head, head

	for fast.Next != nil && fast.Next.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if fast == slow {
			node := head
			for node != slow {
				node = node.Next
				slow = slow.Next
			}

			return node
		}
	}

	return nil
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {

	dum := &ListNode{}
	dum.Next = head

	slow, fast := dum, dum
	for i := 0; i < n+1; i++ {
		fast = fast.Next
	}

	for fast != nil {
		fast = fast.Next
		slow = slow.Next
	}
	slow.Next = slow.Next.Next

	return dum.Next
}

func swapPairs(head *ListNode) *ListNode {

	if head == nil {
		return nil
	}

	dum := &ListNode{}
	dum.Next = head

	head = dum

	for head.Next != nil && head.Next.Next != nil {
		p1, p2 := head.Next, head.Next.Next

		newHead := p2.Next
		head.Next = p2
		p2.Next = p1
		p1.Next = newHead

		head = p1

	}

	return dum.Next
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {

	pa := headA
	pb := headB

	for pa != nil && pb != nil {

		if pa == nil {
			pa = headB
		} else {
			pa = pa.Next
		}
		if pb == nil {
			pb = headA
		} else {
			pb = pb.Next
		}

	}

	return nil
}
