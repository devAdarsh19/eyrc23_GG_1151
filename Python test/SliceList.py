class Solution:
    def ReverseList(self, lst):
        # rev_lst = []
        for i in range(len(lst)-1, -1, -1):
            print(lst[i], end=' ')

    def AddThree(self, lst):
        final_lst = []
        for i in range(1, len(lst)):
            temp = lst[i]
            if i % 3 == 0:
                temp += 3
                final_lst.append(temp)

        for val in final_lst:
            print(val, end=' ')

    def SubtractSeven(self, lst):
        final_lst = []
        for i in range(1, len(lst)):
            temp = lst[i]
            if i % 5 == 0:
                temp -= 7
                final_lst.append(temp)

        for val in final_lst:
            print(val, end=' ')

    def SlicedAdd(self, lst):
        sliced_sum = 0
        for i in range(3,8):
            sliced_sum += lst[i]

        print(sliced_sum)

    def AcceptInput(self, T):
        for i in range(T):
            L = int(input())
            lst = list(map(int, input().split(' ')))

            self.ReverseList(lst)
            print(sep='\n')
            self.AddThree(lst)
            print(sep='\n')
            self.SubtractSeven(lst)
            print(sep='\n')
            self.SlicedAdd(lst)


    
sol = Solution()
T = int(input())
sol.AcceptInput(T)
