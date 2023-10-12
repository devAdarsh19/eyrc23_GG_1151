class Solution:
    def IFsandFORs(self, n):
        inp_lst = []
        output_lst = []
        for i in range(n):
            inp_lst.append(int(input('Enter value to iterate : ')))

        for val in inp_lst:
            output_lst = [0] * val
            for i in range(val):
                temp = i
                if (i == 0):
                    temp += 3
                    output_lst[i] = temp
                elif (i % 2 == 0):
                    temp *= 2
                    output_lst[i] = temp
                else:
                    temp = temp**2
                    output_lst[i] = temp
                print(output_lst[i], end=' ')
            print(sep='\n')

sol = Solution()
n = int(input('Value of n : '))
sol.IFsandFORs(n)

